"""TRELLIS v1 (image-large) on <=8GB VRAM via per-stage CPU offload.

Usage (inside the cloned microsoft/TRELLIS repo, trellis conda env active):

    python run_trellis_low_vram.py your_image.png

Only one sub-model sits on the GPU at a time; all weights stay in system RAM
(64GB is plenty: ~3.4GB of fp16 weights total). Peak VRAM ~= largest single
sub-model + its activations. Two things are load-bearing for 8GB:

- The mesh decoder (256^3 Flexicubes) must run with fp16 convs/linears
  (norms stay fp32, matching the official convert_module_to_f16), else its
  dense grids need ~7GB. Result is converted back to fp32 for export.
- Everything must run under torch.no_grad(): otherwise autograd retains one
  ~440MB block of saved activations per transformer block and OOMs.

Env vars:
  ATTN_BACKEND=xformers         -> skip flash-attn (Ampere OK either way)
  SPCONV_ALGO=native            -> no benchmarking at startup
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 -> avoid 1GB+ contiguous
    allocator blocks on 8GB (deliberate; see the setdefault below)
"""
import os
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPCONV_ALGO', 'native')
# Deliberate single allocator setting: max_split_size_mb:256 (not
# expandable_segments) -- it is the empirically-validated value on this card
# (13+ jobs, peaks 3.9-6.1GB, no allocator-related OOM). The worker sets it
# before importing this module, so this setdefault only matters for direct
# CLI runs; both must agree (see AGENT.md).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')

import sys
import os
import subprocess
import tempfile
import gc
# Self-relocating: lib/ is the single source of truth, but the CLI may be
# invoked through a symlink elsewhere (e.g. ~/trellis/run_trellis_low_vram.py)
# or from any cwd. Resolve to THIS file's real directory for sibling modules
# and add the trellis repo for the trellis package.
_trellis_repo = os.environ.get('TRELLIS_REPO', '/home/pipo/trellis')
if os.path.isdir(_trellis_repo):
    sys.path.insert(0, _trellis_repo)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import torch
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import numpy as np


# C/C++ mesh pipeline (meshproc): decimate + cleanup + smooth run in one
# C/C++ subprocess (in-process cgo C calls corrupt the heap on this box, so
# the math runs in its own process); Python only passes binary mesh dumps.
MESHPROC = os.environ.get('MESHPROC',
                          '/home/pipo/Documents/LocalModelGen/bin/meshproc')
OUT_DIR = os.environ.get('LMG_TMP', tempfile.gettempdir())


def _write_mesh_bin(path, verts, faces):
    with open(path, 'wb') as f:
        f.write(b'CMESH')
        f.write(np.int32(verts.shape[0]).tobytes())
        f.write(np.int32(faces.shape[0]).tobytes())
        f.write(np.ascontiguousarray(verts, dtype='<f4').tobytes())
        f.write(np.ascontiguousarray(faces, dtype='<u4').tobytes())


def _read_mesh_bin(path):
    with open(path, 'rb') as f:
        assert f.read(5) == b'CMESH', 'bad mesh bin magic'
        nv, nf = np.frombuffer(f.read(8), dtype='<i4')
        verts = np.frombuffer(f.read(int(nv) * 3 * 4), dtype='<f4').copy()
        faces = np.frombuffer(f.read(int(nf) * 3 * 4), dtype='<u4').copy()
    return verts.reshape(int(nv), 3), faces.reshape(int(nf), 3)


def _rss_mb():
    """Current process DRAM RSS in MB (Linux /proc/self/status)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


_prev_mem = {}


# Current pipeline phase, surfaced via the worker /status so the UI can show
# live progress ("processing gaussians…", "preparing meshes…", "welding…").
LMG_PHASE = 'idle'
# The sync generate blocks the worker's event loop, so /status can't serve the
# live phase during a job. Write the phase to a file the desktop polls directly.
PHASE_FILE = os.environ.get('LMG_PHASE_FILE', '/tmp/lmg_phase.txt')


def _set_phase(p: str) -> None:
    """Record the current pipeline phase (module global + a status file)."""
    global LMG_PHASE
    LMG_PHASE = p
    try:
        with open(PHASE_FILE, 'w') as f:
            f.write(p)
    except Exception:
        pass


def memlog(tag):
    """Memory snapshot at a stage boundary: GPU allocated/reserved + DRAM RSS,
    with deltas vs the previous call. A persistent +delta across stages is the
    accumulation signature (something not freed / not moved back to DRAM)."""
    ga = torch.cuda.memory_allocated() / 2 ** 20
    gr = torch.cuda.memory_reserved() / 2 ** 20
    rss = _rss_mb()
    d_ga = ga - _prev_mem.get('ga', ga)
    d_rss = rss - _prev_mem.get('rss', rss)
    _prev_mem.update(ga=ga, rss=rss)
    print(f'[mem] {tag}: gpu_alloc={ga:.0f}MB ({d_ga:+.0f}) '
          f'gpu_resv={gr:.0f}MB dram_rss={rss:.0f}MB ({d_rss:+.0f})', flush=True)


DEV = torch.device('cuda')

pipeline = TrellisImageTo3DPipeline.from_pretrained('microsoft/TRELLIS-image-large')
pipeline.cpu()  # keep all weights in the 64GB of system RAM
torch.cuda.empty_cache()
memlog('pipeline loaded (weights in DRAM)')

# The Pipeline.device property resolves from the FIRST model in dict order,
# which we keep on CPU between stages -> inputs would be placed on CPU while
# the active model is on CUDA. Pin it: we always place models deliberately.
from trellis.pipelines.base import Pipeline
Pipeline.device = property(lambda self: torch.device('cuda'))


_KEEP = set()  # models kept GPU-resident between stages (offload ratio)


def _on_gpu(m):
    d = getattr(m, 'device', None)
    if d is not None:
        try:
            return str(d) == 'cuda'
        except Exception:
            pass
    try:
        return next(m.parameters()).is_cuda
    except Exception:
        return False


def to_cuda(*names):
    for n in names:
        if n in _KEEP and _on_gpu(pipeline.models[n]):
            continue
        pipeline.models[n].to(DEV)
    memlog(f'VRAM< {"+".join(names)}')


def to_cpu(*names):
    for n in names:
        if n in _KEEP:
            continue
        pipeline.models[n].cpu()
    torch.cuda.empty_cache()
    memlog(f'DRAM< {"+".join(names)}')



def _mesh_extractor(dec):
    """The SparseFeatures2Mesh grids are plain attrs (not nn.Modules) created
    on CUDA at load -> ~1.4GB GPU resident. Expose them for staging."""
    return getattr(dec, 'mesh_extractor', None)


def _free_mesh_extractor(dec):
    ext = _mesh_extractor(dec)
    if ext is None:
        return
    for attr in ('reg_v', 'reg_c'):
        t = getattr(ext, attr, None)
        if torch.is_tensor(t) and t.is_cuda:
            setattr(ext, attr, t.cpu())
    fc = getattr(ext, 'mesh_extractor', None)
    if fc is not None:
        for n, b in vars(fc).items():
            if torch.is_tensor(b) and b.is_cuda:
                try:
                    setattr(fc, n, b.cpu())
                except Exception:
                    pass
    torch.cuda.empty_cache()


@torch.no_grad()
def _run_low_vram_staged(image: Image.Image, seed: int = 1, steps: int = 8,
                 formats=('gaussian', 'mesh'), ss_steps: int = None,
                 slat_steps: int = None, ss_cfg: float = 7.5,
                 slat_cfg: float = 3.0, target_tris: int = 0,
                 max_faces: int = 500000, smooth_mesh: bool = True,
                 smooth_iters: int = 10, keep_resident: tuple = (),
                 subsample_res: int = 128, mesh_cleanup: bool = False) -> dict:
    image = pipeline.preprocess_image(image)
    torch.cuda.reset_peak_memory_stats()
    global _KEEP
    global LMG_PHASE
    _KEEP = set(keep_resident)
    if _KEEP:
        memlog(f'keep-resident (offload ratio): {sorted(_KEEP)}')
    memlog('run start')
    _set_phase('image conditioning')
    # Stage 1: image conditioning (DINOv2) -> cond tokens
    to_cuda('image_cond_model')
    cond = pipeline.get_cond([image])
    to_cpu('image_cond_model')
    memlog('stage1 image_cond done')
    _set_phase('sparse structure')

    # Stage 2: sparse structure flow (16^3) + occupancy decode
    # Cheap: 1.08GB + 0.14GB weights on GPU together is fine.
    torch.manual_seed(seed)
    to_cuda('sparse_structure_flow_model', 'sparse_structure_decoder')
    coords = pipeline.sample_sparse_structure(
        cond, 1, {'steps': ss_steps or steps, 'cfg_strength': ss_cfg})
    to_cpu('sparse_structure_flow_model', 'sparse_structure_decoder')
    memlog('stage2 sparse structure done')
    _set_phase('structured latent')

    # Stage 3: slat flow (64^3) -- peak VRAM moment
    to_cuda('slat_flow_model')
    slat = pipeline.sample_slat(
        cond, coords, {'steps': slat_steps or steps, 'cfg_strength': slat_cfg})
    to_cpu('slat_flow_model')
    memlog('stage3 slat done')

    # Stage 4: decode each requested format one at a time
    decoders = {
        'gaussian': 'slat_decoder_gs',
        'mesh': 'slat_decoder_mesh',
        'radiance_field': 'slat_decoder_rf',
    }
    outputs = {}
    for fmt in formats:
        dec = pipeline.models[decoders[fmt]]
        _set_phase('decoding gaussians' if fmt == 'gaussian' else 'preparing meshes')
        memlog(f'decode {fmt} start')
        if fmt == 'mesh':
            # Single-grid low-VRAM decode: torso + upsample + out_layer run
            # globally on the 64^3 latent, the 256^3 surface field is
            # subsampled to 128^3, then ONE FlexiCubes extraction. No tile
            # seams -> clean connected manifold. fp32 throughout (fp16 shatters
            # the near-zero SDF/deform field). The 128^3 dense grids (~40MB)
            # fit easily; the big reg_v/reg_c extractor grids are not used
            # (extraction builds local grids via construct_dense_grid).
            dec.to(DEV)
            # FlexiCubes' internal index tables are plain attrs (not nn
            # params), so dec.to() does not move them; after a /clear they sit
            # on CPU and flex() index_selects on CUDA -> device mismatch.
            ext = _mesh_extractor(dec)
            fc = getattr(ext, 'mesh_extractor', None) if ext is not None else None
            if fc is not None:
                for _n, _b in vars(fc).items():
                    if torch.is_tensor(_b) and not _b.is_cuda:
                        try:
                            setattr(fc, _n, _b.cuda())
                        except Exception:
                            pass
            from tiled_mesh_decode import decode_mesh_low_vram
            out = decode_mesh_low_vram(dec, slat, res=subsample_res,
                                       device='cuda')
            out.vertices = out.vertices.float()
            # tiled decode returns vertex_attrs=None: texture comes from the
            # gaussian bake below (same as the quadric-simplified path).
            if out.vertex_attrs is not None:
                out.vertex_attrs = out.vertex_attrs.float()
            out.face_normal = out.face_normal.float()
            # Mesh post-processing: the whole branch (non-manifold repair,
            # degenerate drop, quadric decimation, component cleanup, bilateral
            # smoothing, consistent winding) runs as ONE C/C++ subprocess
            # (bin/meshproc). In-process cgo C calls corrupt the heap on this
            # toolchain, so the math runs in its own process; Python passes
            # CMESH binary dumps over argv paths. Topology-equivalent to the
            # old Python path (fast_simplification + trimesh, verified in
            # todo.md) and it hits the requested face target exactly where the
            # Python decimator overshot (103,906 -> 8,777 at target 3,000).
            cap = target_tris if target_tris else max_faces
            if out.vertices.shape[0] > 0 and out.faces.shape[0] > 0:
                _set_phase('welding mesh')
                inb = os.path.join(OUT_DIR, f'meshproc_{os.getpid()}_in.bin')
                outb = os.path.join(OUT_DIR, f'meshproc_{os.getpid()}_out.bin')
                try:
                    _write_mesh_bin(inb, out.vertices.cpu().numpy(),
                                    out.faces.cpu().numpy())
                    args = [MESHPROC, inb, outb, '--repair', '--dedegen']
                    if cap:
                        args += ['--decimate', str(cap)]
                    args += ['--cleanup', '0.33' if mesh_cleanup else '0.01']
                    if smooth_mesh:
                        iters = smooth_iters if out.vertices.shape[0] < 200000 else 4
                        args += ['--smooth', str(iters)]
                    memlog(f'meshproc: {" ".join(args[3:])}')
                    rc = subprocess.run(args, capture_output=True, text=True)
                    if rc.returncode != 0:
                        raise RuntimeError(
                            f'meshproc failed rc={rc.returncode}: {rc.stderr[-500:]}')
                    if rc.stderr.strip():
                        memlog('meshproc: ' + rc.stderr.strip().replace('\n', ' | '))
                    nv, nf = _read_mesh_bin(outb)
                    out.vertices = torch.tensor(nv, device=DEV, dtype=torch.float32)
                    out.faces = torch.tensor(nf, device=DEV, dtype=torch.long)
                    out.vertex_attrs = None
                    out.face_normal = out.comput_face_normals(out.vertices, out.faces)
                    out.success = out.vertices.shape[0] != 0
                    memlog(f'meshproc done: {nv.shape[0]:,} verts {nf.shape[0]:,} faces')
                finally:
                    for p in (inb, outb):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass
            outputs[fmt] = [out]
            # free the extractor grids immediately: keeps the server baseline at
            # ~CUDA-context size instead of +1.4GB per decoded mesh.
            _free_mesh_extractor(dec)
        else:
            dec.to(DEV)
            outputs[fmt] = dec(slat)
        to_cpu(decoders[fmt])
        memlog(f'decode {fmt} done')
        # Segment the phases: return the just-freed decoder's cached VRAM to
        # the system before decoding the next format, so the gaussian stage's
        # footprint (decoder + renderer) is gone before the mesh stage starts
        # from CUDA-context baseline. Fewer bytes resident -> lower per-stage
        # peak and more headroom for the heavier mesh + bake.
        torch.cuda.empty_cache()
    _KEEP = set()
    _set_phase('done')
    memlog(f'run complete (peak gpu {torch.cuda.max_memory_allocated() / 2 ** 20:.0f}MB)')
    return outputs


# ---------------------------------------------------------------------------
# GPU leak guard
#
# Every stage moves a sub-model to the GPU and moves it back. An exception
# raised between the two leaves that model (and every activation the traceback
# still pins) stranded on the card for the lifetime of the process. Measured
# before this guard, from tmp/worker.log: a failed job ended with 5309MB still
# allocated and the NEXT job started at 2381MB -- the loss was permanent and
# cumulative, shrinking the budget for every later generation.
#
# `_gpu_dirty` is the boolean that makes the cleanup decision explicit: it is
# set before the first model touches the GPU and cleared only after the staged
# runner returns normally. On the success path the outputs are still live CUDA
# tensors that the caller needs (video render + GLB bake), so releasing there
# would be wrong -- hence the flag rather than an unconditional finally.
_gpu_dirty = False


def release_gpu(reason='manual'):
    """Return the card to its idle baseline. Idempotent and never raises.

    Moves every sub-model and the FlexiCubes extractor grids back to DRAM,
    clears the keep-resident set (otherwise `to_cpu` would keep skipping the
    resident models forever) and hands the allocator cache back to the driver.
    """
    global _KEEP, _gpu_dirty
    _KEEP = set()
    try:
        for _name, _m in pipeline.models.items():
            try:
                _m.cpu()
            except Exception:
                pass
        _dec = pipeline.models.get('slat_decoder_mesh')
        if _dec is not None:
            _free_mesh_extractor(_dec)
    finally:
        _gpu_dirty = False
        gc.collect()
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    memlog(f'gpu released ({reason})')


def run_low_vram(*args, **kwargs) -> dict:
    """Leak-guarded entry point. Delegates to the staged runner and guarantees
    the GPU is returned to its idle baseline on every failing exit path."""
    global _gpu_dirty
    _gpu_dirty = True
    try:
        outputs = _run_low_vram_staged(*args, **kwargs)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt or a CUDA OOM
        # escalated to SystemExit must not strand weights on the card either.
        release_gpu(reason='staged run raised')
        raise
    _gpu_dirty = False
    return outputs


if __name__ == '__main__':
    img = Image.open(sys.argv[1])
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    outputs = run_low_vram(img, target_tris=target)

    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave('sample_gs.mp4', video, fps=30)
    if 'mesh' in outputs:
        video = render_utils.render_video(outputs['mesh'][0])['normal']
        imageio.mimsave('sample_mesh.mp4', video, fps=30)
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0], outputs['mesh'][0],
            simplify=0.95, texture_size=1024)
        glb.export('sample.glb')
        # OBJ export: geometry + UVs + material. trimesh does not write the
        # texture file itself, so save it next to the .mtl it references.
        import trimesh
        scene = trimesh.load('sample.glb', force='scene')
        mesh = list(scene.geometry.values())[0]
        mesh.export('sample.obj')
        if getattr(mesh.visual, 'kind', None) == 'texture':
            mat = mesh.visual.material
            img = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
            if img is not None:
                if hasattr(img, 'save'):
                    img.save('material_0.png')
                else:
                    Image.fromarray(img).save('material_0.png')
    outputs['gaussian'][0].save_ply('sample.ply')
    print('done: sample_gs.mp4 sample_mesh.mp4 sample.glb sample.obj sample.ply')
