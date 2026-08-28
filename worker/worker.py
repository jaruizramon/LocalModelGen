"""LocalModelGen inference worker.

A FastAPI service that owns the TRELLIS model and does image->3D generation.
The Go orchestrator spawns this process and drives it over HTTP. The model is
loaded ONCE at startup; each /generate reuses it. Single-job at a time (the
orchestrator enforces the GPU concurrency guard).

Run (from LocalModelGen/worker, in the trellis conda env):
    python worker.py [--port 8001]
"""
import os, sys, io, time, gc, shutil, atexit, ctypes, signal, subprocess, threading
from pathlib import Path

# trellis is a git repo (not pip-installed); find it
_trellis = os.environ.get('TRELLIS_REPO', '/home/pipo/trellis')
if os.path.isdir(_trellis):
    sys.path.insert(0, _trellis)
# shared inference lib is the SINGLE source of truth; insert after trellis so
# it always wins (no duplicate copies may shadow it)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))

os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPCONV_ALGO', 'native')
# Deliberate allocator setting (deliberately set BEFORE importing the runner,
# which setdefaults the same value): max_split_size_mb:256 was chosen over
# expandable_segments by measurement -- 13+ jobs on this card peaked at
# 3.9-6.1GB with no allocator-related OOM. See AGENT.md.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:256')

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

import run_trellis_low_vram as core
from convert_glb_to_obj import convert as glb_to_obj
import hwprofile

app = FastAPI(title='3DModelGen worker')
OUT = Path(__file__).resolve().parents[1] / 'tmp'
OUT.mkdir(exist_ok=True)
core.PHASE_FILE = str(OUT / 'phase.txt')  # desktop polls this for live progress

# `_job_lock` serializes jobs. The orchestrator serializes too, but the worker
# must not rely on that: /clear and /convert can arrive from the Gradio UI,
# which talks to this port directly. A real Lock, not a bool: the endpoints
# are sync defs run in a threadpool, so requests can genuinely overlap now.
_job_lock = threading.Lock()
HW = hwprofile.profile()
DEFAULT_RATIO = float(os.environ.get('LMG_OFFLOAD_RATIO', '0.0'))
RETENTION_DAYS = float(os.environ.get('LMG_RETENTION_DAYS', '14'))
# Wall clock allowed for the headless-Blender .blend assembly before it is
# terminated and reaped. Overridable so the kill-and-reap path is testable.
BLENDER_TIMEOUT = float(os.environ.get('LMG_BLENDER_TIMEOUT', '180'))


# ---------------------------------------------------------------------------
# Child-process registry
#
# This worker shells out to headless Blender. Without an explicit lifecycle a
# timed-out or crashed child is left running (holding a GPU context) or lingers
# as an unreaped zombie. Each entry carries four booleans that make the state
# unambiguous, so cleanup is idempotent and shutdown can tell "never started"
# from "running" from "already reaped":
#
#   started -> Popen returned, the PID exists
#   exited  -> the process left the process table on its own
#   killed  -> WE signalled it (timeout or shutdown), as opposed to `exited`
#   reaped  -> wait() has returned; the PID is gone, no zombie. Terminal.
#
# Children are deliberately NOT put in their own session: they stay in this
# process's group so that when the Go orchestrator kills the worker's process
# group, any straggler dies with it instead of being re-parented to init.
_children = {}
_children_lock = threading.Lock()


def _malloc_trim():
    """Return glibc arenas to the OS. Without this the per-job RSS plateau
    steps up (5.9GB -> 9.2GB -> 11.5GB observed)."""
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def _descendants(pid):
    """Every descendant pid of `pid`, shallowest first (Linux /proc walk).

    Needed because signalling only the direct child is not enough: Blender is
    invoked through a wrapper on some installs, and a shell wrapper does not
    forward SIGTERM to its foreground command. The grandchild then survives
    AND keeps the inherited stdout/stderr pipe open.
    """
    seen, stack, out = set(), [pid], []
    while stack:
        p = stack.pop()
        kids = []
        try:
            for tid in os.listdir(f'/proc/{p}/task'):
                try:
                    with open(f'/proc/{p}/task/{tid}/children') as fh:
                        kids += [int(x) for x in fh.read().split()]
                except OSError:
                    pass
        except OSError:
            pass
        for k in kids:
            if k not in seen:
                seen.add(k)
                out.append(k)
                stack.append(k)
    return out


def _kill_tree(pid, sig):
    """Signal a process and all of its descendants, deepest first."""
    for p in reversed(_descendants(pid)):
        try:
            os.kill(p, sig)
        except OSError:
            pass
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def run_child(argv, timeout=180, name=None):
    """Run a subprocess to completion, guaranteeing it is killed and reaped.

    Returns (returncode, stdout, stderr); returncode is None if it never
    exited. Never raises TimeoutExpired and never blocks unboundedly -- a stuck
    child is a logged failure, not a hung request that leaves a process behind.
    """
    name = name or os.path.basename(argv[0])
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    st = {'proc': proc, 'name': name, 'started': True,
          'exited': False, 'killed': False, 'reaped': False}
    with _children_lock:
        _children[proc.pid] = st
    out = err = ''
    try:
        try:
            out, err = proc.communicate(timeout=timeout)
            st['exited'] = True
        except subprocess.TimeoutExpired:
            st['killed'] = True
            core.memlog(f'child {name} (pid {proc.pid}) timed out after '
                        f'{timeout}s -> killing process tree')
            # Whole tree, not just the direct child: otherwise a surviving
            # grandchild holds the pipe write end and communicate() below
            # blocks forever waiting for an EOF that never comes.
            _kill_tree(proc.pid, signal.SIGTERM)
            try:
                out, err = proc.communicate(timeout=10)
                st['exited'] = True
            except subprocess.TimeoutExpired:
                _kill_tree(proc.pid, signal.SIGKILL)
                # Deliberately NOT communicate(): it waits for pipe EOF, which
                # an unkillable descendant can withhold indefinitely. wait()
                # only waits for our direct child.
                try:
                    proc.wait(timeout=10)
                    st['exited'] = True
                except subprocess.TimeoutExpired:
                    core.memlog(f'child {name} (pid {proc.pid}) survived '
                                f'SIGKILL; abandoning')
                _close_pipes(proc)
        return proc.returncode, out or '', err or ''
    finally:
        # wait() is what actually reaps. communicate() already waited on the
        # normal paths; call it unconditionally so `reaped` cannot lie.
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        _close_pipes(proc)
        st['reaped'] = proc.poll() is not None
        with _children_lock:
            if st['reaped']:
                _children.pop(proc.pid, None)


def _close_pipes(proc):
    for s in (proc.stdout, proc.stderr, proc.stdin):
        try:
            if s is not None:
                s.close()
        except Exception:
            pass


def reap_children(sig=signal.SIGTERM):
    """Kill and reap every registered child (and its descendants). Idempotent;
    safe at interpreter exit. Returns the number that were still live."""
    with _children_lock:
        live = [st for st in _children.values() if not st['reaped']]
    for st in live:
        proc = st['proc']
        if proc.poll() is not None:
            st['exited'] = True
        else:
            st['killed'] = True
            _kill_tree(proc.pid, sig)
            try:
                proc.wait(timeout=5)
                st['exited'] = True
            except Exception:
                _kill_tree(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=5)
                    st['exited'] = True
                except Exception:
                    pass
        _close_pipes(proc)
        st['reaped'] = proc.poll() is not None
    with _children_lock:
        for pid in [p for p, s in _children.items() if s['reaped']]:
            _children.pop(pid, None)
    return len(live)


atexit.register(reap_children)


def _watch_parent(poll=2.0):
    """Exit if the supervising orchestrator disappears.

    The orchestrator kills our process group on shutdown, which covers every
    catchable exit. It cannot cover its own SIGKILL / OOM-kill -- we would be
    re-parented to init and keep the model, ~1.5GB of VRAM and port 8001 for
    ever. Detect the re-parent and shut down uvicorn cleanly (which runs the
    shutdown hook below, so children are reaped and the GPU is released).

    Only armed when the orchestrator asks for it (LMG_SUPERVISED=1), so a
    hand-launched worker in a shell is unaffected.
    """
    ppid0 = os.getppid()
    while True:
        time.sleep(poll)
        if os.getppid() != ppid0:
            print(f'worker: supervisor {ppid0} is gone (reparented to '
                  f'{os.getppid()}); shutting down', flush=True)
            reap_children()
            try:
                core.release_gpu(reason='orphaned')
            except Exception:
                pass
            # SIGTERM to self: uvicorn handles it and runs the shutdown hook.
            os.kill(os.getpid(), signal.SIGTERM)
            return


if os.environ.get('LMG_SUPERVISED') == '1':
    threading.Thread(target=_watch_parent, daemon=True,
                     name='parent-watch').start()


@app.on_event('shutdown')
def _shutdown():
    """Uvicorn runs this on SIGTERM/SIGINT: leave nothing behind."""
    n = reap_children()
    print(f'worker: shutdown -- reaped {n} live child process(es)', flush=True)
    try:
        core.release_gpu(reason='worker shutdown')
    except Exception:
        pass
    _malloc_trim()


@app.on_event('startup')
def load_model():
    print('worker: loading pipeline (one-time)...', flush=True)
    # core imports already loaded the pipeline; force a cuda touch to warm up
    torch.cuda.init()
    torch.cuda.empty_cache()
    _prune_outputs()
    print(f'worker: ready. GPU in use {torch.cuda.memory_allocated()/2**20:.0f} MB', flush=True)


@app.get('/health')
def health():
    return {'ok': True, 'loaded': hasattr(core, 'pipeline'),
            'gpu_mb': int(torch.cuda.memory_allocated() / 2**20),
            'dram_rss_mb': int(core._rss_mb()), 'busy': _job_lock.locked(),
            'live_children': len(_children)}


@app.get('/status')
def status():
    try:
        gpu = int(torch.cuda.memory_allocated() / 2**20)
        resv = int(torch.cuda.memory_reserved() / 2**20)
    except Exception:
        gpu = resv = 0
    with _children_lock:
        kids = [{'name': s['name'], 'pid': p, 'started': s['started'],
                 'exited': s['exited'], 'killed': s['killed'],
                 'reaped': s['reaped']} for p, s in _children.items()]
    return {'loaded': True, 'gpu_mb': gpu, 'gpu_resv_mb': resv,
            'dram_rss_mb': int(core._rss_mb()), 'busy': _job_lock.locked(),
            'phase': getattr(core, 'LMG_PHASE', 'idle'),
            'hw': HW, 'offload_ratio': DEFAULT_RATIO,
            'live_children': kids}


@app.post('/clear')
def clear():
    # A clear during a job would yank the models out from under the running
    # stage. The orchestrator does not serialize /clear (and the Gradio UI hits
    # this port directly), so the guard has to live here. `locked()` is a
    # best-effort pre-check: the real exclusion is generate's lock acquire.
    if _job_lock.locked():
        return JSONResponse({'error': 'a generation is running; clear refused'},
                            status_code=409)
    core.memlog('clear start')
    core.release_gpu(reason='/clear')
    reap_children()
    _malloc_trim()
    core.memlog('clear done')
    return {'cleared': True, 'gpu_mb': int(torch.cuda.memory_allocated() / 2**20)}


def _snap_pow2(n):
    """Round a texture size down to the nearest power of two, clamped to
    [256, 2048]. nvdiffrast's mip-map construction cannot downsample an odd
    extent, so a non-power-of-two size (e.g. 1026) raises at bake time."""
    n = int(n)
    n = max(256, min(2048, n))
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _prune_outputs():
    """Retention: delete generated outputs older than RETENTION_DAYS so tmp/
    cannot grow without bound (~30 MB/job; nothing pruned it before). Only
    whitelisted result extensions are touched; phase.txt / worker.log stay."""
    cutoff = time.time() - RETENTION_DAYS * 86400
    removed = 0
    for f in OUT.iterdir():
        if not f.is_file():
            continue
        if f.name in ('phase.txt', 'worker.log'):
            continue
        if f.suffix.lower() not in ('.glb', '.ply', '.obj', '.zip', '.mp4',
                                    '.blend', '.png', '.mtl'):
            continue
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        core.memlog(f'pruned {removed} stale output(s) > {RETENTION_DAYS:.0f}d')


@app.post('/generate')
def generate(
    image: UploadFile = File(...),
    model: str = Form("trellis-image"),
    seed: int = Form(1),
    ss_steps: int = Form(8),
    slat_steps: int = Form(8),
    ss_cfg: float = Form(7.5),
    slat_cfg: float = Form(3.0),
    target_tris: int = Form(50000),
    texture_size: int = Form(1024),
    subsample_res: int = Form(-1),
    smooth_mesh: bool = Form(True),
    smooth_iters: int = Form(10),
    mesh_cleanup: bool = Form(False),
    offload_ratio: float = Form(-1.0),
):
    if model != "trellis-image":
        return JSONResponse({"error": f"unsupported model: {model}"}, status_code=400)
    # Validate before touching the GPU or the lock: a negative target_tris
    # would reach quadric_simplify(target_count=<0), and an unbounded
    # texture_size squares the bake VRAM (the UI caps at 2048; the API must
    # reject it, not silently clamp).
    if target_tris < 0:
        return JSONResponse({'error': 'target_tris must be >= 0'}, status_code=400)
    if not (256 <= texture_size <= 2048):
        return JSONResponse({'error': 'texture_size must be in [256, 2048]'},
                            status_code=400)
    global _job_lock
    if not _job_lock.acquire(blocking=False):
        return JSONResponse({'error': 'a generation is already running'}, status_code=429)
    core._set_phase('generating')
    core.memlog('job start')
    torch.cuda.reset_peak_memory_stats()
    # Bind every large local up front and track failure explicitly. The finally
    # block below drops these references before freeing: inside `except ... as
    # e` the live traceback still pins this frame, so an empty_cache() there
    # frees nothing (measured: job end at 5309MB allocated, next job starting
    # at 2381MB). `_failed` decides whether a full offload is needed -- on the
    # success path the models are already back in DRAM and force-offloading
    # would defeat the keep-resident offload ratio for the next job.
    outputs = video = mesh = glb = raw = img = None
    keep = ()
    _failed = False
    try:
        gc.collect(); torch.cuda.empty_cache()
        raw = image.file.read()
        img = Image.open(io.BytesIO(raw))
        t0 = time.time()
        ratio = offload_ratio if offload_ratio >= 0 else DEFAULT_RATIO
        # The list is capped against VRAM actually available, so a high ratio
        # can legitimately resolve to fewer models than asked for (or none).
        # Report it back: a slider that silently clamps looks broken.
        keep = hwprofile.keep_resident_list(ratio, HW['vram_mb'])
        # Mesh decode resolution: 0/-1 = auto from the HW tier (128 on 8 GB).
        # Clamped to the 256^3 field's native resolution; big cards (V100) can
        # raise it via UI/API for a finer, less-fragmented mesh.
        if subsample_res <= 0:
            subsample_res = HW['subsample_res']
        subsample_res = max(96, min(256, subsample_res))
        outputs = core.run_low_vram(
            img, seed=seed, ss_steps=ss_steps, slat_steps=slat_steps,
            ss_cfg=ss_cfg, slat_cfg=slat_cfg,
            formats=['gaussian', 'mesh'], target_tris=target_tris,
            smooth_mesh=smooth_mesh, smooth_iters=smooth_iters,
            keep_resident=tuple(keep),
            subsample_res=subsample_res, mesh_cleanup=mesh_cleanup,
        )
        # render preview video (gaussian splat — the mesh's normal-shaded
        # render looked worse; the smooth_mesh toggle only affects the mesh)
        core._set_phase('rendering preview')
        video = core.render_utils.render_video(outputs['gaussian'][0], num_frames=90)['color']
        core.memlog('post: video rendered')
        stamp = int(time.time())
        video_path = OUT / f'gen_{stamp}.mp4'
        core.imageio.mimsave(str(video_path), video, fps=15)
        # save GLB
        mesh = outputs['mesh'][0]
        # Bypass to_glb's postprocess_mesh (simplify / fill_holes / 1000-view
        # invisible-face removal): the mesh is already decimated + fragment-
        # cleaned in run_low_vram, and the postprocess re-fragments it.
        core.postprocessing_utils.postprocess_mesh = lambda v, f, **kw: (v, f)
        core._set_phase('baking texture')
        glb = core.postprocessing_utils.to_glb(
            outputs['gaussian'][0], mesh, simplify=0.0, texture_size=_snap_pow2(texture_size),
            verbose=False)
        core.memlog('post: glb baked')
        glb_path = OUT / f'gen_{stamp}.glb'
        # postprocess_mesh's invisible-face removal re-fragments the tiled
        # mesh; re-keep the coherent components after baking (visuals follow
        # update_faces automatically).
        import trimesh.graph as tg
        _comps = tg.connected_components(glb.face_adjacency, min_len=1)
        if mesh_cleanup and len(_comps) > 1:
            # Only substantial components survive the cleanup (>= 1/3 of largest),
            # not a single tiny sliver on a shattered 128-res mesh.
            _maxlen = max(len(c) for c in _comps)
            _thresh = max(5, _maxlen // 3)
            _keepf = np.sort(np.concatenate([np.array(sorted(c)) for c in _comps if len(c) >= _thresh]))
            glb.update_faces(_keepf)
            glb.remove_unreferenced_vertices()
            core.memlog(f'glb post-cleanup: {len(_comps)} -> {sum(1 for c in _comps if len(c) >= _thresh)} comps')
        else:
            _minf = max(5, int(len(glb.faces) * 0.01))
            _keep = [c for c in _comps if len(c) >= _minf]
            if 0 < len(_keep) < len(_comps):
                _keepf = np.sort(np.concatenate([np.array(sorted(c)) for c in _keep]))
                glb.update_faces(_keepf)
                glb.remove_unreferenced_vertices()
                core.memlog(f'glb cleanup: {len(_comps)} -> {len(_keep)} comps')
        # to_glb writes no normals -> flat-shaded facets in Blender. Accessing
        # the property computes angle-weighted (smooth) vertex normals and
        # caches them; the gltf exporter only writes normals present in cache.
        _ = glb.vertex_normals
        glb.export(str(glb_path))
        # save PLY
        ply_path = OUT / f'gen_{stamp}.ply'
        outputs['gaussian'][0].save_ply(str(ply_path))
        # OBJ + zip: convert in a per-job scratch dir so the fixed
        # material_0.png / material.mtl names cannot clobber another job's
        # loose files in tmp/ (zip there, move the zip out, delete the dir).
        objdir = OUT / f'gen_{stamp}_obj'
        zip_path = glb_to_obj(str(glb_path), out_dir=str(objdir), as_zip=True)
        final_zip = OUT / os.path.basename(zip_path)
        shutil.move(str(zip_path), str(final_zip))
        shutil.rmtree(objdir, ignore_errors=True)
        zip_path = str(final_zip)
        core.memlog('post: obj zip done')
        # assemble a ready-to-open .blend via headless Blender (texture + PBR
        # material + smooth shading applied), so no manual import steps.
        # run_child guarantees the process is killed and reaped even if Blender
        # wedges -- a stuck child would otherwise hold a GPU context forever.
        blend_path = OUT / f'gen_{stamp}.blend'
        blender = os.environ.get('BLENDER', '/usr/bin/blender')
        bscript = Path(__file__).resolve().parents[1] / 'blender_apply_material.py'
        _rc, _bout, _berr = run_child(
            [blender, '-b', '-P', str(bscript), '--', str(glb_path), str(blend_path)],
            timeout=BLENDER_TIMEOUT, name='blender')
        if not blend_path.exists():
            core.memlog(f'blend assembly FAILED (rc={_rc}): {_berr[-300:]}')
        else:
            core.memlog('blend assembled')
        del outputs, video; outputs = video = None
        gc.collect(); torch.cuda.empty_cache()
        _malloc_trim()
        core.memlog('post: trim done')
        secs = time.time() - t0
        # bump timestamped names out for download
        return {
            'id': stamp, 'seconds': round(secs, 1),
            'gpu_mb': int(torch.cuda.memory_allocated() / 2**20),
            'peak_mb': int(torch.cuda.max_memory_allocated() / 2**20),
            'glb': str(glb_path), 'ply': str(ply_path),
            'video': str(video_path), 'zip': zip_path,
            'blend': str(blend_path) if blend_path.exists() else '',
            'faces': int(mesh.faces.shape[0]),
            'offload_ratio': ratio,
            'keep_resident': list(keep),
            'usable_vram_mb': hwprofile.usable_vram_mb(),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        _failed = True
        return JSONResponse({'error': f'{type(e).__name__}: {e}'}, status_code=500)
    finally:
        # Drop the big references FIRST, then free. Order matters: see above.
        core._set_phase('idle')
        outputs = video = mesh = glb = raw = img = None
        try:
            image.file.close()   # release the multipart spool file
        except Exception:
            pass
        gc.collect()
        if _failed:
            # Any raise between a stage's to_cuda and its to_cpu strands
            # weights on the card; release_gpu is the unconditional backstop.
            core.release_gpu(reason='job failed')
            _malloc_trim()
        else:
            # A successful bake leaves ~1.1GB on GPU (gaussian/mesh residual +
            # a stage not fully offloaded). With the default full offload (keep
            # empty) that eats the NEXT job's headroom -> OOM. Clear it; only
            # preserve keep-resident models when the offload ratio > 0 asks to.
            if not keep:
                core.release_gpu(reason='job done')
            else:
                torch.cuda.empty_cache()
        # Nothing this job spawned may outlive it.
        _left = reap_children()
        if _left:
            core.memlog(f'job end: reaped {_left} straggler child process(es)')
        _prune_outputs()
        _job_lock.release()
        core.memlog(f'job end (gpu {torch.cuda.memory_allocated() / 2 ** 20:.0f}MB '
                    f'allocated, failed={_failed})')


@app.post('/convert')
def convert(glb_file: UploadFile = File(...)):
    # The workspace is a scratch dir, not an output dir: the produced zip is
    # moved out and the directory is removed on EVERY exit path. Previously
    # these ct_<ts> dirs accumulated in tmp/ forever, each holding a full copy
    # of the uploaded GLB plus the expanded OBJ/MTL/PNG set.
    ws = OUT / f'ct_{int(time.time_ns())}'
    ws.mkdir(parents=True, exist_ok=True)
    _cleaned = False
    try:
        # never trust a client-supplied filename with a path in it
        src = ws / (os.path.basename(glb_file.filename or 'upload.glb') or 'upload.glb')
        with open(src, 'wb') as f:
            f.write(glb_file.file.read())
        z = Path(glb_to_obj(str(src), as_zip=True))
        final = OUT / z.name
        shutil.move(str(z), str(final))   # out of the scratch dir before rmtree
        return {'zip': str(final)}
    except Exception as e:
        return JSONResponse({'error': f'{type(e).__name__}: {e}'}, status_code=500)
    finally:
        try:
            glb_file.file.close()
        except Exception:
            pass
        shutil.rmtree(ws, ignore_errors=True)
        _cleaned = not ws.exists()
        if not _cleaned:
            core.memlog(f'WARNING: convert workspace not removed: {ws}')


if __name__ == '__main__':
    import uvicorn
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8001
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
