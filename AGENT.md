# AGENT.md — TRELLIS v1 on RTX 3070 8GB / 64GB RAM

Setup for running [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) v1
(image-to-3D) on this machine. **Verified end-to-end 2026-08-24** (torch 2.4.0,
CUDA 11.8, driver 595.84): full pipeline produces GLB/PLY/videos at peak
~6.5GB VRAM.

## Hardware / OS facts that shape everything

- GPU: RTX 3070, **8 GB VRAM** (official minimum is 16 GB — must offload)
- RAM: 60 GB (holds all ~3.4 GB of weights; models live on CPU between stages)
- OS: modern Ubuntu (kernel 7.0), **gcc 15.2, glibc 2.4x, no sudo, no conda
  originally, no system CUDA toolkit**
- Driver 595.84 (any recent driver runs CUDA 11.8 apps fine)

Consequence: the pinned stack (torch 2.4.0 cu118 + nvcc 11.8) is 2+ years older
than the OS. Four hard incompatibilities had to be worked around (below).

## Current install layout

| Path | What |
|---|---|
| `~/miniconda3` | Miniconda (conda 26) |
| `~/miniconda3/envs/trellis` | env: python 3.10, torch 2.4.0+cu118, xformers, spconv-cu118, kaolin 0.18, nvdiffrast 0.4 (built), diff-gaussian-rasterization (built), flash-attn NOT installed |
| `~/trellis` | microsoft/TRELLIS clone (flexicubes submodule) — hosts the `trellis` package only; the shared inference lib lives in `~/Documents/LocalModelGen/lib` (single source of truth) |
| `~/cuda118` | CUDA 11.8 toolkit, extracted without root (build-time only) |
| `~/.cache/huggingface`, `~/.cache/torch` | weights (~3.4 GB) + DINOv2 |

## Quick start (already installed)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate trellis
cd ~/trellis
python run_trellis_low_vram.py your_image.png
# -> sample_gs.mp4 sample_mesh.mp4 sample.glb sample.ply
```

Peak VRAM ~6.5 GB (mesh decode), idle 1.1 GB after, no leak. ~2–6 min/asset
at 8 sampling steps.

## The four incompatibility fixes (needed for any rebuild)

### 1. torch 2.4.0 import crash: `undefined symbol: iJIT_NotifyEvent`

Neither conda nor pip torch 2.4.0 cu118 wheels bundle `libittnotify.so.0`
(`libtorch_cpu.so` has undefined VTune symbols, no DT_NEEDED provider).

Fix (already applied):
```bash
git clone --depth 1 https://github.com/intel/ittapi /tmp/ittapi
gcc -shared -fPIC -o libittnotify.so.0 /tmp/ittapi/src/ittnotify/jitprofiling.c -I/tmp/ittapi/include
cp libittnotify.so.0 ~/miniconda3/envs/trellis/lib/
mkdir -p ~/miniconda3/envs/trellis/etc/conda/activate.d
printf 'export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so.0${LD_PRELOAD:+:$LD_PRELOAD}\n' \
  > ~/miniconda3/envs/trellis/etc/conda/activate.d/ittnotify.sh
```

### 2. nvcc 11.8 vs gcc 15 / glibc 2.4x

nvcc's host pass cannot parse gcc-15 C++ headers (`_Float32` undefined,
`type_traits` errors) and its version guard rejects gcc > 11. Patching
`host_config.h` is NOT enough — real header breakage remains.

Fix (already applied): conda-forge gcc 11.4 used as nvcc's host compiler.
torch's cpp_extension does **not** pass `-ccbin`; nvcc picks plain `g++` from
PATH — so symlink the namespaced conda compiler into the env bin:
```bash
conda install -y -c conda-forge gxx_linux-64=11.4
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ $CONDA_PREFIX/bin/g++
ln -sf $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc $CONDA_PREFIX/bin/gcc
ln -sf $CONDA_PREFIX/bin/g++ $CONDA_PREFIX/bin/c++
```
Plus a complete CUDA 11.8 toolkit **without sudo**: the installer runfile's
`--tar` mode extracts a mergeable tree (bypasses `cuda-installer`, which needs
`libxml2` and bash quirks):
```bash
cd ~ && bash cuda_11.8.0_520.61.05_linux.run --tar xf   # -> cuda118tar/
for d in ~/cuda118tar/builds/*; do
  [ -d "$d/bin" ] && cp -rn "$d/bin/." ~/cuda118/bin/ 2>/dev/null
  [ -d "$d/targets/x86_64-linux" ] && cp -rn "$d/targets/x86_64-linux/." ~/cuda118/ 2>/dev/null
done
cp -rn ~/cuda118tar/builds/cuda_nvcc/nvvm ~/cuda118/   # libdevice lives outside targets/
sed -i 's/__GNUC__ > 11/__GNUC__ > 15/' ~/cuda118/include/crt/host_config.h
```
Build-time env: `export CUDA_HOME=~/cuda118 PATH=~/cuda118/bin:$PATH CXX=$CONDA_PREFIX/bin/g++`.

### 3. setup.sh pip builds fail (nvdiffrast, diff-gaussian-rasterization)

They need torch in the build env; pip's build isolation removes it. Also the
conda `cuda-nvcc` package alone lacks headers (`cuda_runtime.h`, `cusparse.h`)
— hence the full toolkit above. The `cuda-toolkit` conda metapackage is broken
(LICENSE clobber). Build with:
```bash
pip install --no-build-isolation /tmp/extensions/nvdiffrast
pip install --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
```
Everything else in setup.sh is prebuilt wheels:
`. ./setup.sh --basic --xformers --spconv --mipgaussian --kaolin --nvdiffrast`
(no `--flash-attn`, no `--diffoctreerast`; skip radiance-field rendering).

### 4. VRAM: the offload runner + three empirical fixes

`run_trellis_low_vram.py` moves one sub-model to GPU per stage. Beyond the
script, three things were discovered by profiling that are REQUIRED on 8GB:

1. **Pin `Pipeline.device` to cuda** — the property resolves from the first
   model in dict order, which sits on CPU between stages, so inputs would be
   placed on CPU while the active model is on CUDA:
   ```python
   from trellis.pipelines.base import Pipeline
   Pipeline.device = property(lambda self: torch.device('cuda'))
   ```
2. **`@torch.no_grad()` around the whole run** — without it, autograd retains
   ~440 MB of saved activations per transformer block and the gs decoder OOMs
   at ~6.9 GB.
3. **fp16 mesh decode** — the 256³ Flexicubes grids are fp32 by default and
   need ~7 GB. Convert convs/linears to fp16 (norms stay fp32) and half the
   `reg_v` buffer (a plain tensor, invisible to `convert_module_to_f16`):
   ```python
   dec.to(DEV); dec.apply(convert_module_to_f16)
   dec.mesh_extractor.reg_v = dec.mesh_extractor.reg_v.half()
   out = dec(slat.half())[0]   # decoders return a list (one per sample)
   # float() vertices / vertex_attrs / face_normal back for nvdiffrast
   ```
   Plus two one-line library patches (tensors now follow input dtype):
   - `trellis/representations/mesh/utils_cube.py`: `torch.zeros(..., dtype=value.dtype)` in `cubes_to_verts`, `dtype=feats.dtype` in `get_dense_attrs`
   - `trellis/representations/mesh/flexicubes/flexicubes.py`: `dtype=alpha.dtype` / `dtype=beta.dtype` / `dtype=voxelgrid_colors.dtype` for `vd`, `beta_sum`, `vd_color`
   - `trellis/representations/mesh/utils_cube.py` `construct_dense_grid`: index tables as `torch.int32` — `cube_fx8` at 256³ is 1.07 GB as int64 (256³×8), the exact 1024 MiB alloc that OOMs 8 GB cards; index values stay < 2³¹ for res ≤ 1024, so int32 is precision-safe. Halves every dense-grid index tensor.
   Re-apply these if the repo is re-cloned/reset.

Runtime env vars: `ATTN_BACKEND=xformers`, `SPCONV_ALGO=native`,
`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256` — the default, measured on the
128³ path. For `subsample_res=256` use `expandable_segments:True` +
`LMG_FIELD_DTYPE=bf16` (see the 256³ note in Issues; ES leaves ~330 MB more
headroom than max_split_size at the 7.1 GB ceiling). The worker sets the
allocator before importing the runner; the runner's setdefault matches.

## Notes

- kaolin wheel is built for cu121 while torch is cu118 — imports and runs fine
  on this stack (verified).
- flash-attn is unnecessary (xformers path is memory-efficient); a from-source
  build works but takes ~30+ min.
- Samplers: 8 steps at `cfg_strength` 7.5 (ss) / 3.0 (slat) is a good
  quality/speed point; 12 steps is the official default.
- Output mesh was ~15K verts / 30K faces pre-simplification; GLB export uses
  `simplify=0.95, texture_size=1024`.

## Localhost web UI (Gradio)

`webui_gradio.py` (in `~/Documents/LocalModelGen`) — the frontend wired to the
Go-orchestrated worker. Everything runs from LocalModelGen; `~/trellis`
hosts only the `trellis` package (model code) and symlinks back here.

```bash
conda activate trellis && cd ~/Documents/LocalModelGen && python webui_gradio.py
# -> http://127.0.0.1:7860  (run via hub: op=start name=trellis-ui, cwd=LocalModelGen)
```

Verified: `/`, `/config`, `/info` serve 200; the full
image_to_3d → extract_glb → extract_gaussian flow passes in-process; generation
ran end-to-end through the live HTTP API.

### UI dependency pins (hard-won)

gradio 4.44.1 on this machine needs these EXACT versions — newer ones break:

| Package | Pin | Why |
|---|---|---|
| `gradio-client` | **1.2.0** | 1.3.0 crashes on `additionalProperties: true` schemas (`"const" in schema` TypeError) |
| `starlette` | **0.37.2** | 1.x changed `TemplateResponse` signature → gradio 4.44.1 renders fail ("unhashable dict") |
| `fastapi` | **0.111.1** | pairs with starlette 0.37 |
| `websockets` | **12.0** | gradio-client 1.2.0 requires <13 |
| `huggingface_hub` | **1.28.0** | transformers needs `is_offline_mode` (0.x lacks it); gradio's `HfFolder` import is patched out in `gradio/oauth.py` (OAuth unused) |

Also patched: `gradio_client/utils.py` `_json_schema_to_python_type` — skip
recursion when `additionalProperties` is a bool. Re-apply if gradio re-installed.

### API notes

- `/image_to_3d` is fully callable via `gradio_client` (returns state + video).
- `/extract_glb` / `/extract_gaussian` take the session `gr.State`, which is
  **not** exposed to API clients — browser flow only (same as the official app).
- Handlers coerce slider values defensively (`float()`/`int()` — gradio may
  deliver strings).
- `image_to_3d` frees GPU outputs eagerly so extraction has full 8GB. Do NOT
  run two TRELLIS processes concurrently on this GPU (1.5GB server baseline +
  extraction peak ~7GB = OOM).
- OBJ export: the CLI script writes `sample.obj` + `material.mtl` +
  `material_0.png` from the GLB (trimesh 5.x `PBRMaterial` → `baseColorTexture`).

## Desktop app options (researched 2026-08)

- **Trellis Studio** (pwilkin/trellis.cpp) — Tauri desktop app, TRELLIS.2-based
  (16.5GB weights, different stack; GGML/CUDA server).
- **ComfyUI-Trellis2** (visualbruno) — the practical local GUI for TRELLIS.2;
  requires its own wheels + DINOv3.
- **Trellis3D / Microsoft Store app** — Windows-only one-click installers.
- None fit 8GB + Linux better than the setup in this file. TRELLIS image models
  take NO text prompt (image-only); text models are text-only. Prompt-driven
  workflows = text-to-image model (e.g. ComfyUI/SDXL) → TRELLIS.

## GLB -> OBJ converter

`convert_glb_to_obj.py` — standalone .glb → .obj + .mtl + PNG textures
(Blender / three.js OBJLoader-ready). Multi-mesh GLBs get one OBJ per geometry
in subdirs (trimesh names all textures material_N.png). glTF units are meters;
TRELLIS assets are ~[-0.5, 0.5] so 1:1 scale is fine. For WebGL the .glb itself
loads directly via three.js GLTFLoader — OBJ is for Blender/legacy pipelines.
- UI also has a "GLB to OBJ Converter" accordion: upload any .glb, get
  .obj + .mtl + texture (endpoint `/convert_glb_to_obj_ui`, API-testable).

## Issues

### Mesh faceting / jagging (polygons look faceted, flat surfaces rough)
- **Symptoms**: the exported mesh (and the desktop Mesh view) shows faceted,
  jagged polygons — flat regions read as rough/bumpy, and complex objects
  fragment into many disconnected islands (a 128³ decode of a thin/complex
  relief routinely yields 15-28 components; the desktop flags this via a
  "mesh fragmented (N pieces)" hint).
- **Root cause (AI mesh decoder)**: `slat_decoder_mesh` decodes on a subsampled
  **128³** field (`subsample_res=128`, forced by 8 GB). The coarse field can't
  resolve thin/detailed geometry → the FlexiCubes isosurface fragments; the
  quadric decimation + dominant-normal smoothing then leave ~12-16° median
  adjacent-face tilts on surfaces that should be flat. The **256³** decode fixes
  it but is blocked by the decoder's **~5 GB swin-attention workspace**
  (CUDA-only, not cell-proportional, and not CPU-runnable) — see the 256-on-8GB
  notes below.
- **Desktop "Generate Mesh" = `bin/gs2poisson` (screened Poisson, Open3D,
  CPU, ~18 s)** — the density-isosurface path (`bin/gs2mesh`) was measured out
  for sparse splat clouds: max/sum/blurred/dilated fields all yield 5.7–10K
  disconnected islands on thin-relief assets (no connected level set exists),
  while Poisson builds ONE continuous implicit surface. Pipeline: opacity-
  filter floaters → estimate+orient normals (PLY normals are zero placeholders)
  → `create_from_point_cloud_poisson(depth=9)` → crop low-density hallucination
  → quadric-decimate to 60K → keep largest component → plain OBJ. `bin/gs2mesh`
  remains for blob-like dense splat clouds; its Taubin stage had a 2×
  adjacency under-allocation (ASan) fixed 2026-08-27.
- **Open work**: (a) ~~fix the `gs2mesh` field-blur~~ — the "field-blur" never
  existed in the source (doc error); the crash was a **2× adjacency
  under-allocation** in the Taubin stage (`nbr` sized `off[nverts]` while the
  fill writes 6 slots per triangle; ASan-verified heap-buffer-overflow,
  fixed 2026-08-27). Remaining `gs2mesh` quality work: the density field is
  `max()`-scattered — a sum-of-Gaussians field + binary-search level-set would
  smooth the faceting at the source (DreamGaussian/GOF recipes); (b) switch
  marching-tetrahedra → BCC-lattice tets for cleaner topology; (c)
  fragmentation is res-independent (measured 8–33 components at 96–256³) — the
  fix is post-repair (PyMeshFix `joincomp`), not higher resolution.
- **256³ decode on 8 GB (verified 2026-08-27)** — runs with
  `LMG_FIELD_DTYPE=bf16` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` +
  the int32 index-table patch: torch peak 5.6 GB / card 7.15 GB, ~30 s/job.
  Why it works: bf16 keeps fp32's exponent range (fp16 shatters the
  near-zero SDF/deform field) and halves the dense field grids; int32 halves
  `cube_fx8` (the 1.07 GB int64 table at 256³). fp32-256³ alone is genuinely
  over-budget (~7.3 GB peak) — no allocator setting fixes that; ES only
  removes the fragmentation-OOM so the 130 MiB-class failures stop.
