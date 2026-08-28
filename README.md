# 3DModelGen

TRELLIS image→3D on an 8 GB GPU (RTX 3070 + 64 GB DDR4), **Go-orchestrated** with
a chunked, VRAM-aware inference worker.

## Layout

```
LocalModelGen/
├── orchestrator/          # Go: worker lifecycle, job serialization, VRAM watch, web UI + REST API
│   ├── main.go
│   ├── index.html         # embedded web UI (go:embed)
│   ├── go.mod
│   ├── 3dgen              # built binary (go build -o 3dgen .)
│   └── cmesh/             # C/C++ mesh kernels + meshproc/fsdecimate sources
├── worker/                # Python/FastAPI inference worker (owns the model)
│   └── worker.py
├── lib/                   # shared inference library (single source of truth)
│   ├── run_trellis_low_vram.py   # per-stage offload runner (weights in DDR4)
│   ├── convert_glb_to_obj.py
│   └── tiled_mesh_decode.py      # single-grid low-VRAM decode (128^3, seam-free)
├── webui_gradio.py        # legacy Gradio UI (kept as an alternative front end)
├── tmp/                   # generated outputs (worker writes here)
└── AGENT.md
```

## Run

The Go binary is the front door — it spawns and supervises the Python worker
(which loads the model once), serializes GPU jobs, and serves the UI.

```bash
# from orchestrator/
source ~/miniconda3/etc/profile.d/conda.sh && conda activate trellis
LD_PRELOAD=~/miniconda3/envs/trellis/lib/libittnotify.so.0 ./3dgen
# -> http://127.0.0.1:8080   (worker model load ~60s on first boot)
```

Environment (defaults in `main.go`): `WORKER_URL` (`http://127.0.0.1:8001`),
`WORKER_PYTHON` (conda env python), `WORKER_DIR` (`../worker`),
`RESULTS_DIR` (`../tmp`), `LISTEN` (`:8080`), `STATIC_DIR` (`../static`).
Worker-side env: `LMG_BLENDER_TIMEOUT` (180 s), `LMG_RETENTION_DAYS` (14,
prunes `tmp/` outputs older than N days at startup and after each job),
`TRELLIS_REPO` (default `/home/pipo/trellis`).

## REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | web UI |
| `/api/status` | GET | orchestrator + worker + VRAM status |
| `/api/generate` | POST (multipart) | image → GLB/PLY/OBJ/video; `seed`, `target_tris`, `texture_size`, `ss_steps`, `slat_steps`, `ss_cfg`, `slat_cfg`, `smooth_mesh`, `smooth_iters`, `offload_ratio` fields; `target_tris`/`texture_size` validated (`>= 0`, `[256, 2048]`) |
| `/api/clear` | POST | force GPU memory back to the driver (also moves extractor grids to DDR4) |
| `/api/results` | GET | list generated files |
| `/download/<file>` | GET | fetch a result |

GPU jobs are serialized in Go (`/api/generate` returns 429 if a job is running);
the worker is single-job too.

## Why Go + Python split

The heavy math (DiT, spconv, FlexiCubes) runs in PyTorch/CUDA — Go cannot beat
cuBLAS. Go is the **orchestrator**: scheduling, concurrency, VRAM budgeting,
worker supervision, chunk-tile scheduling, and serving. The chunk contract
(tile → halo → extract → crop → weld) lives in `tiled_mesh_decode.py`; the Go
layer schedules tiles and streams results.

## Key numbers (verified on this box)

- Model weights (~3.4 GB fp16) live in DDR4; only the active sub-model is on GPU.
- Swin torso peak 1.8 GB; full mesh decode 4.38 GB; **tiled mesh decode 2.03 GB**.
- Generation (8 steps) ≈ 20–25 s after model load; assets ≈ 6 K faces at
  `target_tris=2000`.

## Notes

- The worker needs `LD_PRELOAD` of `libittnotify.so.0` (torch 2.4.0 on this
  box) — set it in the command or environment above.
- The `trellis` package is a git repo (not pip-installed); point `TRELLIS_REPO`
  at it.
- Legacy Gradio UI: `conda activate trellis && python webui_gradio.py`
  (http://127.0.0.1:7860) — use either front end; both drive the same pipeline.
