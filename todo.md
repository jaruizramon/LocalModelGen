# 3DModelGen — TODO

## Current state (working)

- Generation: ~30 s, one-piece GLB, 3.9 GB peak VRAM, back to 8 MB after.
- Stack: Go orchestrator (`3dgen` :8080) + Python worker (:8001) + Gradio UI
  (:7860) + WebGL viewer + polygon crop + smooth-mesh toggle + `.blend` out.
- HW auto-detect + offload ratio delivered: `tier=mid` (8 GB VRAM / 61.9 GB
  DRAM), ratio slider in both UIs.
- Everything runs from `LocalModelGen/`; `~/trellis` = trellis package +
  symlinks only.
- Mesh post-processing runs as the `bin/meshproc` C/C++ subprocess
  (repair → dedegen → decimate → component cleanup → bilateral smooth →
  orient) and hits the requested face target exactly.
- Worker endpoints are sync (threadpool): `/health`/`/status` answer during a
  job; `_job_lock` is a real lock; `tmp/` is retention-pruned
  (`LMG_RETENTION_DAYS=14`).

## Blocked

### WebGL viewer — headless render check
**Why blocked**: confirmed still blocked. Headless Firefox fails with
`[GFX1-]: RenderCompositorSWGL failed mapping default framebuffer, no dt` and
produces no screenshot; there is no Chrome/Chromium for a puppeteer driver, and
no capture tool at all (`gnome-screenshot`/`grim`/`import`/`scrot`/`maim` all
absent) even though a live Wayland session exists. `xvfb` is still not
installed. Static checks pass: `/static/{viewer,three.module,GLTFLoader,
OrbitControls}.js` all 200, importmap + canvas present, CORS open, `.glb`
fetch works. **Viewer logic is covered by a behavioral test instead** — real
three.js objects against a byte-identical copy of `viewer.js`, 11/11 assertions
(idempotent init, full material+texture disposal, load-race token, teardown).
## On V100 arrival (Volta sm_70 — no bf16, no TF32; 16/32 GB)

Context from 2026-08-27 A/Bs on the 8 GB RTX 3070: 256³ decode is genuinely
over-budget fp32 there (~7.3 GB peak); it only runs via bf16 field
(`LMG_FIELD_DTYPE=bf16`) + int32 dense-grid index tables + expandable_segments
(torch 5.6 GB / card 7.15 GB, ~30 s). A V100 has none of the bf16 path but
doesn't need it — verify that instead.

- **Verify the `ultra` tier on real hardware**: `hwprofile.py` selects
  `subsample_res=256` fp32 above 16 GB. Measure decode peak/timing on full
  fp32-256³ dense grids (the 3070 never ran true fp32-256; it OOM'd at
  ~7.3 GB peak). Record the numbers in AGENT.md. If 16 GB is tight with a
  desktop session, the int32 index-table patch + `expandable_segments:True`
  still apply (both architecture-neutral) — try them before touching dtype.
- **Fragmentation re-check at true fp32-256³**: the 3070 showed 8–33
  components at 96–256³ regardless of res. If the V100's full-resolution
  decode also fragments, the PyMeshFix `repair(joincomp=True,
  remove_smallest_components=False)` post-repair stage is required there too;
  if it doesn't, the 3070's fragmentation was partly a subsampled-field
  artifact — worth knowing either way.
- **bf16 is NOT available on Volta** (`torch.cuda.is_bf16_supported()` False;
  PyTorch needs CC ≥ 8.0). If 256³ ever needs memory relief on the V100:
  use fp16 (native tensor cores — near-zero-SDF shatter caveat documented in
  AGENT.md) or the int32/ES levers. Do NOT rely on `LMG_FIELD_DTYPE=bf16`.
- **Optional precursor (on the 3070, before the V100 matters)**: A/B
  true-bf16 vs bf16-rounded-fp32 at 160³ (fits either way) to isolate the
  8-bit-mantissa effect from the memory win — tells us whether bf16's only
  cost is mantissa precision, which informs the V100 fp16-vs-emulation choice.
- **If bf16 *numerics* are ever required on the V100**: emulate — store fp32,
  round to the bf16 value grid (round-to-nearest: add 0x8000 bias, mask low
  16 bits; or stochastic rounding for unbiased results). Research-only: slower
  than fp32, zero memory savings.

## Done

- [x] **Worker event-loop stall — FIXED.** `/generate` (and `/convert`) were
      `async def` with fully blocking bodies (CUDA sampling, render, bake,
      blender child), freezing the uvicorn loop for the whole ~30 s job so
      `/health`/`/status` could not answer (and `busy` was unobservable). Now
      sync defs (threadpool); `_job_lock` is a real `threading.Lock`; `/clear`
      guards with `locked()`. Verified: `/health` answered 149× during a
      33.6 s job (busy in 33 checks), zero failures; validation 400s land
      without touching the lock.
- [x] **Mesh branch wired to `bin/meshproc` — DONE.** The whole post-process
      (repair → dedegen → decimate → cleanup → smooth → orient) runs as one
      C/C++ subprocess. Fixed first: `--smooth` moved AFTER the post-decimate
      cleanup (was smoothing against garbage topology), and
      `cm_keep_components` got a keep-largest fallback (was annihilating a
      shattered mesh to a valid-looking empty CMESH; unit-tested with a
      60-tiny-comp + 80-face-strip input at frac 0.5 and 0.6 — both keep 80).
      Deviations from this plan: `--decimate cap`, not `cap*4` — the GH
      target is the face count (datapoint: 12000 → 11999; the old Python path
      at 3000 gave 8777 on shattered geometry), so `cap` is contract-correct —
      and `--smooth smooth_iters` (parameter-faithful). Verified on-card:
      shape input 29.4 s / 2913 faces / same bbox / 27 GLB comps (Python
      baseline 2947-2999 faces, 26-29 comps); `decimate -> 3000 faces` hits
      the target exactly; the pathological gray input overshoots to ~8.8 K
      exactly like the Python decimator (both stop on quality, not by design).
- [x] **Housekeeping — DONE.** git repo initialized (initial commit; sources
      only — `tmp/`, built binaries, `__pycache__` ignored); `tmp/` retention
      pruning (`LMG_RETENTION_DAYS=14`, at startup and after each job);
      `/generate` OBJ/zip converts in a per-job scratch dir (no more
      `material_0.png` clobbering); `/generate` input validation
      (`target_tris >= 0`, `texture_size` in [256, 2048] → 400); dead code
      removed (`tiled_extract.py`, `_stage_mesh_extractor_cuda`,
      `cm_weld`/`cm_dedupe_faces`/`cm_quadric_decimate` + ~290 lines of
      private helpers, orphaned `mesh_smooth.py`, empty `orchestrator/cmd/` +
      `web/`; the `#tiled` checkbox was already gone — this file was stale on
      that); README/AGENT.md drift fixed (run command, env/field lists,
      allocator); allocator decided by measurement: `max_split_size_mb:256`
      (13+ jobs, 3.9-6.1 GB peaks, no allocator-related OOM) — worker and
      runner setdefaults agree, AGENT.md updated.

- [x] **Non-128 subsample tiers wrongly scaled — FIXED.** `tiled_mesh_decode.py`
      subsampled with `coords // (256 // res)`, which is only exact when
      256 % res == 0: `potato` 96 → 128-space lattice / 96 (mesh inflated
      1.33×), `high` 160 → 256-space lattice / 160 (1.6×). Root-cause fix:
      `coords * res // 256`, correct for any res in (0, 256] and bit-identical
      for 128/64/32 (no regression). Verified on-card: 96/128/160 all land in
      the same ~[-0.5, 0.5] model space (bounds x±0.24, y±0.50, z±0.24) with
      flat VRAM floor and time (~3.86 GB torch peak, ~27 s). Caveat:
      fragmentation is res-independent (raw 8–22 comps, GLB 26–29 at every
      res) — 160 is a quality tier, not a fragmentation fix.
- [x] **Offload-ratio OOM — FIXED.** Reported from the UI: "Offload to RAM"
      dragged to 0 (⇒ residency 1.0) died with
      `OutOfMemoryError ... this process has 6.40 GiB memory in use`.
      `keep_resident_list` budgeted from the **nameplate** 8192 MB:
      `hard_cap = 8192 - 4800 - 300 = 3092` approved **3 resident models
      (2368 MB)**, but torch only sees 7.66 GiB and the desktop session holds
      ~330 MB, so the real ceiling was ~6.2 GB. `2368 + 4800 = 7168 MB` vs
      **6234 MB usable = a 934 MB over-commit** — the budget itself authorized
      the OOM. New `hwprofile.usable_vram_mb()` reads
      `torch.cuda.mem_get_info()` (falling back to `nvidia-smi` free when torch
      is absent) and adds back our own reserved bytes so the figure is stable;
      the cap is now `usable - _DECODE_HEADROOM_MB` (`_CONTEXT_MB` is no longer
      subtracted twice — `mem_get_info` already excludes the context).
      *Verified:* every slider position now fits (`1148 + 4800 = 5948 MB` vs
      6305 usable), and the exact reported setting completes in **30.4 s at
      peak 5012 MB** — matching the 5,012 MB datapoint recorded below.
      `/generate` now returns `offload_ratio`, `keep_resident` and
      `usable_vram_mb`, and the Gradio status line shows what was actually kept
      resident, because on an 8 GB card the cap makes the slider saturate at one
      model and a silently-clamped slider looks broken.
- [x] **Gradio arity crash — FIXED.** All three error paths in `image_to_3d`
      returned 3 values against 4 declared outputs
      (`[state, video_output, gen_status, viewer_url]`), so any worker error
      surfaced as `An event handler (image_to_3d) didn't receive enough output
      values (needed: 4, received: 3)` and the real message was **discarded** —
      it masked the OOM above. All four returns now yield 4 values (AST-checked).
      *Verified end-to-end through real Gradio marshalling* (second UI instance
      pointed at a dead worker port): no arity exception, and the UI now
      receives `**Error (generate):** worker unreachable at http://127.0.0.1:9:
      [Errno 111] Connection refused`.

- [x] **GPU leak on exception paths — FIXED and measured.** `run_low_vram` is
      now a thin wrapper (`lib/run_trellis_low_vram.py`) around
      `_run_low_vram_staged`, guarded by a `_gpu_dirty` boolean: set before the
      first model touches the card, cleared only on a normal return. On any
      `BaseException` it calls the new `release_gpu()` (all models → DRAM,
      extractor grids → DRAM, `_KEEP` cleared, `empty_cache` + `synchronize`).
      The worker's cleanup moved out of `except` — where the live traceback
      still pinned the frames and made `empty_cache()` a no-op — into `finally`,
      which nulls every large local first and then releases based on a `_failed`
      boolean. Success path deliberately does NOT force-offload (the outputs are
      live CUDA tensors, and it would defeat the keep-resident ratio).
      *Measured, 3 forced OOMs then 2 normal jobs:* every failure now ends at
      **8 MB allocated / 24 MB reserved** — the idle baseline — with the log
      showing `gpu released (job failed): gpu_alloc=8MB (-2400)`, i.e. 2.4 GB
      reclaimed per failure. Normal jobs after the failures still peak at
      3832/3840 MB, and DRAM RSS drifts +31 MB over 5 jobs instead of stepping
      5.9 → 9.2 → 11.5 GB. Compare the pre-fix log: `job end` at 5309/5137/2403
      MB with the next job starting at 2381 MB.
- [x] **Orphaned processes — FIXED for every signal path.** The orchestrator
      now has an explicit boolean state machine (`workerProc`: `started`,
      `ready`, `signaled`, `exited`, `groupCleaned`), exposed at
      `/api/status` → `worker_proc`, plus:
      - `SysProcAttr{Setpgid: true}` — the worker leads its own process group,
        so `kill(-pgid)` takes down anything it spawned (verified: PGID == PID).
      - a `cmd.Wait()` reaper goroutine — no more zombie, and worker death is
        now logged (`signaled` distinguishes "we killed it" from
        `WARNING: worker exited on its own`).
      - `signal.Notify` on SIGINT/SIGTERM → `srv.Shutdown` then `stopWorker`
        (SIGTERM → grace → SIGKILL), all behind one `sync.Once`.
      - `die()` replaces every `log.Fatal` reachable after the worker exists;
        `log.Fatal` skipped teardown and orphaned the worker holding VRAM
        and :8001.
      *Bug the test caught:* returning early from `stopWorker` when `exited` was
      already set left the worker's **children** alive — the process group
      outlives its leader. Hence `groupCleaned` and an unconditional final
      `kill(-pgid, SIGKILL)` sweep (ESRCH = empty group = success).
      *Verified:* SIGTERM, SIGINT and SIGKILL of the orchestrator each leave
      zero processes, zero zombies, including the grandchild; and a
      worker-crash-then-shutdown leaves no orphan
      (`already reaped; sweeping its process group` → `swept leftover children`).
- [x] **Orchestrator SIGKILL orphan — FIXED.** A process-group kill cannot help
      when the orchestrator itself is SIGKILLed/OOM-killed, so the worker now
      runs a `_watch_parent` daemon thread (armed only by `LMG_SUPERVISED=1`,
      which the orchestrator sets) that notices the re-parent and SIGTERMs
      itself. *Verified on the real stack:* `supervisor 232670 is gone
      (reparented to 4226); shutting down` → `gpu released (orphaned)` →
      worker gone in ~5 s, port 8001 free, VRAM back to desktop-only.
- [x] **Subprocess registry with boolean lifecycle flags.** Every child the
      worker spawns is tracked in `_children` with four booleans — `started`,
      `exited`, `killed`, `reaped` (terminal) — so cleanup is idempotent and
      the state is unambiguous. `reaped` is set from `proc.poll()`, never
      assumed. Exposed live at `/status` → `live_children`, and swept by
      `reap_children()` from the job `finally`, `/clear`, the FastAPI shutdown
      hook and `atexit`. Blender's timeout is now `LMG_BLENDER_TIMEOUT`
      (default 180 s) so the kill path is testable.
      *Bug the test caught:* `subprocess.communicate()` **hangs forever** after
      killing a child whose own grandchild inherited the stdout pipe — the
      wrapper shell died on SIGKILL but its `sleep` kept the pipe open, so
      EOF never arrived and a generation request blocked past 400 s with the
      process still running. Fixed with a `/proc`-based `_descendants()` walk +
      `_kill_tree()` (deepest first), and by using `proc.wait()` instead of
      `communicate()` after the escalation, plus explicit pipe closes.
      *Verified:* hanging-blender job now returns in **26 s** with `rc=-15`,
      `live_children: []`, and zero leftover wrapper/grandchild processes.
- [x] **Temp-directory leaks — FIXED.** `/convert` (worker) and the Gradio
      converter now build in a throwaway scratch dir, move the produced zip out,
      and `rmtree` on every exit path; the client-supplied filename is
      `basename`d (it was a path-traversal). *Verified:* 3 consecutive converts
      leave 0 `ct_*` dirs. This also retires the vacuous "temp-file cleanup"
      claim below.
- [x] **Client-side GPU leak — FIXED.** `static/viewer.js` disposed geometry and
      `m.map` only; materials and every other texture slot (normalMap /
      emissiveMap / metalnessRoughness, all present in TRELLIS bakes) leaked on
      each auto-load, and the UI reloads whenever the newest result changes.
      Now: `disposeObject` walks all `isTexture` material properties and
      disposes the material; an `initialized` boolean makes `initViewer`
      idempotent (the Gradio retry loop was building a second WebGLRenderer per
      attempt); a monotonic `loadToken` drops superseded in-flight loads
      (disposing what they already uploaded); and `disposeViewer()` removes the
      resize listener, stops the animation loop and releases the GL context.
- [x] **Resource-leak hardening in the orchestrator.** Dedicated HTTP clients
      with finite timeouts (`probeClient` 3 s, `ctlClient` 2 min, `jobClient`
      30 min) replace the no-timeout `http.DefaultClient` — a wedged worker used
      to pin the job flag forever (**permanent 429** until restart); the generate
      proxy is now bound to `r.Context()` so a client disconnect releases it.
      `requirePOST` guards `/api/generate` and `/api/clear` — a GET or browser
      prefetch used to **take the GPU job lock** (verified: 405, `job_running`
      stays false). `nvidia-smi` runs under `exec.CommandContext` (3 s) so a
      wedged probe cannot accumulate; the watchdog takes one sample per line
      instead of forking twice per log message and stops via context. The worker
      log's `os.OpenFile` error is checked (it was silently handing the child a
      nil `*os.File`) and the handle is closed by the reaper. `/download/` now
      shares the `/api/results` extension whitelist — it previously served
      `tmp/worker.log` under `Allow-Origin: *` (verified: 404 vs 200 for a
      `.glb`).

- [x] **C decimator divergence — ROOT CAUSED AND FIXED.** The gcc-15 hypothesis
      was **wrong**. `vendor/Simplify.h` and `vendor/wrapper.h` are byte-identical
      to the pip wheel's (`cmp` → identical), and the decimator args match
      exactly (target face count, agg 7.0, `preserve_border=false`). The real bug
      was one line in each C++ driver: they read faces back with
      `Simplify::get_faces_int32`, which emits **VTK-padded 4-int cells**
      `[3,v0,v1,v2]` (`wrapper.h:138-153`), while consuming the buffer at
      **stride 3**. The wheel calls `get_faces_int32_no_padding`
      (`fast_simplification/simplify.py:168`). Reading a stride-4 stream 3-wide
      makes exactly 1 face in 4 correct and injects the literal index `3` into
      the rest, turning vertex 3 into a mega-hub — which is precisely the
      "882 non-manifold edges → 223 shards" signature. Same line was also a heap
      OOB write whenever the target exceeded 75 % of the input face count.
      Fixed at `cxx/fsdecimate_cli.cpp:49-56` and `cxx/meshproc.cpp:86-90` by
      switching to the unpadded accessor and taking the face count from **its**
      return value (`n_triangles()` is only valid after `compact_mesh` and says
      nothing about stride).
      *Measured, icosphere 2562 v / 5120 f → target 1000:*
      | path | verts | faces | comps | non-manifold | degenerate |
      |---|---|---|---|---|---|
      | `fsdecimate` before | 502 | 1000 | **604** | — | 16 |
      | `fsdecimate` after | 502 | 1000 | **1** | **0** | **0** |
      | pip wheel | 502 | 1000 | 1 | 0 | 0 |
      *And on real TRELLIS geometry (`--repair --dedegen --decimate 12000
      --cleanup 0.01 --smooth 3`), `bin/meshproc` and the live Python path now
      agree exactly: 6879 v / 11999 f / 36 comps / 1709 non-manifold / 0
      degenerate.* The "identical on a clean icosphere" datapoint that misled the
      original diagnosis was a false negative: `meshproc.cpp:80` gates decimation
      on `nfaces > decimate`, so a control run with target ≥ face count passes
      through untouched.
- [x] `orchestrator/cmesh/Makefile` — the build recipe existed nowhere and had
      to be reconstructed. `cmesh.c` **must** be compiled by `gcc` as C11 (g++
      rejects its implicit `void*` casts), and the build must **not** use
      `-ffast-math`: `Simplify.h`'s `normalize()` has its zero-guard commented
      out, so degenerate triangles yield NaN quadrics, and the main loop relies
      on IEEE NaN compares being false.
- [x] C kernels + winding orient (`orchestrator/cmesh/cmesh.c`)
- [x] `bin/meshproc` unified C/C++ mesh subprocess (repair → dedegen →
      decimate → cleanup → smooth → orient)
- [x] `bin/fsdecimate` standalone decimator (this is what isolated the bug —
      keep it)
- [x] C hash bug fixed (`hmap_put` now updates existing keys, `cmesh.c:47-52`).
      Note: this silently changed `cm_weld`'s semantics (each grid cell's stored
      representative is now the *last* vertex inserted, not the first) — matters
      only if `cm_weld` is ever re-enabled.
- [x] Subprocess-only rule (in-process cgo corrupts the heap on this box);
      cgo wrappers + dead bridges removed — verified, zero `import "C"` in the
      single Go file
- [x] HW auto-detect (`lib/hwprofile.py`): potato ≤4 GB / mid ≤8 GB / high;
      settings exposed in `/api/status` (`worker.hw`). Caveat: only
      `HW['subsample_res']` is actually consumed (`worker.py:118`) — the tier's
      `max_faces` / `texture_size` / `ss_steps` / `slat_steps` are advisory-only
      dead settings, and see Next action 3 for the res bug.
- [x] Offload ratio: sampling models stay GPU-resident by ratio;
      `LMG_OFFLOAD_RATIO` env default. Caveats: `sparse_structure_decoder` *is*
      in the keep list despite the "decoders excluded" comment
      (`hwprofile.py`). The nominal-VRAM `hard_cap` that let a budget-approved
      residency OOM is **fixed** — see "Offload-ratio OOM" above. `/status`
      still reports only `DEFAULT_RATIO`, never the per-job override, though
      `/generate` now returns the effective `offload_ratio` + `keep_resident`.
- [x] UI: Gradio "Offload to RAM" slider + orchestrator "GPU residency" input.
      Note the two are **inverted** conventions for the same number
      (`webui_gradio.py:80` sends `1 - offload_to_ram`; `index.html:63` passes
      residency straight through).
- [x] Known-good Python mesh path active (29.6 s, 2 comps)
- [x] ~~Temp-file cleanup restored in the runner~~ — was **vacuous** when
      reviewed (no `os.remove` / `rmtree` / `mkdtemp` anywhere in `lib/` or
      `worker/`); the runner still creates no temp files, its `_write_mesh_bin`
      writers being dead code. Now genuinely true for the `/convert` paths, which
      use throwaway scratch dirs, AND for `/generate`'s OBJ conversion (per-job
      scratch dir). Output retention is handled by `LMG_RETENTION_DAYS` pruning.

## Key files

| Path | Role |
|---|---|
| `lib/run_trellis_low_vram.py` | offload runner; mesh post-process via `bin/meshproc` subprocess |
| `lib/hwprofile.py` | detection + tiers + keep-resident budget |
| `lib/tiled_mesh_decode.py` | single-grid 128³ decode (LIVE) |
| `worker/worker.py` | FastAPI worker; sync endpoints (threadpool), `/status` exposes `hw` + `offload_ratio` |
| `orchestrator/cmesh/` | C kernels (`cmesh.c/h`), vendored `Simplify.h`, `cxx/meshproc.cpp`, `cxx/fsdecimate_cli.cpp`, `Makefile` |
| `bin/meshproc`, `bin/fsdecimate` | built subprocesses (`make -C orchestrator/cmesh`) |
| `orchestrator/3dgen` | Go orchestrator (embed UI, static, CORS, watchdog) |
