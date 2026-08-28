"""Hardware auto-detection + per-setup defaults + DRAM offload ratio.

Tiers (VRAM total):
  potato: <= 4 GB  -> lowest subsample res, smaller caps, 512 texture, 6 steps
  mid   : <= 8 GB  -> current 8GB tuning (128^3 subsample, 500K cap, 1024 tex)
  high  : > 8 GB   -> higher quality (160^3 subsample, 800K cap, 1536 tex)

Offload ratio: 0..1 fraction of VRAM spent keeping the most-used sub-models
GPU-resident between stages (0 = full DRAM offload, the safe default). The
ratio maps to a keep-resident model list via measured per-model weight sizes.
"""
import os
import subprocess


def detect_gpu_vram_mb():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def detect_dram_mb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def profile(vram_mb=None, dram_mb=None):
    vram = vram_mb or int(os.environ.get('LMG_VRAM_MB', 0)) or detect_gpu_vram_mb()
    dram = dram_mb or int(os.environ.get('LMG_DRAM_MB', 0)) or detect_dram_mb()
    if vram <= 0:
        vram = 8192
    if dram <= 0:
        dram = 16384
    if vram <= 4096:
        tier = 'potato'
        d = dict(subsample_res=96, max_faces=250000, texture_size=512,
                 ss_steps=6, slat_steps=6)
    elif vram <= 8192:
        tier = 'mid'
        d = dict(subsample_res=128, max_faces=500000, texture_size=1024,
                 ss_steps=8, slat_steps=8)
    elif vram <= 16384:
        tier = 'high'
        d = dict(subsample_res=160, max_faces=800000, texture_size=1536,
                 ss_steps=12, slat_steps=12)
    else:
        # >16 GB (e.g. a V100 32 GB): afford the full 256^3 field -> no
        # subsample, clean high-res mesh. The 128^3 default is a DRAM/VRAM
        # compromise for 8 GB cards; big cards get the real resolution.
        tier = 'ultra'
        d = dict(subsample_res=256, max_faces=1200000, texture_size=2048,
                 ss_steps=12, slat_steps=12)
    return dict(tier=tier, vram_mb=vram, dram_mb=dram, **d)


# measured per-model GPU weight sizes (MB, from memlog stage transfers)
_MODEL_MB = {
    'image_cond_model': 1161,
    'sparse_structure_flow_model': 700,
    'sparse_structure_decoder': 520,
    'slat_flow_model': 1148,
    'slat_decoder_gs': 22,
    'slat_decoder_mesh': 1414,
}
# keep candidates are the SAMPLING models only: the decoders are used once per
# job and are the biggest (keeping them resident costs more than it saves).
_ORDER = ['slat_flow_model', 'sparse_structure_flow_model',
          'sparse_structure_decoder', 'image_cond_model']

# VRAM headroom the decode + bake need beyond resident weights (measured:
# 128^3 mesh decode ~2.2GB + gaussian + texture-bake transients, plus
# allocator fragmentation on a resident-weights layout)
# The GLB texture bake (to_glb) is the VRAM wall: it reserves ~4.4GB
# (render_multiview 100 views + bake optimizer). Resident weights + bake +
# context must fit the card.
_DECODE_HEADROOM_MB = 4800
_CONTEXT_MB = 300


def usable_vram_mb():
    """VRAM this process can actually reach right now, or 0 if unknown.

    Deliberately NOT the nameplate capacity. On a nominal 8192 MB card torch
    reports 7.66 GiB (7844 MB), and a desktop session holds a few hundred MB
    more (nautilus/discord/gnome ~330 MB here). Budgeting residency from 8192
    over-commits by ~700 MB, which is exactly enough to turn a
    budget-*approved* keep-resident set into an OOM during the GLB bake:
    ratio=1.0 used to approve 3 models / 2368 MB and then die at 5.98 GiB
    allocated.

    `mem_get_info()` free already accounts for other processes and the CUDA
    context; adding back our own reserved bytes keeps the answer stable whether
    or not weights happen to be resident when this is called.
    """
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int((free + torch.cuda.memory_reserved()) / 2 ** 20)
    except Exception:
        pass
    # No torch (standalone import): fall back to the driver's free figure.
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return 0


def keep_resident_list(ratio, vram_mb, model_mb=None, usable_mb=None):
    """Sampling models to keep GPU-resident between stages.

    `ratio` in [0,1] is the share of VRAM the caller is willing to spend on
    resident weights. The result is additionally capped so that resident
    weights + the decode/bake wall fit in the VRAM actually available, so no
    slider position can produce an OOM. Returns [] (full offload) when even the
    smallest candidate would not fit.
    """
    if ratio <= 0:
        return []
    usable = usable_vram_mb() if usable_mb is None else usable_mb
    if usable > 0:
        # Real ceiling: mem_get_info already excludes the CUDA context and
        # other processes, so _CONTEXT_MB must NOT be subtracted again here.
        hard_cap = usable - _DECODE_HEADROOM_MB
    else:
        hard_cap = int(vram_mb) - _DECODE_HEADROOM_MB - _CONTEXT_MB
    budget = min(int(vram_mb * min(ratio, 1.0) * 0.8), hard_cap)
    if budget <= 0:
        return []
    mb = dict(model_mb or _MODEL_MB)
    keep, used = [], 0
    for name in _ORDER:
        w = mb.get(name, 0)
        if used + w <= budget:
            keep.append(name)
            used += w
    return keep
