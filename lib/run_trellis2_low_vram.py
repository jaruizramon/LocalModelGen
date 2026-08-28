"""TRELLIS.2 (microsoft/TRELLIS.2-4B) integration reference runner.

PRESERVED for documentation only - the TRELLIS.2 WEIGHTS (~16GB), the trellis2
conda env, the ~/TRELLIS.2 git checkout, the flex_gemm/cumesh/o_voxel kernels
and the nvdiffrast build were removed from storage (see AGENT.md decision).

To bring TRELLIS.2 back you would need ALL of the following, in order:
  1. An HF token whose account was GRANTED access to the gated
     facebook/dinov3-vitl16-pretrain-lvd1689m (gated: MANUAL - Meta reviews
     every request) and briaai/RMBG-2.0 (gated: auto). Without DINOv3 the
     shape/texture flow models (trained on DINOv3 conditioning) produce an
     EMPTY sparse structure -> RuntimeError: max(): input.numel() == 0.
  2. The trellis2 env: torch 2.6+cu124, xformers 0.0.29.post3, nvdiffrast
     (built for cp311), and the flex_gemm/cumesh/o_voxel .so kernels on the
     path plus ~/trellis2 pip node.
  3. Point pipeline.json's image_cond_model at DinoV3FeatureExtractor and
     rembg_model at BiRefNet (a backup of the ORIGINAL config was left as
     pipeline.json.bak before the no-token DINOv2/NoopRembg experiment).

The no-token substitutes (public, cached DINOv2 + a NoopRembg) make the
pipeline LOAD and run fully offloaded to DRAM (gpu MB=0) but only prove the
offload path - they cannot generate a valid asset. Do not expect them to.
"""
import os, sys, gc, time
for p in ['/tmp/flexgemm', '/tmp/cumesh', '/home/pipo/TRELLIS.2', '/home/pipo/trellis']:
    sys.path.insert(0, p)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch, numpy as np
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline

PATH = '/home/pipo/models/TRELLIS.2-4B'
IMG = '/tmp/lmg_test.png'
OUT = '/tmp/t2_test.glb'

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
print('loading pipeline (low_vram)…', flush=True)
pipe = Trellis2ImageTo3DPipeline.from_pretrained(PATH)
pipe.cuda()
pipe.low_vram = True
print('loaded in %.1fs; gpu MB=%d' % (time.time()-t0, torch.cuda.memory_allocated()/2**20), flush=True)

image = Image.open(IMG).convert('RGB')
print('running…', flush=True)
t1 = time.time()
meshes = pipe.run(image, seed=7, pipeline_type='512', max_num_tokens=49152)
print('run in %.1fs; peak MB=%d' % (time.time()-t1, torch.cuda.max_memory_allocated()/2**20), flush=True)

mesh = meshes[0]
print('mesh verts=%d faces=%d' % (mesh.vertices.shape[0], mesh.faces.shape[0]), flush=True)
import o_voxel
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
    coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    decimation_target=200000, texture_size=1024, remesh=True, remesh_band=1,
    remesh_project=0, verbose=False)
glb.export(OUT, extension_webp=True)
print('SAVED', OUT, 'peak MB=%d' % (torch.cuda.max_memory_allocated()/2**20), flush=True)


# ---- Re-add this text to trellis2/pipelines/rembg/BiRefNet.py to use a no-op
# ---- background remover (the gated BiRefNet/RMBG-2.0 was the other blocker):
#
# class NoopRembg:
#     """No-op background remover: keeps the input image (as RGBA with an opaque
#     alpha channel) unchanged. Used when no segmentation model is available
#     (e.g. the gated BiRefNet/RMBG-2.0 download is skipped). Matches the
#     BiRefNet interface so the pipeline runs the same."""
#     def __init__(self, model_name: str = ""):
#         self.model_name = model_name
#     def to(self, device: str): return self
#     def cuda(self): return self
#     def cpu(self): return self
#     def __call__(self, image: Image.Image) -> Image.Image:
#         return image.convert("RGBA")
#
# And pipeline.json:
#   image_cond_model = {'name':'DinoV2FeatureExtractor','args':{'model_name':'dinov2_vitl14_reg'}}
#   rembg_model      = {'name':'NoopRembg','args':{'model_name':''}}
