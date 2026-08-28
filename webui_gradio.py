"""localhost Gradio UI for TRELLIS v1 on <=8GB VRAM — LocalModelGen worker backend.

Frontend only: no model is loaded in this process. Every generate/extract/clear
call is proxied to the LocalModelGen worker (FastAPI on :8001, spawned and
supervised by the Go orchestrator `3dgen` on :8080), which owns the model and
does the DRAM-offload / per-stage-GPU inference.

Run (with the orchestrator stack up, from LocalModelGen/):
    python webui_gradio.py
Then: http://127.0.0.1:7860
(~/trellis/app_low_vram.py is a symlink to this file.)
"""
import os, io, json, sys, time, shutil
from pathlib import Path

# shared inference lib: single source of truth in LocalModelGen/lib
_LIB = os.environ.get('LOCALMODELGEN_LIB', '/home/pipo/Documents/LocalModelGen/lib')
if os.path.isdir(_LIB) and _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import numpy as np
import gradio as gr
import httpx
from PIL import Image, ImageDraw, ImageChops

from convert_glb_to_obj import convert as glb_to_obj

WORKER_URL = os.environ.get('WORKER_URL', 'http://127.0.0.1:8001')
ORCH_URL = os.environ.get('ORCH_URL', 'http://127.0.0.1:8080')
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

_client = httpx.Client(timeout=300)


def _err_status(e, stage):
    if isinstance(e, httpx.HTTPError):
        return f'**Error ({stage}):** worker unreachable at {WORKER_URL}: {e}'
    return f'**Error ({stage}):** {type(e).__name__}: {e}'


def _apply_polygon(img, poly_json):
    """Mask everything outside the polygon (normalized 0-1 points) to alpha 0.
    TRELLIS preprocess crops to the alpha bbox, so the polygon defines the
    target region exactly."""
    try:
        pts = json.loads(poly_json)
        assert isinstance(pts, list) and len(pts) >= 3
        xy = [(float(p[0]) * img.width, float(p[1]) * img.height) for p in pts]
    except Exception:
        return img
    mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mask).polygon(xy, fill=255)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    r, g, b, a = img.split()
    return Image.merge('RGBA', (r, g, b, ImageChops.multiply(a, mask)))


def image_to_3d(image, polygon, seed, ss_steps, slat_steps, ss_cfg, slat_cfg,
                target_tris, texture_size, smooth_mesh, offload_to_ram):
    # NOTE: every return below MUST yield 4 values to match
    # outputs=[state, video_output, gen_status, viewer_url]. Returning 3 makes
    # Gradio raise "didn't receive enough output values" and the real error
    # message never reaches the user.
    if image is None:
        return None, None, '**No image uploaded.**', None
    try:
        poly_note = ''
        if polygon:
            try:
                pts = json.loads(polygon)
                if isinstance(pts, list) and len(pts) >= 3:
                    poly_note = f' — polygon mask applied ({len(pts)} pts)'
                    image = _apply_polygon(image, polygon)
            except Exception:
                pass
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        # UI slider is "offload to RAM" (1 = max offload); the worker's
        # offload_ratio is GPU residency (0 = full offload), so invert it.
        residency = 1.0 - float(offload_to_ram)
        r = _client.post(
            f'{WORKER_URL}/generate',
            files={'image': ('image.png', buf, 'image/png')},
            data={
                'seed': int(seed), 'ss_steps': int(ss_steps),
                'slat_steps': int(slat_steps), 'ss_cfg': float(ss_cfg),
                'slat_cfg': float(slat_cfg), 'target_tris': int(target_tris),
                'texture_size': int(texture_size),
                'smooth_mesh': bool(smooth_mesh),
                'offload_ratio': residency,
            },
        )
        j = r.json()
        if 'error' in j:
            return None, None, f'**Error (generate):** {j["error"]}', None
        state = {k: j[k] for k in ('id', 'video', 'glb', 'ply', 'zip', 'blend')}
        # Surface what residency the worker actually used: the request is capped
        # against real free VRAM, so asking for more can legitimately keep
        # fewer models resident. Without this the slider looks like a no-op.
        kept = [n.replace('_model', '') for n in (j.get('keep_resident') or [])]
        if kept:
            res_note = ' · GPU-resident: ' + ', '.join(kept)
        elif residency > 0:
            res_note = (f' · requested {residency:.2f} residency did not fit '
                        f'{j.get("usable_vram_mb", "?")} MB usable — fully offloaded')
        else:
            res_note = ' · fully offloaded to RAM'
        status = (f'**Done in {j["seconds"]} s** — {j["faces"]:,} faces '
                  f'· peak {j.get("peak_mb", "?")} MB{res_note}{poly_note}. '
                  f'GLB/PLY/.blend ready — use the buttons below to grab them.')
        glb_url = f'{ORCH_URL}/download/{j["glb"].split("/")[-1]}'
        return state, j['video'], status, glb_url
    except Exception as e:
        return None, None, _err_status(e, 'generate'), None


def extract_glb(state, mesh_simplify, texture_size):
    if not state:
        return None, 'Generate first.'
    return state['glb'], '**GLB** already extracted by the worker during generation.'


def extract_gaussian(state):
    if not state:
        return None, 'Generate first.'
    return state['ply'], '**Gaussian PLY** already extracted by the worker during generation.'


def convert_glb_to_obj_ui(glb_file):
    if glb_file is None:
        return None, 'No file uploaded.'
    # The workspace is scratch: the zip is moved out and the directory is
    # removed on every exit path. These glb_<ts> dirs used to accumulate in
    # tmp/ forever, each holding a copy of the upload plus the expanded set.
    ws = os.path.join(TMP_DIR, f'glb_{time.time_ns()}')
    os.makedirs(ws, exist_ok=True)
    try:
        src = shutil.copy(glb_file, os.path.join(ws, os.path.basename(glb_file)))
        out = glb_to_obj(src, as_zip=True)
        final = os.path.join(TMP_DIR, os.path.basename(out))
        shutil.move(out, final)
        return final, f'**Converted:** `{os.path.basename(final)}`'
    except Exception as e:
        return None, _err_status(e, 'convert')
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def clear_gpu_cache():
    try:
        r = _client.post(f'{WORKER_URL}/clear', timeout=60)
        j = r.json()
        if 'error' in j:
            return f'**Clear failed:** {j["error"]}'
        return (f'**Cache cleared.** Worker GPU in use: '
                f'{j["gpu_mb"]} MB (extractor grids moved to DRAM).')
    except Exception as e:
        return f'**Clear failed:** worker unreachable: {e}'


VIEWER_JS = f"""
<script type="importmap">{{"imports":{{"three":"{ORCH_URL}/static/three.module.js","three/addons/":"{ORCH_URL}/static/jsm/"}}}}</script>
<script type="module">
import {{ initViewer, loadModel, zoomIn, zoomOut, zoomBy, rotateBy,
         resetView, toggleAutoRotate, getDistance, currentKind }}
  from '{ORCH_URL}/static/viewer.js';
window.viewer3d = {{ initViewer, loadModel, zoomIn, zoomOut, zoomBy, rotateBy,
                    resetView, toggleAutoRotate, getDistance, currentKind }};
// gr.HTML content is inserted with innerHTML, so its <script> tags never run
// and inline onclick= can only reach GLOBAL functions. Publish them here (the
// same pattern the polygon editor uses).
window.viewerZoomIn = () => window.viewer3d.zoomIn();
window.viewerZoomOut = () => window.viewer3d.zoomOut();
window.viewerReset = () => window.viewer3d.resetView();
window.viewerRotate = (deg) => window.viewer3d.rotateBy(deg);
window.viewerSpin = (btn) => {{
  const on = window.viewer3d.toggleAutoRotate();
  if (btn) btn.textContent = on ? 'Pause spin' : 'Auto-spin';
}};
function bootViewer(){{ var c=document.getElementById('glb-viewer'); if(!c) return false; initViewer(c,''); return true; }}
(function w(){{ if(!bootViewer()) setTimeout(w, 400); }})();
</script>
"""


def _vbtn(onclick, label, title, bold=False):
    """One viewer control button. Inline onclick is required: gr.HTML is
    inserted via innerHTML, so <script> inside it never executes and only
    globals published by VIEWER_JS (head) are reachable."""
    return (
        f'<button type="button" onclick="{onclick}" title="{title}" '
        f'style="min-width:34px;padding:4px 10px;border:1px solid #555;'
        f'border-radius:6px;background:#222;color:#eee;cursor:pointer;'
        f'font-size:{"16px;font-weight:700" if bold else "13px"}">'
        f'{label}</button>')


POLY_HTML = """
<div id="poly-editor">
 <canvas id="poly-canvas" style="width:100%;border:1px dashed #555;border-radius:8px;cursor:crosshair;background:#1f1f1f"></canvas>
 <div style="display:flex;gap:8px;margin-top:6px;align-items:center;flex-wrap:wrap">
  <button type="button" id="poly-load" onclick="polyLoad()" style="padding:4px 10px;border:1px solid #555;border-radius:6px;background:#222;color:#eee;cursor:pointer">Load image</button>
  <button type="button" id="poly-done" onclick="polyDone()" style="padding:4px 10px;border:1px solid #2b6cb0;border-radius:6px;background:#1a3a5c;color:#fff;cursor:pointer">Done</button>
  <button type="button" id="poly-clear" onclick="polyClear()" style="padding:4px 10px;border:1px solid #555;border-radius:6px;background:#222;color:#eee;cursor:pointer">Clear</button>
  <span id="poly-status" style="color:#999;font-size:12px">Upload an image, click Load image, then click outline points; double-click or Done to close.</span>
 </div>
</div>
"""


# Injected into the page <head>: gr.HTML content is inserted via innerHTML,
# which NEVER executes <script> tags -- head scripts do.
POLY_JS = """
<script>
(function(){
 var canvas=null, ctx=null, status=null, imgObj=null, pts=[], closed=false, lastClick=0;
 function findImg(){
   var qs=['#prompt-image img','#prompt-image .image-container img',
           '[data-testid="image"] img','#prompt-image img[src]'];
   for(var i=0;i<qs.length;i++){ var e=document.querySelector(qs[i]); if(e&&e.src) return e; }
   var imgs=document.querySelectorAll('img[src^="blob:"]');
   return imgs.length?imgs[imgs.length-1]:null;
 }
 function redraw(){
   if(!ctx) return;
   ctx.clearRect(0,0,canvas.width,canvas.height);
   if(!imgObj) return;
   ctx.drawImage(imgObj,0,0,canvas.width,canvas.height);
   if(pts.length>0){
     ctx.beginPath();
     ctx.moveTo(pts[0][0]*canvas.width, pts[0][1]*canvas.height);
     for(var i=1;i<pts.length;i++) ctx.lineTo(pts[i][0]*canvas.width, pts[i][1]*canvas.height);
     if(closed) ctx.closePath();
     ctx.strokeStyle='#4fc3f7'; ctx.lineWidth=2; ctx.stroke();
     if(closed){ ctx.fillStyle='rgba(79,195,247,0.15)'; ctx.fill(); }
     for(var j=0;j<pts.length;j++){ ctx.beginPath(); ctx.arc(pts[j][0]*canvas.width, pts[j][1]*canvas.height, 3, 0, 7); ctx.fillStyle='#fff'; ctx.fill(); }
   }
 }
 function push(){
   var qs=['#polygon-box input[data-testid="textbox"]','#polygon-box input',
           '#polygon-box textarea'];
   for(var i=0;i<qs.length;i++){
     var t=document.querySelector(qs[i]);
     if(t){
       t.value=JSON.stringify(pts);
       t.dispatchEvent(new Event('input',{bubbles:true}));
       t.dispatchEvent(new Event('change',{bubbles:true}));
       return true;
     }
   }
   return false;
 }
 function setStatus(s){ if(status) status.textContent=s; }
 function done(){
   if(!imgObj){ setStatus('Load an image first.'); return; }
   if(pts.length<3){ setStatus('Need at least 3 points (have '+pts.length+').'); return; }
   closed=true; redraw();
   var ok=push();
   setStatus(pts.length+' points'+(ok?' - polygon applied; the result line will say "polygon mask applied".':' - ERROR: could not store polygon, click Done again.'));
 }
 function boot(){
   canvas=document.getElementById('poly-canvas');
   if(!canvas) return false;
   ctx=canvas.getContext('2d');
   status=document.getElementById('poly-status');
   canvas.onclick=function(e){
     if(!imgObj){ setStatus('Load an image first.'); return; }
     var now=Date.now();
     if(now-lastClick<300) return;   // let dblclick handle the close
     lastClick=now;
     var r=canvas.getBoundingClientRect();
     pts.push([(e.clientX-r.left)/r.width, (e.clientY-r.top)/r.height]);
     closed=false; redraw();
   };
   canvas.ondblclick=function(e){
     e.preventDefault();
     if(!imgObj){ setStatus('Load an image first.'); return; }
     var r=canvas.getBoundingClientRect();
     pts.push([(e.clientX-r.left)/r.width, (e.clientY-r.top)/r.height]);
     redraw();
     done();
   };
   return true;
 }
 (function waitBoot(){
   if(!boot()){
     if(window.__polyTries===undefined) window.__polyTries=0;
     if(window.__polyTries++ < 60) setTimeout(waitBoot, 500);
   }
 })();
 window.polyLoad=function(){
   var img=findImg();
   if(!img){ setStatus('Upload an image first (no image found in the page).'); return; }
   imgObj=new Image();
   imgObj.onload=function(){
     var sc=Math.min(1, 720/imgObj.naturalWidth);
     canvas.width=Math.round(imgObj.naturalWidth*sc);
     canvas.height=Math.round(imgObj.naturalHeight*sc);
     pts=[]; closed=false; redraw();
     setStatus('Click outline points; double-click or Done to close.');
   };
   imgObj.onerror=function(){ setStatus('Could not load the uploaded image.'); };
   imgObj.src=img.src;
 };
 window.polyDone=done;
 window.polyClear=function(){ pts=[]; closed=false; redraw(); push(); setStatus('Cleared.'); };
})();
</script>
"""


with gr.Blocks(title='3DModelGen — TRELLIS (low VRAM)', head=POLY_JS + VIEWER_JS) as demo:
    gr.Markdown('## Image to 3D Asset — TRELLIS v1 on 8GB VRAM\n'
                '*Frontend drives the LocalModelGen worker (Go-orchestrated, '
                'weights in DRAM, one sub-model on GPU at a time).*')
    with gr.Row():
        with gr.Column():
            image_prompt = gr.Image(label='Image Prompt', format='png',
                                    image_mode='RGBA', type='pil', height=300,
                                    elem_id='prompt-image')
            polygon_box = gr.Textbox(value='', visible=False, elem_id='polygon-box')
            gr.HTML(POLY_HTML)
            seed = gr.Slider(0, MAX_SEED, label='Seed', value=1, step=1)
            gr.Markdown('**Stage 1: Sparse Structure**')
            with gr.Row():
                ss_guidance = gr.Slider(0.0, 10.0, label='Guidance', value=7.5, step=0.1)
                ss_steps = gr.Slider(1, 50, label='Steps', value=8, step=1)
            gr.Markdown('**Stage 2: Structured Latent**')
            with gr.Row():
                slat_guidance = gr.Slider(0.0, 10.0, label='Guidance', value=3.0, step=0.1)
                slat_steps = gr.Slider(1, 50, label='Steps', value=8, step=1)
            target_tris = gr.Slider(0, 20000, label='Target Triangles (0 = raw)', value=3000, step=100)
            smooth_mesh = gr.Checkbox(value=True, label='Smooth mesh (dominant-orientation)',
                                      info='Taubin + dominant-orientation smoothing on the exported mesh')
            offload_to_ram = gr.Slider(0.0, 1.0, value=1.0, step=0.05,
                                       label='Offload to RAM (1 = max DRAM offload)',
                                       info='0 keeps the most-used model on GPU (faster, more VRAM); the safe default is 1')
            gr.Markdown('**VRAM tips** (parameter up = more VRAM):\n'
                        '- **Target Triangles**: down = less VRAM + flatter faces, BUT less texture detail '
                        '(UVs spread thinner). 0 = raw (soft-capped at 500K faces for tractable export). ~3000 = good balance.\n'
                        '- **Texture Size** (extract): down = much less VRAM (size²). 1024 = detail, 512 = safe.\n'
                        '- **Steps**: down = less VRAM + faster.\n'
                        '- **Guidance**: negligible VRAM.\n'
                        '- **Clear GPU Cache**: press if GPU-in-use creeps up.')
            clear_btn = gr.Button('Clear GPU Cache', variant='secondary')
            clear_status = gr.Markdown('')
            generate_btn = gr.Button('Generate')
            with gr.Accordion('GLB Extraction', open=True):
                gr.Markdown('*Extraction runs inside generation on the worker. '
                            'Decimation is controlled by Target Triangles.*')
                texture_size = gr.Slider(256, 2048, label='Texture Size (lower = less VRAM)', value=1024, step=256)
            with gr.Row():
                extract_glb_btn = gr.Button('Extract GLB', interactive=False)
                extract_gs_btn = gr.Button('Extract Gaussian', interactive=False)
        with gr.Column():
            gr.HTML(
                '<canvas id="glb-viewer" tabindex="0" style="width:100%;'
                'height:320px;border-radius:10px;background:#0d1117;'
                'touch-action:none;outline:none"></canvas>'
                '<div id="viewer-controls" style="display:flex;gap:6px;'
                'margin-top:6px;align-items:center;flex-wrap:wrap">'
                + _vbtn('viewerZoomIn()', '+', 'Zoom in (or scroll / + key)',
                        bold=True)
                + _vbtn('viewerZoomOut()', '\u2212', 'Zoom out (or scroll / - key)',
                        bold=True)
                + _vbtn('viewerRotate(-30)', '\u21b6', 'Rotate left 30\u00b0')
                + _vbtn('viewerRotate(30)', '\u21b7', 'Rotate right 30\u00b0')
                + _vbtn('viewerSpin(this)', 'Pause spin', 'Toggle the idle spin')
                + _vbtn('viewerReset()', 'Reset', 'Back to the default framing (0 key)')
                + '</div>'
                '<div style="color:#999;font-size:12px;margin-top:4px">'
                'Realtime render — drag to rotate, scroll or +/\u2212 to zoom, '
                'click the canvas then use +/\u2212/0 keys.</div>')
            viewer_url = gr.Textbox(value='', visible=False)
            video_output = gr.Video(label='Generated 3D Asset', autoplay=True, loop=True, height=300)
            gen_status = gr.Markdown('')
            glb_file = gr.File(label='GLB')
            glb_status = gr.Markdown('')
            blend_file = gr.File(label='Blender scene (.blend)', interactive=False)
            gs_file = gr.File(label='Gaussian PLY')
            gs_status = gr.Markdown('')
    with gr.Accordion('GLB to OBJ Converter', open=False):
        gr.Markdown('*Upload a .glb (e.g. the generated one) to get .obj + .mtl + '
                    'texture for Blender / WebGL.*')
        with gr.Row():
            glb_input = gr.File(label='GLB file', file_types=['.glb'])
            convert_btn = gr.Button('Convert to OBJ')
        obj_output = gr.File(label='Download .zip (.obj + .mtl + texture)',
                             interactive=False)
        convert_status = gr.Markdown('')
        convert_btn.click(convert_glb_to_obj_ui, inputs=[glb_input],
                          outputs=[obj_output, convert_status],
                          api_name='convert_glb_to_obj_ui')
    state = gr.State()

    generate_btn.click(
        image_to_3d,
        inputs=[image_prompt, polygon_box, seed, ss_steps, slat_steps,
                ss_guidance, slat_guidance, target_tris, texture_size,
                smooth_mesh, offload_to_ram],
        outputs=[state, video_output, gen_status, viewer_url],
        api_name='image_to_3d',
    ).then(
        lambda state: state.get('blend', '') if state else '',
        inputs=[state],
        outputs=[blend_file],
    )
    viewer_url.change(
        lambda u: u, inputs=[viewer_url], outputs=[viewer_url],
        js="(u) => { if (u && window.viewer3d) window.viewer3d.loadModel(u); return u; }",
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )
    extract_glb_btn.click(
        extract_glb,
        inputs=[state, target_tris, texture_size],
        outputs=[glb_file, glb_status],
        api_name='extract_glb',
    )
    extract_gs_btn.click(
        extract_gaussian,
        inputs=[state],
        outputs=[gs_file, gs_status],
        api_name='extract_gaussian',
    )
    clear_btn.click(clear_gpu_cache, inputs=[], outputs=[clear_status],
                    api_name='clear_cache')

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, show_error=True)
