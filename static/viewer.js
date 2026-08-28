// Shared non-CDN WebGL GLB viewer (three.js, served locally).
// Drag to rotate, scroll / +- buttons / +- keys to zoom, auto-spins until
// the user interacts. Used by both front ends (orchestrator :8080 and the
// Gradio UI :7860, which loads this module cross-origin from /static/).
import * as THREE from 'three';
// Canonical three.js addon layout (examples/jsm/...): GLTFLoader itself does a
// RELATIVE import of ../utils/BufferGeometryUtils.js, so the addons MUST sit in
// the loaders/ + controls/ + utils/ tree. Serving them flat made that resolve
// to /utils/BufferGeometryUtils.js -- outside /static/ -- which 404'd and took
// the whole module graph down, so the viewer never initialised at all.
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// Needed for correct PBR: TRELLIS assets are exported by trimesh WITHOUT a
// metallicFactor, and the glTF default is 1.0 -- fully metallic. A metal with
// no environment to reflect renders BLACK no matter how many direct lights are
// in the scene, which is exactly how these models used to look. An IBL
// environment is the physically correct fix (as opposed to overwriting the
// asset's own material properties).
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { parseGaussianPly, createGaussianPoints } from './gaussians.js';

let renderer = null, scene = null, camera = null, controls = null;
// PMREM generator + the environment texture it produced; both own GPU memory
// and must be released in disposeViewer.
let pmrem = null, envTexture = null;
let current = null;
// initViewer is called from a retry loop in the Gradio front end; without this
// boolean each retry built a second WebGLRenderer and leaked a GL context.
let initialized = false;
// Monotonic token: an in-flight load whose token is stale must not add its
// model to the scene, otherwise a slow earlier response lands after a newer
// one and two models stay resident.
let loadToken = 0;
const loader = new GLTFLoader();

export function initViewer(canvas, url) {
  if (initialized) {
    if (url) loadModel(url);
    return;
  }
  initialized = true;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // ACES filmic keeps the bright IBL highlights from clipping to flat white.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14181f);

  // Prefiltered radiance environment so metallic/rough PBR resolves properly.
  // Built once, then the source scene is thrown away (the texture is kept).
  pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  const roomScene = new RoomEnvironment();
  envTexture = pmrem.fromScene(roomScene, 0.04).texture;
  scene.environment = envTexture;
  roomScene.traverse(o => o.geometry?.dispose());

  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(1.1, 0.7, 1.1);

  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  // Explicit rather than relying on defaults: these three are the documented
  // interactions (drag to rotate, wheel to zoom) and must not silently depend
  // on an OrbitControls default flipping in a future three.js drop.
  controls.enableRotate = true;
  controls.enableZoom = true;
  controls.zoomSpeed = 1.0;
  controls.enablePan = false;   // panning a single centred asset just loses it
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.8;
  controls.minDistance = 0.3;
  controls.maxDistance = 6;
  // stop auto-spin once the user interacts (OrbitControls fires 'start' for
  // drag AND wheel); the button/key zoom path calls stopAutoRotate() itself.
  controls.addEventListener('start', stopAutoRotate);
  canvas.addEventListener('keydown', onKey);
  canvas.tabIndex = 0;          // canvas must be focusable to get key events
  window.addEventListener('keydown', onKey);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3f55, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(2.5, 3.5, 3);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x99bbff, 0.6);
  fill.position.set(-2.5, -0.5, -2);
  scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffffff, 0.7);
  rim.position.set(0, -2, 2.5);
  scene.add(rim);

  resize();
  window.addEventListener('resize', resize);
  renderer.setAnimationLoop(animate);
  if (url) loadModel(url);
}

// --- camera controls ---------------------------------------------------------
// The home framing the model is loaded at, so "reset" is meaningful after the
// user has orbited and zoomed away.
const HOME_POS = new THREE.Vector3(1.1, 0.7, 1.1);

function stopAutoRotate() {
  if (controls) controls.autoRotate = false;
}

/** Multiply the camera's distance to the orbit target. <1 zooms in, >1 out.
 *  Clamped to the same min/maxDistance the wheel obeys, so buttons, keys and
 *  scrollwheel can never disagree about the limits. */
export function zoomBy(factor) {
  if (!initialized || !controls) return null;
  stopAutoRotate();
  const t = controls.target;
  const dir = camera.position.clone().sub(t);
  const d = Math.min(Math.max(dir.length() * factor, controls.minDistance),
                     controls.maxDistance);
  camera.position.copy(t).add(dir.setLength(d));
  controls.update();
  return d;
}

export const zoomIn = () => zoomBy(1 / 1.25);
export const zoomOut = () => zoomBy(1.25);

/** Current camera distance to the orbit target (for tests / UI readouts). */
export function getDistance() {
  return initialized && controls
    ? camera.position.distanceTo(controls.target) : null;
}
/** What the viewer currently holds: 'splat' | 'mesh' | 'none'. */
export function currentKind() {
  return !current ? 'none' : (current.isSplat ? 'splat' : 'mesh');
}

/** Orbit programmatically, in degrees — drives the on-screen rotate buttons.
 *  Applied in the camera's own frame so it matches what dragging does. */
export function rotateBy(deltaAzimuthDeg, deltaPolarDeg = 0) {
  if (!initialized || !controls) return null;
  stopAutoRotate();
  const t = controls.target;
  const off = camera.position.clone().sub(t);
  const sph = new THREE.Spherical().setFromVector3(off);
  sph.theta += THREE.MathUtils.degToRad(deltaAzimuthDeg);
  sph.phi = Math.min(Math.max(sph.phi + THREE.MathUtils.degToRad(deltaPolarDeg),
                              1e-3), Math.PI - 1e-3);
  camera.position.copy(t).add(off.setFromSpherical(sph));
  controls.update();
  return { theta: sph.theta, phi: sph.phi };
}

/** Back to the framing a freshly loaded model gets, and resume the spin. */
export function resetView() {
  if (!initialized || !controls) return;
  controls.target.set(0, 0, 0);
  camera.position.copy(HOME_POS);
  controls.autoRotate = true;
  controls.update();
}

/** Toggle the idle spin; returns the new state so a button can label itself. */
export function toggleAutoRotate() {
  if (!initialized || !controls) return null;
  controls.autoRotate = !controls.autoRotate;
  return controls.autoRotate;
}

function onKey(e) {
  if (!initialized) return;
  // Never steal keys from a text field (the Gradio page is full of them).
  const tag = (e.target && e.target.tagName || '').toLowerCase();
  if (tag === 'input' || tag === 'textarea' || e.target?.isContentEditable) return;
  if (e.key === '+' || e.key === '=') { zoomIn(); e.preventDefault(); }
  else if (e.key === '-' || e.key === '_') { zoomOut(); e.preventDefault(); }
  else if (e.key === '0') { resetView(); e.preventDefault(); }
}

// disposeObject releases every GPU resource a loaded glTF owns. The previous
// version disposed geometry and `m.map` only, so materials and every other
// texture slot (normalMap / emissiveMap / metalnessRoughnessTexture, all
// present in TRELLIS bakes) leaked on each auto-load -- and the UI reloads
// whenever the newest result changes.
function disposeObject(root) {
  root.traverse(o => {
    o.geometry?.dispose();
    if (!o.material) return;
    for (const m of (Array.isArray(o.material) ? o.material : [o.material])) {
      for (const v of Object.values(m)) {
        // Never dispose the shared IBL environment: it is owned by the scene,
        // not by this model, and a swap would leave every later model unlit.
        if (v && v.isTexture && v !== envTexture) v.dispose();
      }
      m.dispose();
    }
  });
}

// Tear the viewer down completely: without this the resize listener, the
// animation loop and the GL context itself outlive the page's use of it.
export function disposeViewer() {
  if (!initialized) return;
  loadToken++;
  window.removeEventListener('resize', resize);
  window.removeEventListener('keydown', onKey);
  renderer?.domElement?.removeEventListener('keydown', onKey);
  renderer?.setAnimationLoop(null);
  if (current) { scene.remove(current); disposeObject(current); current = null; }
  controls?.dispose();
  // The IBL texture and the PMREM generator both hold render targets.
  envTexture?.dispose();
  pmrem?.dispose();
  renderer?.dispose();
  renderer?.forceContextLoss();
  renderer = scene = camera = controls = pmrem = envTexture = null;
  initialized = false;
}

function resize() {
  if (!renderer) return;
  const el = renderer.domElement;
  const w = el.clientWidth || el.parentElement?.clientWidth || 640;
  const h = el.clientHeight || el.parentElement?.clientHeight || 420;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function animate() {
  controls?.update();
  // Splat objects carry an updateSplat hook (depth re-sort + focal refresh).
  current?.updateSplat?.(camera, renderer);
  renderer?.render(scene, camera);
}

// The baked video (gen_*.mp4) is a Gaussian-splat render, so the interactive
// preview must render the *_ply splat (same data) to look like it. A mesh GLB
// is a different (opaque, decimated) reconstruction and can only be loaded as
// a fallback when no sibling splat exists.
function setSplatAppearance() {
  scene.background = new THREE.Color(0x000000); // baked video background
  renderer.toneMapping = THREE.NoToneMapping;   // raw rasterizer output
  // Output the fragment as-is (the video saved rasterizer values straight to
  // uint8), so we must NOT apply the linear->sRGB OETF. LinearSRGBColorSpace
  // outputs the fragment untouched (identity transfer). NoColorSpace is a
  // bare '' and crashes getPrimaries in this three build, so it's unusable.
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
}

function setGlbAppearance() {
  scene.background = new THREE.Color(0x14181f);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
}

function showSplat(points, token) {
  if (token !== loadToken || !initialized) { points.dispose(); return; }
  points.isSplat = true;
  const arr = points.geometry.getAttribute('position').array;
  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  for (let i = 0; i < arr.length; i += 3) { v.set(arr[i], arr[i + 1], arr[i + 2]); box.expandByPoint(v); }
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const center = box.getCenter(new THREE.Vector3());
  points.position.sub(center);
  points.scale.setScalar(1.7 / size);
  current = points;
  setSplatAppearance();
  scene.add(points);
  points.updateSplat(camera, renderer, true);
  resetView();
}

function showGlb(gltf, url, token) {
  const model = gltf.scene || gltf.scenes[0];
  if (token !== loadToken || !initialized) {
    disposeObject(model);
    return;
  }
  current = model;
  const box = new THREE.Box3().setFromObject(current);
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const center = box.getCenter(new THREE.Vector3());
  current.position.sub(center);
  current.scale.setScalar(1.7 / size);
  setGlbAppearance();
  scene.add(current);
  resetView();
}

export function loadModel(url) {
  if (!url || !initialized) return;
  const token = ++loadToken;
  if (current) {
    scene.remove(current);
    disposeObject(current);
    current = null;
  }
  // Prefer the Gaussian splat (matches the baked video); fall back to the GLB.
  const isPly = /\.ply$/i.test(url);
  const splatUrl = isPly ? url : /\.glb$/i.test(url) ? url.replace(/\.glb$/i, '.ply') : null;
  if (splatUrl) {
    const loadSplat = (attempt) => {
      fetch(splatUrl)
        .then(r => (r.ok ? r.arrayBuffer() : Promise.reject(new Error('ply ' + r.status))))
        .then(buf => showSplat(createGaussianPoints(parseGaussianPly(buf)), token))
        .catch(() => {
          // A cold cross-origin fetch can fail once; retry before giving up.
          if (attempt < 2) { setTimeout(() => loadSplat(attempt + 1), 800); return; }
          console.warn('splat load failed after retries:', splatUrl);
          if (!isPly) loader.load(url, g => showGlb(g, url, token), undefined,
            e => console.error('GLB load failed:', url, e));
        });
    };
    loadSplat(0);
    return;
  }
  loader.load(url, g => showGlb(g, url, token), undefined,
    err => console.error('GLB load failed:', url, err));
}
