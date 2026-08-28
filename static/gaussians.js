// Gaussian-splat PLY renderer for three.js.
//
// Renders a TRELLIS `save_ply` Gaussian (x,y,z / f_dc_0..2 / opacity /
// scale_0..2 / rot_0..3) as translucent back-to-front splats, reproducing the
// appearance of trellis' `GaussianRenderer` output (the baked `gen_*.mp4`).
// The color is SH degree-0 (view-independent): color = 0.5 + C0 * f_dc, and
// the mip-gaussian low-pass (kernel_size=0.1) that the CUDA rasterizer applies
// is replicated so tiny splats don't over-brighten.
import * as THREE from 'three';

const SH_C0 = 0.28209479177387814;
const KERNEL = 0.1;

// ---- binary PLY parser ----------------------------------------------------
function parseHeader(bytes) {
  let p = 0;
  const line = () => {
    let s = '';
    while (p < bytes.length) {
      const c = bytes[p++];
      if (c === 10) break;
      if (c !== 13) s += String.fromCharCode(c);
    }
    return s;
  };
  if (line().trim() !== 'ply') throw new Error('not a PLY');
  let format = '';
  const elements = [];
  let cur = null;
  for (;;) {
    const l = line();
    if (l === 'end_header') break;
    const parts = l.trim().split(/\s+/);
    if (parts[0] === 'format') format = parts[1];
    else if (parts[0] === 'element') {
      cur = { name: parts[1], count: parseInt(parts[2], 10), props: [] };
      elements.push(cur);
    } else if (parts[0] === 'property') cur.props.push({ name: parts[2], type: parts[1] });
  }
  return { format, elements, dataOffset: p };
}

const TYPE_SIZE = { char: 1, uchar: 1, short: 2, ushort: 2, int: 4, uint: 4, float: 4, double: 8 };

export function parseGaussianPly(data) {
  const bytes = new Uint8Array(data);
  const { format, elements, dataOffset } = parseHeader(bytes);
  if (!format.startsWith('binary')) throw new Error('only binary PLY supported: ' + format);
  const little = format.includes('little_endian');
  const vert = elements.find(e => e.name === 'vertex');
  if (!vert) throw new Error('no vertex element');
  const N = vert.count;

  const stride = vert.props.reduce((t, p) => t + TYPE_SIZE[p.type], 0);
  const colOff = {};
  let off = 0;
  for (const p of vert.props) { colOff[p.name] = off; off += TYPE_SIZE[p.type]; }

  const dv = new DataView(data, dataOffset, N * stride);
  const read = (idx, name, type) => dv['get' + ({ float: 'Float32', double: 'Float64', int: 'Int32',
    uint: 'Uint32', short: 'Int16', ushort: 'Uint16', uchar: 'Uint8', char: 'Int8' }[type])](idx * stride + colOff[name], little);

  const propsOf = prefix => vert.props.map(p => p.name).filter(n => n.startsWith(prefix))
    .sort((a, b) => parseInt(a.split('_').pop(), 10) - parseInt(b.split('_').pop(), 10));
  const dc = propsOf('f_dc');
  const sc = propsOf('scale');
  const rt = propsOf('rot');

  const centers = new Float32Array(N * 3);
  const colors = new Float32Array(N * 3);
  const opacity = new Float32Array(N);
  const scales = new Float32Array(N * 3);
  const quats = new Float32Array(N * 4);
  for (let i = 0; i < N; i++) {
    centers[i * 3] = read(i, 'x', 'float');
    centers[i * 3 + 1] = read(i, 'y', 'float');
    centers[i * 3 + 2] = read(i, 'z', 'float');
    for (let c = 0; c < 3; c++) {
      const v = 0.5 + SH_C0 * read(i, dc[c], 'float');
      colors[i * 3 + c] = v < 0 ? 0 : v; // glm::max(result, 0)
    }
    opacity[i] = 1 / (1 + Math.exp(-read(i, 'opacity', 'float'))); // sigmoid
    for (let c = 0; c < 3; c++) scales[i * 3 + c] = Math.exp(read(i, sc[c], 'float'));
    let q0 = read(i, rt[0], 'float'), q1 = read(i, rt[1], 'float'),
        q2 = read(i, rt[2], 'float'), q3 = read(i, rt[3], 'float');
    const n = Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3) || 1;
    quats[i * 4] = q0 / n; quats[i * 4 + 1] = q1 / n;
    quats[i * 4 + 2] = q2 / n; quats[i * 4 + 3] = q3 / n;
  }
  // TRELLIS save_ply stores the asset in a frame where trellis +Z (the baked
  // video's up) maps to -Y, so an upright Y-up viewer shows it upside-down.
  // Rotate 180 deg about X (negate Y and Z) to reproduce the video's pose:
  // centers (x,y,z) -> (x,-y,-z); quaternion (w,x,y,z) -> (-x,w,-z,y) via
  // Rx(180) composed on the left (q' = q_fix * q).
  for (let i = 0; i < N; i++) {
    centers[i * 3 + 1] = -centers[i * 3 + 1];
    centers[i * 3 + 2] = -centers[i * 3 + 2];
    const w = quats[i * 4], x = quats[i * 4 + 1],
          y = quats[i * 4 + 2], z = quats[i * 4 + 3];
    quats[i * 4] = -x; quats[i * 4 + 1] = w;
    quats[i * 4 + 2] = -z; quats[i * 4 + 3] = y;
  }
  return { count: N, centers, colors, opacity, scales, quats };
}

// ---- splat shaders --------------------------------------------------------
const VERT = /* glsl */ `
attribute vec3 aColor;
attribute float aOpacity;
attribute vec3 aScale;
attribute vec4 aRot;          // (w,x,y,z)
uniform vec2 uFocal;          // px
uniform vec2 uTanFov;         // tan(fovx/2), tan(fovy/2)
uniform float uKernel;
uniform float uScale;         // object's uniform scale (framing)
varying vec3 vColor;
varying float vOpacity;
varying vec2 vConicXY;        // inv_xx, inv_xy
varying float vConicYY;       // inv_yy
varying float vPointSize;

// world-space rotation R (column-major so R*v = build_rotation * v)
mat3 rotMat(vec4 q) {
  float r = q.x, x = q.y, y = q.z, z = q.w;
  return mat3(
    1.0 - 2.0*(y*y + z*z), 2.0*(x*y + r*z), 2.0*(x*z - r*y),
    2.0*(x*y - r*z), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z + r*x),
    2.0*(x*z + r*y), 2.0*(y*z - r*x), 1.0 - 2.0*(x*x + y*y));
}

void main() {
  // camera-space centre, depth (forward is -Z so z is negative).
  // 'position' is the geometry's position attribute (the splat centres).
  vec4 cam = modelViewMatrix * vec4(position, 1.0);
  float z = cam.z;

  // clamp mean to frustum (matches CUDA computeCov2D before projection)
  float limx = 1.3 * uTanFov.x, limy = 1.3 * uTanFov.y;
  float tx = clamp(cam.x / z, -limx, limx) * z;
  float ty = clamp(cam.y / z, -limy, limy) * z;

  // S3 = R diag(s^2) R^T
  mat3 R = rotMat(aRot);
  vec3 s2 = (aScale * uScale) * (aScale * uScale);
  mat3 RS = R;
  RS[0] *= s2.x; RS[1] *= s2.y; RS[2] *= s2.z;
  mat3 Sig3 = RS * transpose(R);

  // Scam = R_view S3 R_view^T  (R_view = world->camera rotation)
  mat3 Rv = mat3(modelViewMatrix);
  mat3 Sig = Rv * Sig3 * transpose(Rv);

  // projection Jacobian (focal in pixels), matching the CUDA rasterizer
  float fx = uFocal.x, fy = uFocal.y;
  mat3 J = mat3(fx / z, 0.0, -fx * tx / (z * z),
                0.0, fy / z, -fy * ty / (z * z),
                0.0, 0.0, 0.0);

  mat3 cov = transpose(J) * Sig * J;
  float covxx = cov[0][0], covyy = cov[1][1], covxy = cov[0][1];

  // mip-gaussian low-pass (kernel_size=0.1)
  float det0 = max(1e-6, covxx * covyy - covxy * covxy);
  float det1 = max(1e-6, (covxx + uKernel) * (covyy + uKernel) - covxy * covxy);
  float coef = sqrt(det0 / (det1 + 1e-6) + 1e-6);
  if (det0 <= 1e-6 || det1 <= 1e-6) coef = 0.0;

  // augment covariance, then invert (EWA conic)
  covxx += uKernel; covyy += uKernel;
  float det = covxx * covyy - covxy * covxy;
  if (det <= 1e-12) { gl_Position = vec4(0.0); gl_PointSize = 0.0; vOpacity = -1.0; return; }
  float det_inv = 1.0 / det;
  vConicXY = vec2(covyy * det_inv, -covxy * det_inv);
  vConicYY = covxx * det_inv;

  // radius = 3*sqrt(lambda_max) in pixels; point diameter = 2*radius
  float mid = 0.5 * (covxx + covyy);
  float lam1 = mid + sqrt(max(0.1, mid * mid - det));
  float lam2 = mid - sqrt(max(0.1, mid * mid - det));
  float radius = ceil(3.0 * sqrt(max(lam1, lam2)));
  vPointSize = 2.0 * radius;

  vColor = aColor;
  vOpacity = aOpacity * coef;

  gl_Position = projectionMatrix * cam;
  gl_PointSize = max(2.0, vPointSize);
}
`;

const FRAG = /* glsl */ `
varying vec3 vColor;
varying float vOpacity;
varying vec2 vConicXY;
varying float vConicYY;
varying float vPointSize;
void main() {
  if (vOpacity < 0.0) discard;
  vec2 d = (gl_PointCoord - 0.5) * vPointSize;
  float power = -0.5 * (vConicXY.x * d.x * d.x + vConicYY * d.y * d.y)
                - vConicXY.y * d.x * d.y;
  if (power > 0.0) discard;
  float alpha = min(0.99, vOpacity * exp(power));
  if (alpha < 1.0 / 255.0) discard;
  gl_FragColor = vec4(vColor, alpha);
}
`;

// ---- splat object ----------------------------------------------------------
export function createGaussianPoints(data) {
  const N = data.count;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(data.centers, 3));
  geo.setAttribute('aColor', new THREE.BufferAttribute(data.colors, 3));
  geo.setAttribute('aOpacity', new THREE.BufferAttribute(data.opacity, 1));
  geo.setAttribute('aScale', new THREE.BufferAttribute(data.scales, 3));
  geo.setAttribute('aRot', new THREE.BufferAttribute(data.quats, 4));

  const mat = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms: {
      uFocal: { value: new THREE.Vector2(1000, 1000) },
      uTanFov: { value: new THREE.Vector2(Math.tan(Math.PI / 8), Math.tan(Math.PI / 8)) },
      uKernel: { value: KERNEL },
      uScale: { value: 1.0 },
    },
    transparent: true,
    depthWrite: false,
    depthTest: true,
    toneMapped: false,
    blending: THREE.NormalBlending,
  });

  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false; // full scene; per-splat culling is in-shader

  // Depth-sorted draw order (back-to-front). TypedArray.sort() ignores a
  // comparator, so we use an O(N) bucket sort keyed by view depth, throttled.
  const order = new Uint32Array(N);
  let counts = null;
  const depth = new Float32Array(N);
  const BUCKETS = 1024;
  let frame = 0;
  let indexAttr = null;

  points.updateSplat = (camera, renderer, force) => {
    camera.updateMatrixWorld();
    camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
    const vm = camera.matrixWorldInverse.elements;

    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const tanY = Math.tan(THREE.MathUtils.degToRad(camera.fov) * 0.5);
    const tanX = tanY * camera.aspect;
    mat.uniforms.uTanFov.value.set(tanX, tanY);
    mat.uniforms.uFocal.value.set(size.x / (2 * tanX), size.y / (2 * tanY));
    mat.uniforms.uScale.value = points.scale.x || 1.0;

    if (!force && (frame++ % 8) !== 0) return;

    const c = data.centers;
    let minD = Infinity, maxD = -Infinity;
    for (let i = 0; i < N; i++) {
      const d = vm[2] * c[i * 3] + vm[6] * c[i * 3 + 1] + vm[10] * c[i * 3 + 2] + vm[14];
      depth[i] = d;
      if (d < minD) minD = d;
      if (d > maxD) maxD = d;
    }
    const range = (maxD - minD) || 1;
    if (!counts) counts = new Uint32Array(BUCKETS);
    else counts.fill(0);
    for (let i = 0; i < N; i++) {
      let b = Math.floor((depth[i] - minD) / range * BUCKETS);
      if (b >= BUCKETS) b = BUCKETS - 1;
      if (b < 0) b = 0;
      counts[b]++;
    }
    let acc = 0;
    for (let b = 0; b < BUCKETS; b++) { const k = counts[b]; counts[b] = acc; acc += k; }
    for (let i = 0; i < N; i++) {
      let b = Math.floor((depth[i] - minD) / range * BUCKETS);
      if (b >= BUCKETS) b = BUCKETS - 1;
      if (b < 0) b = 0;
      order[counts[b]++] = i;
    }
    if (!indexAttr) { indexAttr = new THREE.BufferAttribute(order, 1); geo.setIndex(indexAttr); }
    indexAttr.needsUpdate = true;
  };

  points.dispose = () => { geo.dispose(); mat.dispose(); };
  return points;
}
