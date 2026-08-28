package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/go-gl/gl/v3.3-core/gl"
)

const shC0 = 0.28209479177387814
const splatKernel = 0.1

type SplatData struct {
	N                 int
	Centers, Colors   []float32
	Opacity           []float32
	Scales, Quats     []float32
	MinBB, MaxBB      [3]float32
}

// parseGaussianPly reads a TRELLIS save_ply gaussian (x,y,z / f_dc_0..2 /
// opacity / scale_0..2 / rot_0..3) and applies the 180-deg X orientation fix
// (trellis +Z maps to -Y in the stored frame).
func parseGaussianPly(path string) (*SplatData, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	idx := bytes.Index(b, []byte("end_header"))
	if idx < 0 {
		return nil, fmt.Errorf("no end_header")
	}
	off := idx + len("end_header") + 1
	n := 0
	for _, line := range bytes.Split(b[:idx], []byte("\n")) {
		parts := bytes.Fields(line)
		if len(parts) >= 3 && string(parts[0]) == "element" && string(parts[1]) == "vertex" {
			fmt.Sscanf(string(parts[2]), "%d", &n)
		}
	}
	const ncol = 17 // x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,rot_0..3
	stride := ncol * 4
	if off+n*stride > len(b) {
		return nil, fmt.Errorf("ply data truncated")
	}
	le := binary.LittleEndian
	col := func(base, j int) float32 {
		return math.Float32frombits(le.Uint32(b[off+base+j*4:]))
	}
	d := &SplatData{N: n, Centers: make([]float32, n*3), Colors: make([]float32, n*3),
		Opacity: make([]float32, n), Scales: make([]float32, n*3), Quats: make([]float32, n*4)}
	for i := 0; i < n; i++ {
		base := i * stride
		x, y, z := col(base, 0), col(base, 1), col(base, 2)
		for k := 0; k < 3; k++ {
			v := 0.5 + shC0*col(base, 6+k)
			if v < 0 {
				v = 0
			}
			d.Colors[i*3+k] = v
		}
		op := col(base, 9)
		d.Opacity[i] = float32(1 / (1 + math.Exp(-float64(op))))
		for k := 0; k < 3; k++ {
			d.Scales[i*3+k] = float32(math.Exp(float64(col(base, 10+k))))
		}
		q0, q1, q2, q3 := col(base, 13), col(base, 14), col(base, 15), col(base, 16)
		nn := math.Sqrt(float64(q0*q0 + q1*q1 + q2*q2 + q3*q3))
		if nn == 0 {
			nn = 1
		}
		// Rx(180): (w,x,y,z)->(-x,w,-z,y)
		d.Quats[i*4], d.Quats[i*4+1] = -q1/float32(nn), q0/float32(nn)
		d.Quats[i*4+2], d.Quats[i*4+3] = -q3/float32(nn), q2/float32(nn)
		d.Centers[i*3], d.Centers[i*3+1], d.Centers[i*3+2] = x, -y, -z
	}
	return d, nil
}

func (d *SplatData) centerAndScale() {
	minX, minY, minZ := float32(1e9), float32(1e9), float32(1e9)
	maxX, maxY, maxZ := float32(-1e9), float32(-1e9), float32(-1e9)
	for i := 0; i < d.N; i++ {
		x, y, z := d.Centers[i*3], d.Centers[i*3+1], d.Centers[i*3+2]
		if x < minX { minX = x }; if x > maxX { maxX = x }
		if y < minY { minY = y }; if y > maxY { maxY = y }
		if z < minZ { minZ = z }; if z > maxZ { maxZ = z }
	}
	cx, cy, cz := (minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2
	size := float32(math.Sqrt(float64((maxX-minX)*(maxX-minX) + (maxY-minY)*(maxY-minY) + (maxZ-minZ)*(maxZ-minZ))))
	if size == 0 {
		size = 1
	}
	s := 1.7 / size
	for i := 0; i < d.N; i++ {
		d.Centers[i*3] = (d.Centers[i*3] - cx) * s
		d.Centers[i*3+1] = (d.Centers[i*3+1] - cy) * s
		d.Centers[i*3+2] = (d.Centers[i*3+2] - cz) * s
		d.Scales[i*3] *= s
		d.Scales[i*3+1] *= s
		d.Scales[i*3+2] *= s
	}
	d.MinBB = [3]float32{minX, minY, minZ}
	d.MaxBB = [3]float32{maxX, maxY, maxZ}
}

const splatVertSrc = `#version 150 core
in vec3 position;
in vec3 aColor;
in float aOpacity;
in vec3 aScale;
in vec4 aRot;
uniform mat4 uView, uProj;
uniform vec2 uFocal, uTanFov;
uniform float uKernel;
uniform float uOpaque;
uniform float uColorize;
uniform float uSizeMul;
uniform vec3 uFlatColor;
out vec3 vColor;
out float vOpacity;
out vec2 vConicXY;
out float vConicYY;
out float vPointSize;
mat3 rotMat(vec4 q){
  float r=q.x, x=q.y, y=q.z, z=q.w;
  return mat3(
    1.0-2.0*(y*y+z*z), 2.0*(x*y+r*z), 2.0*(x*z-r*y),
    2.0*(x*y-r*z), 1.0-2.0*(x*x+z*z), 2.0*(y*z+r*x),
    2.0*(x*z+r*y), 2.0*(y*z-r*x), 1.0-2.0*(x*x+y*y));
}
void main(){
  vec4 cam = uView * vec4(position,1.0);
  float z = cam.z;
  float limx=1.3*uTanFov.x, limy=1.3*uTanFov.y;
  float tx=clamp(cam.x/z,-limx,limx)*z, ty=clamp(cam.y/z,-limy,limy)*z;
  mat3 R=rotMat(aRot);
  vec3 s2=aScale*aScale;
  mat3 RS=R; RS[0]*=s2.x; RS[1]*=s2.y; RS[2]*=s2.z;
  mat3 Sig3=RS*transpose(R);
  mat3 Rv=mat3(uView);
  mat3 Sig=Rv*Sig3*transpose(Rv);
  float fx=uFocal.x, fy=uFocal.y;
  mat3 J=mat3(fx/z, 0.0, -fx*tx/(z*z),
              0.0, fy/z, -fy*ty/(z*z),
              0.0, 0.0, 0.0);
  mat3 cov=transpose(J)*Sig*J;
  float covxx=cov[0][0], covyy=cov[1][1], covxy=cov[0][1];
  float det0=max(1e-6,covxx*covyy-covxy*covxy);
  float det1=max(1e-6,(covxx+uKernel)*(covyy+uKernel)-covxy*covxy);
  float coef=sqrt(det0/(det1+1e-6)+1e-6);
  if(det0<=1e-6||det1<=1e-6){ coef=0.0; }
  covxx+=uKernel; covyy+=uKernel;
  float det=covxx*covyy-covxy*covxy;
  if(det<=1e-12){ vOpacity=-1.0; gl_PointSize=0.0; gl_Position=vec4(0.0); return; }
  float det_inv=1.0/det;
  vConicXY=vec2(covyy*det_inv,-covxy*det_inv);
  vConicYY=covxx*det_inv;
  float mid=0.5*(covxx+covyy);
  float lam1=mid+sqrt(max(0.1,mid*mid-det));
  float lam2=mid-sqrt(max(0.1,mid*mid-det));
  float radius=ceil(3.0*sqrt(max(lam1,lam2)));
  vPointSize=2.0*radius*uSizeMul;
  vColor=aColor;
  vOpacity=aOpacity*coef;
  if(uColorize<0.5){ vColor=uFlatColor; }
  if(uOpaque>0.5){ vOpacity=1.0; }
  gl_Position=uProj*cam;
  gl_PointSize=max(2.0,vPointSize);
}
`

const splatFragSrc = `#version 150 core
in vec3 vColor;
in float vOpacity;
in vec2 vConicXY;
in float vConicYY;
in float vPointSize;
out vec4 fragColor;
void main(){
  if(vOpacity<0.0) discard;
  vec2 d=(gl_PointCoord-0.5)*vPointSize;
  float power=-0.5*(vConicXY.x*d.x*d.x+vConicYY*d.y*d.y)-vConicXY.y*d.x*d.y;
  if(power>0.0) discard;
  float alpha=min(0.99,vOpacity*exp(power));
  if(alpha<1.0/255.0) discard;
  fragColor=vec4(vColor,alpha);
}
`

type SplatRenderer struct {
	data                        *SplatData
	vao, ebo                    uint32
	prog uint32
	vbos                    []uint32
	uView, uProj             int32
	uFocal, uTanFov, uKernel    int32
	uOpaque, uColorize, uSizeMul  int32
	uFlatColor                      int32
	order []uint32
	tmp                     []float32
	pos, col, op, sc, qu        int32
}

func compileShader(typ uint32, src string) (uint32, error) {
	s := gl.CreateShader(typ)
	cs, free := gl.Strs(src + "\x00")
	defer free()
	gl.ShaderSource(s, 1, cs, nil)
	gl.CompileShader(s)
	var ok int32
	gl.GetShaderiv(s, gl.COMPILE_STATUS, &ok)
	if ok == 0 {
		var l int32
		gl.GetShaderiv(s, gl.INFO_LOG_LENGTH, &l)
		logb := make([]byte, l)
		gl.GetShaderInfoLog(s, l, nil, &logb[0])
		return 0, fmt.Errorf("shader compile: %s", string(logb))
	}
	return s, nil
}

func linkProgram(vs, fs uint32) (uint32, error) {
	p := gl.CreateProgram()
	gl.AttachShader(p, vs)
	gl.AttachShader(p, fs)
	gl.LinkProgram(p)
	var ok int32
	gl.GetProgramiv(p, gl.LINK_STATUS, &ok)
	if ok == 0 {
		var l int32
		gl.GetProgramiv(p, gl.INFO_LOG_LENGTH, &l)
		lg := make([]byte, l)
		gl.GetProgramInfoLog(p, l, nil, &lg[0])
		return 0, fmt.Errorf("program link: %s", string(lg))
	}
	return p, nil
}

func newSplatRenderer(d *SplatData) (*SplatRenderer, error) {
	vs, err := compileShader(gl.VERTEX_SHADER, splatVertSrc)
	if err != nil {
		return nil, err
	}
	fs, err := compileShader(gl.FRAGMENT_SHADER, splatFragSrc)
	if err != nil {
		return nil, err
	}
	prog, err := linkProgram(vs, fs)
	if err != nil {
		return nil, err
	}
	gl.DeleteShader(vs)
	gl.DeleteShader(fs)
	r := &SplatRenderer{data: d, prog: prog, order: make([]uint32, d.N), tmp: make([]float32, d.N)}
	r.uView = gl.GetUniformLocation(prog, gl.Str("uView\x00"))
	r.uProj = gl.GetUniformLocation(prog, gl.Str("uProj\x00"))
	r.uFocal = gl.GetUniformLocation(prog, gl.Str("uFocal\x00"))
	r.uTanFov = gl.GetUniformLocation(prog, gl.Str("uTanFov\x00"))
	r.uKernel = gl.GetUniformLocation(prog, gl.Str("uKernel\x00"))
	r.uOpaque = gl.GetUniformLocation(prog, gl.Str("uOpaque\x00"))
	r.uColorize = gl.GetUniformLocation(prog, gl.Str("uColorize\x00"))
	r.uSizeMul = gl.GetUniformLocation(prog, gl.Str("uSizeMul\x00"))
	r.uFlatColor = gl.GetUniformLocation(prog, gl.Str("uFlatColor\x00"))

	gl.GenVertexArrays(1, &r.vao)
	gl.BindVertexArray(r.vao)
	attr := func(name string, data []float32, size int32) {
		var vbo uint32
		gl.GenBuffers(1, &vbo)
		r.vbos = append(r.vbos, vbo)
		gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
		gl.BufferData(gl.ARRAY_BUFFER, len(data)*4, gl.Ptr(data), gl.STATIC_DRAW)
		loc := gl.GetAttribLocation(prog, gl.Str(name + "\x00"))
		gl.EnableVertexAttribArray(uint32(loc))
		gl.VertexAttribPointer(uint32(loc), size, gl.FLOAT, false, 0, gl.PtrOffset(0))
	}
	attr("position", d.Centers, 3)
	attr("aColor", d.Colors, 3)
	attr("aOpacity", d.Opacity, 1)
	attr("aScale", d.Scales, 3)
	attr("aRot", d.Quats, 4)
	gl.GenBuffers(1, &r.ebo)
	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, r.ebo)
	gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(r.order)*4, gl.Ptr(r.order), gl.DYNAMIC_DRAW)
	gl.BindVertexArray(0)
	return r, nil
}

func (r *SplatRenderer) setCamUniforms(cam *OrbitCamera, o RenderOpts) {
	gl.UseProgram(r.prog)
	v := cam.view()
	p := cam.proj()
	gl.UniformMatrix4fv(r.uView, 1, false, &v[0])
	gl.UniformMatrix4fv(r.uProj, 1, false, &p[0])
	tanY := float32(math.Tan(cam.FovY * 0.5))
	tanX := tanY * float32(cam.Width) / float32(cam.Height)
	gl.Uniform2f(r.uTanFov, tanX, tanY)
	gl.Uniform2f(r.uFocal, float32(cam.Width)/(2*tanX), float32(cam.Height)/(2*tanY))
	gl.Uniform1f(r.uKernel, splatKernel)
	opa := float32(0)
	if o.Opaque {
		opa = 1
	}
	col := float32(1)
	if !o.Colorize {
		col = 0
	}
	gl.Uniform1f(r.uOpaque, opa)
	gl.Uniform1f(r.uColorize, col)
	gl.Uniform1f(r.uSizeMul, o.SizeMul)
	gl.Uniform3f(r.uFlatColor, o.FlatColor[0], o.FlatColor[1], o.FlatColor[2])
}

const splatBuckets = 1024

func (r *SplatRenderer) sort(cam *OrbitCamera) {
	v := cam.view()
	d := r.data.Centers
	n := r.data.N
	minD, maxD := float32(1e30), float32(-1e30)
	for i := 0; i < n; i++ {
		z := v[2]*d[i*3] + v[6]*d[i*3+1] + v[10]*d[i*3+2] + v[14]
		r.tmp[i] = z
		if z < minD { minD = z }
		if z > maxD { maxD = z }
	}
	if maxD-minD < 1e-6 {
		maxD = minD + 1
	}
	cur := make([]uint32, splatBuckets+1)
	inv := float32(splatBuckets) / (maxD - minD)
	for i := 0; i < n; i++ {
		b := int((r.tmp[i] - minD) * inv)
		if b >= splatBuckets { b = splatBuckets - 1 }
		if b < 0 { b = 0 }
		cur[b+1]++
	}
	for b := 0; b < splatBuckets; b++ { cur[b+1] += cur[b] }
	for i := 0; i < n; i++ {
		b := int((r.tmp[i] - minD) * inv)
		if b >= splatBuckets { b = splatBuckets - 1 }
		if b < 0 { b = 0 }
		r.order[cur[b]] = uint32(i)
		cur[b]++
	}
}

func (r *SplatRenderer) dispose() {
	for _, v := range r.vbos {
		gl.DeleteBuffers(1, &v)
	}
	gl.DeleteVertexArrays(1, &r.vao)
	gl.DeleteBuffers(1, &r.ebo)
	gl.DeleteProgram(r.prog)
}

func (r *SplatRenderer) draw(cam *OrbitCamera, o RenderOpts) {
	r.setCamUniforms(cam, o)
	r.sort(cam)
	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, r.ebo)
	gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(r.order)*4, gl.Ptr(r.order), gl.DYNAMIC_DRAW)
	gl.Enable(gl.PROGRAM_POINT_SIZE)
	gl.Enable(gl.BLEND)
	gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
	gl.Disable(gl.DEPTH_TEST)
	gl.BindVertexArray(r.vao)
	gl.DrawElements(gl.POINTS, int32(r.data.N), gl.UNSIGNED_INT, gl.PtrOffset(0))
	gl.BindVertexArray(0)
}
