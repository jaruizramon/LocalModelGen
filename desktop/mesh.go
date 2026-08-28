package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"strings"

	"github.com/go-gl/gl/v3.3-core/gl"
)

// MeshData is a triangulated mesh loaded from a TRELLIS .glb export.
type MeshData struct {
	Pos   []float32 // xyz per vertex
	Norm  []float32 // xyz per vertex
	Idx   []uint32  // triangle indices
	UV    []float32 // uv per vertex (2 floats) — may be empty
	Tex   []byte    // baseColor image bytes (png/jpeg) — may be nil
	NV    int
	MinBB [3]float32
	MaxBB [3]float32
}

// gltfDoc is the subset of the glTF JSON we need: the first primitive's
// POSITION/indices/TEXCOORD_0 and the material's baseColor texture image.
type gltfDoc struct {
	Accessors []struct {
		BufferView    int    `json:"bufferView"`
		ByteOffset    int    `json:"byteOffset"`
		ComponentType int    `json:"componentType"`
		Count         int    `json:"count"`
		Type          string `json:"type"`
	} `json:"accessors"`
	BufferViews []struct {
		Buffer     int `json:"buffer"`
		ByteOffset int `json:"byteOffset"`
		ByteLength int `json:"byteLength"`
		ByteStride int `json:"byteStride"`
	} `json:"bufferViews"`
	Meshes []struct {
		Primitives []struct {
			Attributes map[string]int `json:"attributes"`
			Indices    int            `json:"indices"`
			Mode       int            `json:"mode"`
		} `json:"primitives"`
	} `json:"meshes"`
	Images []struct {
		BufferView int    `json:"bufferView"`
		MimeType   string `json:"mimeType"`
	} `json:"images"`
	Textures []struct {
		Source int `json:"source"`
	} `json:"textures"`
	Materials []struct {
		PBR struct {
			BaseColorTexture struct {
				Index int `json:"index"`
			} `json:"baseColorTexture"`
		} `json:"pbrMetallicRoughness"`
	} `json:"materials"`
}

// parseGLBMesh reads the first mesh primitive of a binary glTF (.glb):
// POSITION (VEC3 float), indices, TEXCOORD_0, and the baseColor texture. The
// TRELLIS worker writes a single-primitive mesh, so we take primitives[0].
func parseGLBMesh(path string) (*MeshData, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) < 20 || string(b[0:4]) != "glTF" {
		return nil, fmt.Errorf("not a GLB: %s", path)
	}
	le := binary.LittleEndian
	var jsonChunk, binChunk []byte
	for off := 12; off+8 <= len(b); {
		clen := int(le.Uint32(b[off:]))
		ctype := le.Uint32(b[off+4:])
		data := b[off+8 : off+8+clen]
		switch ctype {
		case 0x4E4F534A:
			jsonChunk = data
		case 0x004E4942:
			binChunk = data
		}
		off += 8 + clen
	}
	if jsonChunk == nil {
		return nil, fmt.Errorf("GLB has no JSON chunk")
	}
	var doc gltfDoc
	if err := json.Unmarshal(jsonChunk, &doc); err != nil {
		return nil, fmt.Errorf("gltf json: %w", err)
	}
	if len(doc.Meshes) == 0 || len(doc.Meshes[0].Primitives) == 0 {
		return nil, fmt.Errorf("no mesh primitives")
	}
	prim := doc.Meshes[0].Primitives[0]

	readAccessor := func(ai int) ([]byte, int, error) {
		if ai < 0 || ai >= len(doc.Accessors) {
			return nil, 0, fmt.Errorf("bad accessor %d", ai)
		}
		a := doc.Accessors[ai]
		if a.BufferView < 0 || a.BufferView >= len(doc.BufferViews) {
			return nil, 0, fmt.Errorf("accessor %d: bad bufferView", ai)
		}
		bv := doc.BufferViews[a.BufferView]
		start := bv.ByteOffset + a.ByteOffset
		if start < 0 || start > len(binChunk) {
			return nil, 0, fmt.Errorf("accessor %d: offset out of range", ai)
		}
		return binChunk[start:], a.ComponentType, nil
	}

	posAI, ok := prim.Attributes["POSITION"]
	if !ok {
		return nil, fmt.Errorf("no POSITION attribute")
	}
	posBytes, compType, err := readAccessor(posAI)
	if err != nil {
		return nil, err
	}
	if compType != 5126 {
		return nil, fmt.Errorf("POSITION componentType %d (want float)", compType)
	}
	a := doc.Accessors[posAI]
	n := a.Count
	pos := make([]float32, n*3)
	for i := 0; i < n; i++ {
		x := math.Float32frombits(le.Uint32(posBytes[i*12:]))
		y := math.Float32frombits(le.Uint32(posBytes[i*12+4:]))
		z := math.Float32frombits(le.Uint32(posBytes[i*12+8:]))
		// The glb is glTF Y-up already; the standard camera shows it correctly.
		// No extra re-rotation (an (x,-z,y) spin tipped it upside-down).
		pos[i*3] = x
		pos[i*3+1] = y
		pos[i*3+2] = z
	}

	idx := []uint32{}
	if prim.Indices >= 0 {
		idxBytes, icomp, err := readAccessor(prim.Indices)
		if err != nil {
			return nil, err
		}
		ia := doc.Accessors[prim.Indices]
		idx = make([]uint32, ia.Count)
		if icomp == 5125 {
			for i := 0; i < ia.Count; i++ {
				idx[i] = le.Uint32(idxBytes[i*4:])
			}
		} else if icomp == 5123 {
			for i := 0; i < ia.Count; i++ {
				idx[i] = uint32(le.Uint16(idxBytes[i*2:]))
			}
		} else {
			return nil, fmt.Errorf("indices componentType %d", icomp)
		}
	} else {
		for i := 0; i < n; i++ {
			idx = append(idx, uint32(i))
		}
	}

	// TRELLIS exports the glb with inward (clockwise) winding; flip each
	// triangle so the face normals point OUTWARD (correct lighting + culling).
	for i := 0; i+2 < len(idx); i += 3 {
		idx[i+1], idx[i+2] = idx[i+2], idx[i+1]
	}

	uv := make([]float32, 0)
	if uvAI, ok := prim.Attributes["TEXCOORD_0"]; ok && uvAI >= 0 && uvAI < len(doc.Accessors) {
		uvBytes, ucomp, err := readAccessor(uvAI)
		if err == nil && ucomp == 5126 {
			uv = make([]float32, n*2)
			for i := 0; i < n; i++ {
				uv[i*2] = math.Float32frombits(le.Uint32(uvBytes[i*8:]))
				uv[i*2+1] = math.Float32frombits(le.Uint32(uvBytes[i*8+4:]))
			}
		}
	}

	tex := baseColorImage(&doc, binChunk)

	md := &MeshData{Pos: pos, Idx: idx, UV: uv, Tex: tex, NV: n}
	md.computeNormals()
	md.centerAndScale()
	return md, nil
}

// baseColorImage pulls the material's baseColor texture image bytes (embedded
// png/jpeg) out of the BIN chunk, or nil if the glb has no material/texture.
func baseColorImage(doc *gltfDoc, binChunk []byte) []byte {
	if len(doc.Materials) == 0 {
		return nil
	}
	texIdx := doc.Materials[0].PBR.BaseColorTexture.Index
	if texIdx < 0 || texIdx >= len(doc.Textures) {
		return nil
	}
	src := doc.Textures[texIdx].Source
	if src < 0 || src >= len(doc.Images) {
		return nil
	}
	bv := doc.Images[src].BufferView
	if bv < 0 || bv >= len(doc.BufferViews) {
		return nil
	}
	b := doc.BufferViews[bv]
	start := b.ByteOffset
	if start < 0 || start > len(binChunk) {
		return nil
	}
	end := start + b.ByteLength
	if end > len(binChunk) {
		end = len(binChunk)
	}
	return binChunk[start:end]
}

// parseObjMesh reads a Wavefront .obj (v x y z / f a b c). Accepts the
// Open3D-produced variants (v with trailing r g b, interleaved vn lines) --
// only lines starting "v " / "f " are geometry; "vn"/"vt" are skipped.
func parseObjMesh(path string) (*MeshData, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var pos []float32
	var idx []uint32
	for _, line := range bytes.Split(b, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if line[0] == 'v' && line[1] == ' ' && len(line) > 2 {
			var x, y, z float32
			fmt.Sscanf(string(line[2:]), "%f %f %f", &x, &y, &z)
			pos = append(pos, x, y, z)
		} else if line[0] == 'f' && line[1] == ' ' && len(line) > 2 {
			// tokens may be "1", "1/2" or "1//2" (v/vt/vn) -- take the vertex
			// index only; %d on "6//6" reads 6 but then the next %d fails and
			// the whole face collapses to (0,0,0).
			var t []string = strings.Fields(string(line[2:]))
			if len(t) >= 3 {
				var a, b, c int
				na, ea := fmt.Sscanf(t[0], "%d", &a)
				nb, eb := fmt.Sscanf(t[1], "%d", &b)
				nc, ec := fmt.Sscanf(t[2], "%d", &c)
				if na == 1 && ea == nil && nb == 1 && eb == nil && nc == 1 && ec == nil {
					idx = append(idx, uint32(a-1), uint32(b-1), uint32(c-1))
				}
			}
		}
	}
	if len(pos) < 3 || len(idx) < 3 {
		return nil, fmt.Errorf("empty obj: %s", path)
	}
	md := &MeshData{Pos: pos, Idx: idx, NV: len(pos) / 3}
	md.computeNormals()
	md.centerAndScale()
	return md, nil
}

// computeNormals does per-vertex normals by summing incident face normals
// (smooth shading) so the surface reads cleanly under lighting.
func (d *MeshData) computeNormals() {
	d.Norm = make([]float32, d.NV*3)
	for i := 0; i < len(d.Idx); i += 3 {
		a, bb, c := d.Idx[i], d.Idx[i+1], d.Idx[i+2]
		if a >= uint32(d.NV) || bb >= uint32(d.NV) || c >= uint32(d.NV) {
			continue
		}
		ax, ay, az := d.Pos[a*3], d.Pos[a*3+1], d.Pos[a*3+2]
		bx, by, bz := d.Pos[bb*3], d.Pos[bb*3+1], d.Pos[bb*3+2]
		cx, cy, cz := d.Pos[c*3], d.Pos[c*3+1], d.Pos[c*3+2]
		ux, uy, uz := bx-ax, by-ay, bz-az
		vx, vy, vz := cx-ax, cy-ay, cz-az
		nx := uy*vz - uz*vy
		ny := uz*vx - ux*vz
		nz := ux*vy - uy*vx
		d.Norm[a*3] += nx
		d.Norm[a*3+1] += ny
		d.Norm[a*3+2] += nz
		d.Norm[bb*3] += nx
		d.Norm[bb*3+1] += ny
		d.Norm[bb*3+2] += nz
		d.Norm[c*3] += nx
		d.Norm[c*3+1] += ny
		d.Norm[c*3+2] += nz
	}
	for i := 0; i < d.NV; i++ {
		x, y, z := d.Norm[i*3], d.Norm[i*3+1], d.Norm[i*3+2]
		l := math.Sqrt(float64(x*x + y*y + z*z))
		if l == 0 {
			continue
		}
		d.Norm[i*3] = float32(float64(x) / l)
		d.Norm[i*3+1] = float32(float64(y) / l)
		d.Norm[i*3+2] = float32(float64(z) / l)
	}
}

// centerAndScale frames the mesh like the splat view (centered, ~1.7 span).
func (d *MeshData) centerAndScale() {
	minX, minY, minZ := float32(1e9), float32(1e9), float32(1e9)
	maxX, maxY, maxZ := float32(-1e9), float32(-1e9), float32(-1e9)
	for i := 0; i < d.NV; i++ {
		x, y, z := d.Pos[i*3], d.Pos[i*3+1], d.Pos[i*3+2]
		if x < minX {
			minX = x
		}
		if x > maxX {
			maxX = x
		}
		if y < minY {
			minY = y
		}
		if y > maxY {
			maxY = y
		}
		if z < minZ {
			minZ = z
		}
		if z > maxZ {
			maxZ = z
		}
	}
	cx, cy, cz := (minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2
	size := float32(math.Sqrt(float64((maxX-minX)*(maxX-minX) + (maxY-minY)*(maxY-minY) + (maxZ-minZ)*(maxZ-minZ))))
	if size == 0 {
		size = 1
	}
	s := 1.7 / size
	for i := 0; i < d.NV; i++ {
		d.Pos[i*3] = (d.Pos[i*3] - cx) * s
		d.Pos[i*3+1] = (d.Pos[i*3+1] - cy) * s
		d.Pos[i*3+2] = (d.Pos[i*3+2] - cz) * s
	}
	d.MinBB = [3]float32{minX, minY, minZ}
	d.MaxBB = [3]float32{maxX, maxY, maxZ}
}

const meshVertSrc = `#version 150 core
in vec3 position;
in vec3 normal;
in vec2 uv;
uniform mat4 uView, uProj, uModel;
uniform vec3 uColor;
uniform float uFlat;
out vec3 vColor;
out vec2 fUV;
void main(){
  vec3 n = normalize(mat3(uModel) * normal);
  vec3 lightDir = normalize(vec3(0.5, 0.8, 0.6));
  float diff = max(dot(n, lightDir), 0.0);
  float rim = 0.35 + 0.65 * abs(n.z);
  vec3 base = mix(uColor, uColor * (0.35 + 0.65*diff), uFlat);
  vColor = base * rim;
  fUV = uv;
  gl_Position = uProj * uView * uModel * vec4(position, 1.0);
}
`

const meshFragSrc = `#version 150 core
in vec3 vColor;
in vec2 fUV;
uniform sampler2D uTex;
uniform float uHasTex;
out vec4 fragColor;
void main(){
  vec3 col = vColor;
  if(uHasTex > 0.5){
    vec4 t = texture(uTex, fUV);
    col = mix(vColor, t.rgb, 1.0);
  }
  fragColor = vec4(col, 1.0);
}
`

type MeshRenderer struct {
	data                        *MeshData
	vao, posVbo, nrmVbo, uvVbo  uint32
	ebo                         uint32
	tex                         uint32
	hasTex                      bool
	prog                        uint32
	uView, uProj, uModel, uColor, uFlat int32
	uTex, uHasTex               int32
	indexCount                  int32
	wireframe                   bool
}

func newMeshRenderer(d *MeshData) (*MeshRenderer, error) {
	vs, err := compileShader(gl.VERTEX_SHADER, meshVertSrc)
	if err != nil {
		return nil, err
	}
	fs, err := compileShader(gl.FRAGMENT_SHADER, meshFragSrc)
	if err != nil {
		return nil, err
	}
	prog, err := linkProgram(vs, fs)
	if err != nil {
		return nil, err
	}
	gl.DeleteShader(vs)
	gl.DeleteShader(fs)
	r := &MeshRenderer{data: d, prog: prog, indexCount: int32(len(d.Idx))}
	r.uView = gl.GetUniformLocation(prog, gl.Str("uView\x00"))
	r.uProj = gl.GetUniformLocation(prog, gl.Str("uProj\x00"))
	r.uModel = gl.GetUniformLocation(prog, gl.Str("uModel\x00"))
	r.uColor = gl.GetUniformLocation(prog, gl.Str("uColor\x00"))
	r.uFlat = gl.GetUniformLocation(prog, gl.Str("uFlat\x00"))
	r.uTex = gl.GetUniformLocation(prog, gl.Str("uTex\x00"))
	r.uHasTex = gl.GetUniformLocation(prog, gl.Str("uHasTex\x00"))

	gl.GenVertexArrays(1, &r.vao)
	gl.BindVertexArray(r.vao)
	attr := func(name string, data []float32, size int32) {
		var vbo uint32
		gl.GenBuffers(1, &vbo)
		switch name {
		case "position":
			r.posVbo = vbo
		case "normal":
			r.nrmVbo = vbo
		case "uv":
			r.uvVbo = vbo
		}
		gl.BindBuffer(gl.ARRAY_BUFFER, vbo)
		gl.BufferData(gl.ARRAY_BUFFER, len(data)*4, gl.Ptr(data), gl.STATIC_DRAW)
		loc := gl.GetAttribLocation(prog, gl.Str(name+"\x00"))
		gl.EnableVertexAttribArray(uint32(loc))
		gl.VertexAttribPointer(uint32(loc), size, gl.FLOAT, false, 0, gl.PtrOffset(0))
	}
	attr("position", d.Pos, 3)
	attr("normal", d.Norm, 3)
	if len(d.UV) >= 2*d.NV {
		attr("uv", d.UV, 2)
	} else {
		// no UVs: point everything at (0,0) so sampling is safe
		dummy := make([]float32, d.NV*2)
		attr("uv", dummy, 2)
	}
	gl.GenBuffers(1, &r.ebo)
	gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, r.ebo)
	gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(d.Idx)*4, gl.Ptr(d.Idx), gl.STATIC_DRAW)
	gl.BindVertexArray(0)

	// baked texture
	if len(d.Tex) > 0 {
		if img, _, err := image.Decode(bytes.NewReader(d.Tex)); err == nil {
			rgba := toRGBA(img)
			gl.GenTextures(1, &r.tex)
			gl.BindTexture(gl.TEXTURE_2D, r.tex)
			gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, int32(rgba.Bounds().Dx()), int32(rgba.Bounds().Dy()), 0, gl.RGBA, gl.UNSIGNED_BYTE, gl.Ptr(rgba.Pix))
			gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
			gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
			gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
			gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)
			gl.BindTexture(gl.TEXTURE_2D, 0)
			r.hasTex = true
		}
	}
	return r, nil
}

func toRGBA(img image.Image) *image.RGBA {
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba
	}
	b := img.Bounds()
	rgba := image.NewRGBA(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			rgba.Set(x, y, img.At(x, y))
		}
	}
	return rgba
}

func (r *MeshRenderer) setCamUniforms(cam *OrbitCamera, o RenderOpts) {
	gl.UseProgram(r.prog)
	v := cam.view()
	p := cam.proj()
	gl.UniformMatrix4fv(r.uView, 1, false, &v[0])
	gl.UniformMatrix4fv(r.uProj, 1, false, &p[0])
	var m [16]float32
	m[0], m[5], m[10], m[15] = 1, 1, 1, 1
	gl.UniformMatrix4fv(r.uModel, 1, false, &m[0])
	gl.Uniform3f(r.uColor, o.FlatColor[0], o.FlatColor[1], o.FlatColor[2])
	gl.Uniform1f(r.uFlat, 1.0)
	if r.hasTex {
		gl.Uniform1i(r.uTex, 0)
		gl.Uniform1f(r.uHasTex, 1.0)
	} else {
		gl.Uniform1f(r.uHasTex, 0.0)
	}
	r.wireframe = o.Wireframe
}

func (r *MeshRenderer) draw() {
	gl.UseProgram(r.prog)
	// The mesh is often an OPEN / fragmented surface (128-res decode of a
	// complex object), so culling backfaces shows through the holes -> looks
	// like shredded paper. Draw both sides: front occludes, back faces fill
	// the gaps via depth, so the surface reads as one piece.
	gl.Disable(gl.BLEND)
	gl.Enable(gl.DEPTH_TEST)
	gl.DepthFunc(gl.LESS)
	gl.BindVertexArray(r.vao)
	if r.hasTex {
		gl.ActiveTexture(gl.TEXTURE0)
		gl.BindTexture(gl.TEXTURE_2D, r.tex)
	}
	if r.wireframe {
		gl.PolygonMode(gl.FRONT_AND_BACK, gl.LINE)
	} else {
		gl.PolygonMode(gl.FRONT_AND_BACK, gl.FILL)
	}
	gl.DrawElements(gl.TRIANGLES, r.indexCount, gl.UNSIGNED_INT, gl.PtrOffset(0))
	gl.PolygonMode(gl.FRONT_AND_BACK, gl.FILL)
	gl.BindTexture(gl.TEXTURE_2D, 0)
	gl.BindVertexArray(0)
	gl.Disable(gl.CULL_FACE)
}

func (r *MeshRenderer) dispose() {
	if r.vao != 0 {
		gl.DeleteVertexArrays(1, &r.vao)
	}
	if r.posVbo != 0 {
		gl.DeleteBuffers(1, &r.posVbo)
	}
	if r.nrmVbo != 0 {
		gl.DeleteBuffers(1, &r.nrmVbo)
	}
	if r.uvVbo != 0 {
		gl.DeleteBuffers(1, &r.uvVbo)
	}
	if r.ebo != 0 {
		gl.DeleteBuffers(1, &r.ebo)
	}
	if r.tex != 0 {
		gl.DeleteTextures(1, &r.tex)
	}
	if r.prog != 0 {
		gl.DeleteProgram(r.prog)
	}
}
