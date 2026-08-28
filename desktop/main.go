package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/go-gl/gl/v3.3-core/gl"
	"github.com/go-gl/glfw/v3.3/glfw"
	imgui "github.com/micahke/imgui-go"
	ibackend "localmodelgen/desktop/ibackend"
)

var gwin *glfw.Window

func init() { runtime.LockOSThread() }

const viewW, viewH = 760, 560

func latestPly() string {
	matches, _ := filepath.Glob(filepath.Join("..", "tmp", "*.ply"))
	if len(matches) == 0 {
		return ""
	}
	sort.Sort(sort.Reverse(sort.StringSlice(matches)))
	return matches[0]
}
func latestGlb() string {
	matches, _ := filepath.Glob(filepath.Join("..", "tmp", "*.glb"))
	if len(matches) == 0 {
		return ""
	}
	sort.Sort(sort.Reverse(sort.StringSlice(matches)))
	return matches[0]
}
// meshComponentCount returns the number of connected vertex-components of a
// mesh (union-find over the triangle indices). A 128-res decode of a complex
// object routinely splits into many islands; >3 is a good "fragmented" signal.
func meshComponentCount(md *MeshData) int {
	n := md.NV
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	for i := 0; i+2 < len(md.Idx); i += 3 {
		a, b, c := int(md.Idx[i]), int(md.Idx[i+1]), int(md.Idx[i+2])
		if a >= n || b >= n || c >= n {
			continue
		}
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
		rc := find(c)
		if find(b) != rc {
			parent[find(b)] = rc
		}
	}
	seen := map[int]bool{}
	for i := 0; i+2 < len(md.Idx); i += 3 {
		seen[find(int(md.Idx[i]))] = true
	}
	return len(seen)
}

type RenderOpts struct {
	Opaque   bool
	Colorize bool
	FlatColor [3]float32
	SizeMul  float32
	Wireframe bool
}

type appState struct {
	seed                                            int32
	ssCfg, slatCfg                                  float32
	ssSteps, slatSteps                              int32
	tris, texSize, subsampleRes                      int32
	smoothIters                                     int32
	smooth                                          bool
	cleanupMesh                                     bool
	offload                                         float32
	imagePath                                       string
	img                                             *image.RGBA
	imgTex, imgW, imgH                              int32
	mask                                            []pt
	maskClosed, maskEditing                         bool
	undo, redo                                      [][]pt
	dragIdx                                         int
	status                                          string
	phase                                           string
	meshHint                                        string
	pendingObj                                      string
	glb, ply, zip, blend                            string
	busy                                            bool
	needReload                                      bool
	flash                                           int
	renderOpts                                      RenderOpts
	showMaskFill                                    bool
	model                                           string
	models                                          []string
}

func homeDir() string {
	h, _ := os.UserHomeDir()
	if h == "" { return "/home/pipo/Pictures" }
	return h + "/Pictures"
}

func browseFile() (string, error) {
	cmd := exec.Command("zenity", "--file-selection",
		"--title=Select an image",
		"--filename="+homeDir(),
		"--file-filter=Images | *.png *.PNG *.jpg *.jpeg *.JPG *.webp")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func main() {
	if err := glfw.Init(); err != nil {
		log.Fatalf("glfw init: %v", err)
	}
	defer glfw.Terminate()
	glfw.WindowHint(glfw.ContextVersionMajor, 3)
	glfw.WindowHint(glfw.ContextVersionMinor, 3)
	glfw.WindowHint(glfw.OpenGLProfile, glfw.OpenGLCoreProfile)
	glfw.WindowHint(glfw.OpenGLForwardCompatible, glfw.True)
	win, err := glfw.CreateWindow(1500, 900, "3DModelGen — desktop", nil, nil)
	if err != nil {
		log.Fatalf("window: %v", err)
	}
	win.MakeContextCurrent()
	gwin = win
	glfw.SwapInterval(1)
	if err := gl.Init(); err != nil {
		log.Fatalf("gl init: %v", err)
	}
	log.Printf("GL: %s", gl.GoStr(gl.GetString(gl.VERSION)))

	ctx := imgui.CreateContext(nil)
	defer ctx.Destroy()
	io := imgui.CurrentIO()
	io.SetIniFilename("")
	impl := ibackend.ImguiGlfw3Init(win, io)
	defer impl.Shutdown()

	st := &appState{seed: 1, ssCfg: 7.5, slatCfg: 3.0, ssSteps: 8, slatSteps: 8,
		tris: 50000, texSize: 1024, smooth: true, smoothIters: 10, offload: 1.0, renderOpts: RenderOpts{Colorize: true, SizeMul: 1}, showMaskFill: true, model: "trellis-image"}

	log.Printf("[orch] backend: %s", orchURL())

	if m, err := fetchModels(); err == nil {
		st.models = m
	}

	// ---- splat renderer (latest asset) ----
	var renderer *SplatRenderer
	var splatCount int
	var meshRenderer *MeshRenderer
	var meshCount int
	var viewMode int32
	reload := func() {
		ply := latestPly()
		if ply == "" {
			return
		}
		d, err := parseGaussianPly(ply)
		if err != nil {
			log.Printf("parse: %v", err)
			return
		}
		d.centerAndScale()
		r2, err := newSplatRenderer(d)
		if err != nil {
			log.Printf("renderer: %v", err)
			return
		}
		if renderer != nil {
			renderer.dispose()
		}
		renderer = r2
		splatCount = d.N
		st.ply = ply
		log.Printf("loaded %s (%d splats)", ply, d.N)
		// polygons: newest .glb mesh (for the Splat/Mesh toggle)
		if glb := latestGlb(); glb != "" {
			if md, err := parseGLBMesh(glb); err == nil {
				if meshRenderer != nil {
					meshRenderer.dispose()
				}
				if mr, err := newMeshRenderer(md); err == nil {
					meshRenderer = mr
					meshCount = md.NV
					nc := meshComponentCount(md)
					if nc > 3 {
						st.meshHint = fmt.Sprintf("mesh fragmented (%d pieces) — try a simpler mask or turn on 'cleanup mesh'", nc)
					} else {
						st.meshHint = ""
					}
					log.Printf("loaded mesh %s (%d verts, %d tris, %d pieces)", glb, md.NV, len(md.Idx)/3, nc)
				}
			}
		}
	}
	reload()

	// ---- FBO viewport ----
	var fbo, fboTex, fboDepth uint32
	var curVpW, curVpH = viewW, viewH
	gl.GenFramebuffers(1, &fbo)
	gl.GenTextures(1, &fboTex)
	gl.BindTexture(gl.TEXTURE_2D, fboTex)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, viewW, viewH, 0, gl.RGBA, gl.UNSIGNED_BYTE, nil)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.GenRenderbuffers(1, &fboDepth)
	gl.BindRenderbuffer(gl.RENDERBUFFER, fboDepth)
	gl.RenderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, viewW, viewH)
	gl.BindFramebuffer(gl.FRAMEBUFFER, fbo)
	gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0)
	gl.FramebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, fboDepth)
	gl.BindFramebuffer(gl.FRAMEBUFFER, 0)

	cam := &OrbitCamera{Yaw: 2.3, Pitch: 0.35, Distance: 2.6, FovY: 0.62, Width: viewW, Height: viewH}

	refreshMask := func() {
		if st.img == nil {
			return
		}
		scale := float64(st.imgW) / float64(minInt(st.imgW, 440))
		if scale < 1 { scale = 1 }
		overlay := drawPolygonOverlay(st.img, st.mask, st.maskClosed, scale, st.showMaskFill)
		st.imgTex = uploadRGBATexture(overlay, uint32(st.imgTex))
	}

	loadImage := func(path string) {
		f, err := os.Open(path)
		if err != nil {
			st.status = "image open: " + err.Error()
			return
		}
		defer f.Close()
		im, err := png.Decode(f)
		if err != nil {
			f.Seek(0, 0)
			if im, err = jpeg.Decode(f); err != nil {
				st.status = "image decode: " + err.Error()
				return
			}
		}
		st.img = imageToRGBA(im)
		st.imgW, st.imgH = int32(st.img.Rect.Dx()), int32(st.img.Rect.Dy())
		st.imgTex = uploadRGBATexture(st.img, 0)
		st.mask = nil
		st.maskClosed = false
		st.status = fmt.Sprintf("loaded %s (%dx%d)", path, st.imgW, st.imgH)
	}

	for !win.ShouldClose() {
		glfw.PollEvents()
		impl.NewFrame()

		if st.maskEditing && ctrlHeld() {
			if imgui.IsKeyPressed(imgui.KeyZ) { st.undoMask(); refreshMask() }
			if imgui.IsKeyPressed(imgui.KeyY) { st.redoMask(); refreshMask() }
		}
		if st.needReload {
			reload()
			st.needReload = false
		}
		if st.pendingObj != "" {
			obj := st.pendingObj
			st.pendingObj = ""
			if md, err := parseObjMesh(obj); err != nil {
				st.status = "mesh parse: " + err.Error()
			} else if mr, err := newMeshRenderer(md); err == nil {
				if meshRenderer != nil {
					meshRenderer.dispose()
				}
				meshRenderer = mr
				meshCount = md.NV
				viewMode = 1
				st.meshHint = ""
				st.status = fmt.Sprintf("mesh generated (CPU gs2mesh) · %d tris", len(md.Idx)/3)
			} else {
				st.status = "renderer: " + err.Error()
			}
		}
		fw, fh := win.GetSize()
		if fw <= 0 || fh <= 0 {
			fw, fh = 1500, 900
		}
		vpw, vph := fw-470-24, fh-64
		if vpw < 100 {
			vpw = 100
		}
		if vph < 100 {
			vph = 100
		}
		if vpw != curVpW || vph != curVpH {
			curVpW, curVpH = vpw, vph
			gl.BindTexture(gl.TEXTURE_2D, fboTex)
			gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, int32(vpw), int32(vph), 0, gl.RGBA, gl.UNSIGNED_BYTE, nil)
			gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
			gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
			gl.BindRenderbuffer(gl.RENDERBUFFER, fboDepth)
			gl.RenderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, int32(vpw), int32(vph))
			cam.Width, cam.Height = vpw, vph
		}
		imgui.SetNextWindowPos(imgui.Vec2{X: 0, Y: 0})
		imgui.SetNextWindowSize(imgui.Vec2{X: float32(fw), Y: float32(fh)})
		imgui.BeginV("3DModelGen — desktop", nil, imgui.WindowFlagsNoTitleBar|imgui.WindowFlagsNoMove|imgui.WindowFlagsNoResize|imgui.WindowFlagsNoCollapse)

		// ---- LEFT: form ----
		imgui.BeginChildV("left", imgui.Vec2{X: 470, Y: 0}, false, 0)
		imgui.Text("Image")
		imgui.InputTextWithHint("##imagepath", "path to image", &st.imagePath, 0, nil)
		imgui.SameLine()
		if imgui.Button("Browse") {
			if p, err := browseFile(); err == nil && p != "" {
				st.imagePath = p
				loadImage(p)
			}
		}
		imgui.SameLine()
		if imgui.Button("Load") && st.imagePath != "" {
			loadImage(st.imagePath)
		}
		if st.img != nil && st.imgTex != 0 {
			dw := float32(minInt(st.imgW, 440))
			dh := float32(st.imgH) * dw / float32(st.imgW)
			drawImageWidget(imgui.TextureID(uintptr(st.imgTex)), imgui.Vec2{X: dw, Y: dh}, st, refreshMask)
		}
		imgui.Separator()
		imgui.Text("Mask")
		if imgui.Button("Draw mask") {
			st.maskEditing = true
			st.maskClosed = false
			refreshMask()
		}
		imgui.SameLine()
		if imgui.Button("Done") {
			st.maskClosed = true
			st.maskEditing = false
			refreshMask()
		}
		imgui.SameLine()
		if imgui.Button("Clear") {
			st.pushHistory()
			st.mask = nil
			st.maskClosed = false
			refreshMask()
		}
		imgui.SameLine()
		if imgui.Button("Undo") { st.undoMask(); refreshMask() }
		imgui.SameLine()
		if imgui.Button("Redo") { st.redoMask(); refreshMask() }
		imgui.Text(fmt.Sprintf("%d points%s", len(st.mask), maskClosedSuffix(st.maskClosed)))
		imgui.Checkbox("show mask fill", &st.showMaskFill)
		imgui.Separator()
		imgui.Text("Model")
		if imgui.BeginCombo("##model", st.model) {
			for _, m := range st.models {
				if imgui.Selectable(m) {
					st.model = m
				}
			}
			imgui.EndCombo()
		}
		imgui.Separator()
		imgui.Text("Generation")
		imgui.InputInt("seed", &st.seed)
		imgui.Text("Stage 1: Sparse Structure")
		imgui.SliderFloat("ss guidance", &st.ssCfg, 0, 10)
		imgui.SliderInt("ss steps", &st.ssSteps, 1, 50)
		imgui.Text("Stage 2: Structured Latent")
		imgui.SliderFloat("slat guidance", &st.slatCfg, 0, 10)
		imgui.SliderInt("slat steps", &st.slatSteps, 1, 50)
		imgui.SliderInt("target tris", &st.tris, 0, 200000)
		imgui.Text("texture size:")
		imgui.SameLine()
		if imgui.RadioButton("256", st.texSize == 256) { st.texSize = 256 }
		imgui.SameLine()
		if imgui.RadioButton("512", st.texSize == 512) { st.texSize = 512 }
		imgui.SameLine()
		if imgui.RadioButton("1024", st.texSize == 1024) { st.texSize = 1024 }
		imgui.SameLine()
		if imgui.RadioButton("2048", st.texSize == 2048) { st.texSize = 2048 }
		imgui.Text("mesh res: (0=auto)")
		imgui.SameLine()
		if imgui.RadioButton("auto", st.subsampleRes == 0) { st.subsampleRes = 0 }
		imgui.SameLine()
		if imgui.RadioButton("128", st.subsampleRes == 128) { st.subsampleRes = 128 }
		imgui.SameLine()
		if imgui.RadioButton("160", st.subsampleRes == 160) { st.subsampleRes = 160 }
		imgui.SameLine()
		if imgui.RadioButton("192", st.subsampleRes == 192) { st.subsampleRes = 192 }
		imgui.SameLine()
		if 		imgui.RadioButton("res256", st.subsampleRes == 256) { st.subsampleRes = 256 }
		imgui.Checkbox("smooth mesh", &st.smooth)
		imgui.Checkbox("cleanup mesh", &st.cleanupMesh)
		imgui.SliderInt("smooth iters", &st.smoothIters, 0, 20)
		imgui.SliderFloat("offload to ram", &st.offload, 0, 1)
		if imgui.Button("Generate") {
			st.generate()
		}
		imgui.SameLine()
		if imgui.Button("Generate Mesh") && !st.busy {
			go func() {
				st.busy = true
				st.phase = "meshing (CPU)…"
				st.status = "meshing (CPU)…"
				ply := latestPly()
				if ply == "" {
					st.status = "no gaussian (.ply) to mesh."
					st.busy = false
					st.phase = ""
					return
				}
				obj := filepath.Join("..", "tmp", fmt.Sprintf("gs_mesh_%d.obj", time.Now().Unix()))
				cmd := exec.Command(filepath.Join("..", "bin", "gs2mesh"), ply, obj, "128", "0.35")
				if err := cmd.Run(); err != nil {
					st.status = "mesh: " + err.Error()
				} else {
					st.status = "mesh done — loading…"
					// Signal the MAIN loop to load it (single-threaded swap, so
					// the goroutine never races the renderer/render state).
					st.pendingObj = obj
				}
				st.busy = false
				st.phase = ""
			}()
		}
		if imgui.Button("Clear GPU") {
			if j, err := orchClear(); err != nil {
				st.status = "clear: " + err.Error()
			} else {
				st.status = fmt.Sprintf("cleared, gpu %v MB", j["gpu_mb"])
			}
		}
		imgui.Separator()
		if imgui.Button("Convert OBJ") && st.glb != "" {
			if z, err := orchConvert(st.glb); err != nil {
				st.status = "convert: " + err.Error()
			} else {
				st.status = "converted: " + z
			}
		}
		if st.blend != "" {
			imgui.Text("blend: " + st.blend)
			if imgui.Button("Open .blend in Blender") {
				openBlend(st.blend)
			}
		}
		imgui.Text("status: " + st.status)
		imgui.EndChild()

		imgui.SameLine()
		// ---- RIGHT: viewport ----
		imgui.BeginChildV("right", imgui.Vec2{X: 0, Y: 0}, true, 0)
		imgui.Text("Viewport — drag to orbit, scroll to zoom.")
		if st.busy {
			p := st.phase
			if p == "" {
				p = "generating…"
			}
			imgui.Text("⏳ " + p)
		}
		if st.meshHint != "" {
			imgui.Text("⚠ " + st.meshHint)
		}
		if imgui.RadioButton("Splat", viewMode == 0) {
			viewMode = 0
		}
		imgui.SameLine()
		if imgui.RadioButton("Mesh", viewMode == 1) {
			viewMode = 1
		}
		imgui.Separator()
		if viewMode == 0 {
			imgui.Checkbox("Opaque (disable transparency)", &st.renderOpts.Opaque)
			imgui.Checkbox("Baked color", &st.renderOpts.Colorize)
			imgui.SliderFloat("splat size", &st.renderOpts.SizeMul, 0.2, 4.0)
			imgui.ColorEdit3("flat color", &st.renderOpts.FlatColor)
		} else {
			imgui.Checkbox("Wireframe", &st.renderOpts.Wireframe)
			imgui.ColorEdit3("mesh color", &st.renderOpts.FlatColor)
		}
		if (viewMode == 0 && renderer != nil) || (viewMode == 1 && meshRenderer != nil) {
			gl.BindFramebuffer(gl.FRAMEBUFFER, fbo)
			gl.Viewport(0, 0, int32(vpw), int32(vph))
			gl.ClearColor(0.08, 0.09, 0.12, 1.0)
			gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)
			if viewMode == 0 {
				renderer.setCamUniforms(cam, st.renderOpts)
				renderer.draw(cam, st.renderOpts)
			} else {
				meshRenderer.setCamUniforms(cam, st.renderOpts)
				meshRenderer.draw()
			}
			gl.BindFramebuffer(gl.FRAMEBUFFER, 0)
		}
		// The FBO texture's row 0 is the BOTTOM (OpenGL), but imgui's UV origin
		// is top-left -> without flipping V the rendered scene displays
		// upside-down. Flip V so the mesh/splat render upright.
		imgui.ImageV(imgui.TextureID(uintptr(fboTex)), imgui.Vec2{X: float32(vpw), Y: float32(vph)},
			imgui.Vec2{X: 0, Y: 1}, imgui.Vec2{X: 1, Y: 0},
			imgui.Vec4{X: 1, Y: 1, Z: 1, W: 1}, imgui.Vec4{})
		if imgui.IsItemHovered() {
			if imgui.IsMouseDown(0) {
				md := imgui.CurrentIO().GetMouseDelta()
				cam.Yaw -= float64(md.X) * 0.01
				cam.Pitch += float64(md.Y) * 0.01
			}
			if w := imgui.CurrentIO().GetMouseWheelDelta(); w != 0 {
				cam.Distance *= math.Pow(0.9, float64(w))
			}
		}
		if viewMode == 0 {
			imgui.Text("splats: " + itoa(splatCount))
		} else {
			imgui.Text("verts: " + itoa(meshCount))
		}
		imgui.EndChild()
		if st.flash > 0 {
			alpha := float32(st.flash) / 60.0
			col := imgui.Vec4{X: 0.30 * alpha, Y: 0.05 * alpha, Z: 0.55 * alpha, W: 0.55 * alpha}
			imgui.GetWindowDrawList().AddRectFilled(
				imgui.Vec2{X: 0, Y: 0},
				imgui.Vec2{X: float32(fw), Y: float32(fh)},
				col, 0, 0)
			st.flash--
		}
		imgui.End()

		gl.Viewport(0, 0, int32(fw), int32(fh))
		gl.ClearColor(0.10, 0.12, 0.16, 1.0)
		gl.Clear(gl.COLOR_BUFFER_BIT)
		imgui.Render()
		impl.Render(imgui.RenderedDrawData())
		win.SwapBuffers()
	}
}

// drawImageWidget draws the image and, when mask editing, captures clicks into
// normalized points (click = add, double-click = add + close).
func drawImageWidget(id imgui.TextureID, size imgui.Vec2, st *appState, refresh func()) {
	imgui.Image(id, size)
	if !st.maskEditing || !imgui.IsItemHovered() {
		return
	}
	minR := imgui.GetItemRectMin()
	maxR := imgui.GetItemRectMax()
	mp := imgui.MousePos()
	ws, hs := maxR.X-minR.X, maxR.Y-minR.Y
	if ws <= 0 || hs <= 0 {
		return
	}
	ux := clamp01(float64((mp.X - minR.X) / ws))
	uy := clamp01(float64((mp.Y - minR.Y) / hs))
	near, nd := -1, 1e18
	for i, p := range st.mask {
		sx := minR.X + float32(p.X)*ws
		sy := minR.Y + float32(p.Y)*hs
		dx, dy := float64(mp.X-sx), float64(mp.Y-sy)
		d := dx*dx + dy*dy
		if d < nd {
			nd, near = d, i
		}
	}
	nd = math.Sqrt(nd)
	const grabThresh = 14.0
	if imgui.IsMouseDoubleClicked(0) {
		st.pushHistory()
		st.mask = append(st.mask, pt{X: ux, Y: uy})
		st.maskClosed = true
		st.maskEditing = false
		refresh()
	} else if imgui.IsMouseClicked(0) {
		if near >= 0 && nd < grabThresh {
			st.pushHistory()
			st.dragIdx = near
			st.maskClosed = false
		} else {
			st.pushHistory()
			st.mask = append(st.mask, pt{X: ux, Y: uy})
			st.maskClosed = false
			refresh()
		}
	}
	if st.dragIdx >= 0 && imgui.IsMouseDown(0) {
		st.mask[st.dragIdx] = pt{X: ux, Y: uy}
		refresh()
	}
	if imgui.IsMouseReleased(0) {
		st.dragIdx = -1
	}
}

func maskClosedSuffix(closed bool) string {
	if closed {
		return " (closed)"
	}
	return ""
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func (st *appState) generate() {
	if st.busy {
		return
	}
	if st.img == nil {
		st.status = "no image loaded."
		return
	}
	st.busy = true
	st.status = "generating…"
	beep("start")
	genStart := time.Now()
	log.Printf("[gen] start (seed=%d tris=%d)…", st.seed, st.tris)
	go func() {
		toSend := st.img
		if len(st.mask) >= 3 {
			toSend = applyPolygonMask(cloneRGBA(st.img), st.mask)
		}
		j, err := orchGenerate(toSend, genParams{
			Model: st.model,
			Seed: int(st.seed), SSSteps: int(st.ssSteps), SlatSteps: int(st.slatSteps),
			SSCfg: float64(st.ssCfg), SlatCfg: float64(st.slatCfg),
			Tris: int(st.tris), TexSize: int(st.texSize), SubsampleRes: int(st.subsampleRes),
			Smooth: st.smooth, SmoothIters: int(st.smoothIters), MeshCleanup: st.cleanupMesh, Offload: float64(1 - st.offload),
		})
		if err != nil {
			st.status = "generate: " + err.Error()
			beep("error")
			st.flash = 60
			log.Printf("[gen] FAILED after %.0fs: %v", time.Since(genStart).Seconds(), err)
		} else {
			st.glb = strOf(j["glb"]); st.zip = strOf(j["zip"]); st.blend = strOf(j["blend"])
			st.status = fmt.Sprintf("done in %vs · %v faces", j["seconds"], j["faces"])
			beep("done")
			st.flash = 60
			st.needReload = true
			log.Printf("[gen] done in %vs · %v faces", j["seconds"], j["faces"])
		}
		st.busy = false
	}()
	go func() {
		// Poll the worker's phase so the viewport can show live progress
		// ("processing gaussians…", "preparing meshes…", "baking texture…").
		for st.busy {
			st.phase = fetchPhase()
			time.Sleep(500 * time.Millisecond)
		}
		st.phase = ""
	}()
}

func strOf(v any) string { s, _ := v.(string); return s }
func itoa(n int) string   { return fmt.Sprintf("%d", n) }

func clonePts(pts []pt) []pt { return append([]pt(nil), pts...) }

func (st *appState) pushHistory() {
	st.undo = append(st.undo, clonePts(st.mask))
	if len(st.undo) > 50 {
		st.undo = st.undo[1:]
	}
	st.redo = nil
}

func (st *appState) undoMask() {
	if len(st.undo) == 0 {
		return
	}
	st.redo = append(st.redo, clonePts(st.mask))
	st.mask = st.undo[len(st.undo)-1]
	st.undo = st.undo[:len(st.undo)-1]
}

func (st *appState) redoMask() {
	if len(st.redo) == 0 {
		return
	}
	st.undo = append(st.undo, clonePts(st.mask))
	st.mask = st.redo[len(st.redo)-1]
	st.redo = st.redo[:len(st.redo)-1]
}

func ctrlHeld() bool {
	if gwin == nil {
		return false
	}
	return gwin.GetKey(glfw.KeyLeftControl) == glfw.Press ||
		gwin.GetKey(glfw.KeyRightControl) == glfw.Press
}

func fetchModels() ([]string, error) {
	body, err := callOrch("GET", orchURL()+"/api/models", "", nil, 20*time.Second)
	if err != nil {
		return nil, err
	}
	var models []string
	if err := json.Unmarshal(body, &models); err != nil {
		return nil, err
	}
	return models, nil
}

func openBlend(path string) {
	bin := os.Getenv("BLENDER")
	if bin == "" {
		bin = "blender"
	}
	exec.Command(bin, path).Start()
}

func minInt(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

func cloneRGBA(img *image.RGBA) *image.RGBA {
	out := image.NewRGBA(img.Rect)
	copy(out.Pix, img.Pix)
	return out
}

func imageToRGBA(im image.Image) *image.RGBA {
	b := im.Bounds()
	rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			rgba.Set(x, y, im.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	return rgba
}

func uploadRGBATexture(img *image.RGBA, reuse uint32) int32 {
	var tex uint32
	w, h := img.Rect.Dx(), img.Rect.Dy()
	// Upload the pixel rows as-is: ImGui UV(0,0)=top-left samples the first
	// uploaded row (which is the image top), so the image displays upright.
	pix := img.Pix
	tex = reuse
	if tex == 0 {
		gl.GenTextures(1, &tex)
	}
	gl.BindTexture(gl.TEXTURE_2D, tex)
	gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, int32(w), int32(h), 0, gl.RGBA, gl.UNSIGNED_BYTE, gl.Ptr(pix))
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	return int32(tex)
}
