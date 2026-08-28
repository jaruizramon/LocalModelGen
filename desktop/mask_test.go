package main

import (
	"image"
	"image/color"
	"testing"
)

func TestParsePoly(t *testing.T) {
	pts := parsePoly(`[[0.1,0.2],[0.9,0.2],[0.5,0.9]]`)
	if len(pts) != 3 {
		t.Fatalf("want 3 pts, got %d", len(pts))
	}
	if pts[0].X != 0.1 || pts[0].Y != 0.2 {
		t.Fatalf("bad first pt: %+v", pts[0])
	}
	if parsePoly(`bad`) != nil {
		t.Fatal("expected nil for bad json")
	}
}

func TestPointInPoly(t *testing.T) {
	// unit square
	pts := []pt{{0, 0}, {1, 0}, {1, 1}, {0, 1}}
	if !pointInPoly(0.5, 0.5, pts) {
		t.Fatal("center should be inside")
	}
	if pointInPoly(1.5, 0.5, pts) {
		t.Fatal("outside should be outside")
	}
}

func TestApplyPolygonMask(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 10, 10))
	for i := range img.Pix {
		img.Pix[i] = 255
	}
	// mask the top-left quadrant only: [[0,0],[0.5,0],[0.5,0.5],[0,0.5]]
	left := applyPolygonMask(img, []pt{{0, 0}, {0.5, 0}, {0.5, 0.5}, {0, 0.5}})
	if c := left.RGBAAt(1, 1); c.A == 0 {
		t.Fatal("inside point should have alpha > 0")
	}
	if c := left.RGBAAt(8, 8); c.A != 0 {
		t.Fatalf("outside point should have alpha 0, got %v", c.A)
	}
}

func TestDrawPolygonOverlay(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	out := drawPolygonOverlay(img, []pt{{0.1, 0.1}, {0.9, 0.1}, {0.5, 0.9}}, true, 2.0, true)
	if out.Rect.Dx() != 100 {
		t.Fatal("overlay size mismatch")
	}
	// a pixel on the outline edge should have alpha
	found := false
	for y := 0; y < 100 && !found; y++ {
		for x := 0; x < 100; x++ {
			if out.RGBAAt(x, y).A == 255 && out.RGBAAt(x, y).R != 0 {
				found = true
				break
			}
		}
	}
	if !found {
		t.Fatal("expected an opaque non-black pixel in the overlay")
	}
}

func TestClamp01(t *testing.T) {
	if clamp01(-0.1) != 0 || clamp01(1.2) != 1 || clamp01(0.5) != 0.5 {
		t.Fatal("clamp01 broken")
	}
}

func TestUVMapping(t *testing.T) {
	// rectMin=(100,200) rectMax=(300,400); mouse=(200,300) -> (0.5, 0.5)
	min := struct{ X, Y float32 }{100, 200}
	max := struct{ X, Y float32 }{300, 400}
	mx, my := float32(200), float32(300)
	ux := (mx - min.X) / (max.X - min.X)
	uy := (my - min.Y) / (max.Y - min.Y)
	if ux != 0.5 || uy != 0.5 {
		t.Fatalf("uv map: %v %v", ux, uy)
	}
}

var _ = color.RGBA{}
