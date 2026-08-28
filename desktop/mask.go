package main

import (
	"encoding/json"
	"image"
	"image/color"
)

type pt struct {
	X, Y float64 // normalized 0..1
}

func parsePoly(s string) []pt {
	var raw [][2]float64
	if err := json.Unmarshal([]byte(s), &raw); err != nil {
		return nil
	}
	pts := make([]pt, 0, len(raw))
	for _, p := range raw {
		pts = append(pts, pt{X: p[0], Y: p[1]})
	}
	return pts
}

// pointInPoly is an even-odd test (cast ray to +x).
func pointInPoly(px, py float64, pts []pt) bool {
	in := false
	n := len(pts)
	for i, j := 0, n-1; i < n; j, i = i, i+1 {
		xi, yi := pts[i].X, pts[i].Y
		xj, yj := pts[j].X, pts[j].Y
		if (yi > py) != (yj > py) {
			x := (xj-xi)*(py-yi)/(yj-yi) + xi
			if px < x {
				in = !in
			}
		}
	}
	return in
}

// applyPolygonMask zeroes the alpha channel outside the polygon.
func applyPolygonMask(img *image.RGBA, pts []pt) *image.RGBA {
	if len(pts) < 3 {
		return img
	}
	w, h := img.Rect.Dx(), img.Rect.Dy()
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !pointInPoly(float64(x)/float64(w), float64(y)/float64(h), pts) {
				c := img.RGBAAt(x, y)
			img.SetRGBA(x, y, color.RGBA{R: c.R, G: c.G, B: c.B, A: 0})
			}
		}
	}
	return img
}

// drawPolygonOverlay draws the image then the polygon (outline + vertex dots,
// translucent fill for the closed state) into an RGBA buffer for display.
func drawPolygonOverlay(img *image.RGBA, pts []pt, closed bool, scale float64, fill bool) *image.RGBA {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	copy(out.Pix, img.Pix)
	if len(pts) < 1 {
		return out
	}
	px := func(p pt) (int, int) { return int(p.X * float64(w)), int(p.Y * float64(h)) }
	bright := color.RGBA{0x35, 0xe0, 0x8a, 255} // bright green
	dark := color.RGBA{20, 30, 24, 255}
	white := color.RGBA{255, 255, 255, 255}
	lineR := int(2.4*scale) + 1
	dotR := int(7*scale) + 2
	if fill && closed && len(pts) >= 3 {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				if pointInPoly(float64(x)/float64(w), float64(y)/float64(h), pts) {
					blend(out, x, y, color.RGBA{0x21, 0x9b, 0x8a, 255}, 55)
				}
			}
		}
	}
	// outline: dark halo then bright core
	for i := 0; i < len(pts); i++ {
		a := pts[i]
		b := pts[(i+1)%len(pts)]
		if i == len(pts)-1 && !closed {
			break
		}
		ax, ay := px(a)
		bx, by := px(b)
		drawThickLine(out, ax, ay, bx, by, dark, lineR+1)
		drawThickLine(out, ax, ay, bx, by, bright, lineR)
	}
	// vertex dots: dark ring + white center
	for _, p := range pts {
		x, y := px(p)
		drawCircle(out, x, y, dotR+2, dark)
		drawCircle(out, x, y, dotR, white)
	}
	return out
}

func drawThickLine(out *image.RGBA, x0, y0, x1, y1 int, c color.RGBA, r int) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		drawCircle(out, x0, y0, r, c)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func blend(out *image.RGBA, x, y int, c color.RGBA, amt int) {
	i := y*out.Stride + x*4
	r := int(out.Pix[i]) * (255 - amt) / 255 + int(c.R) * amt / 255
	g := int(out.Pix[i+1]) * (255 - amt) / 255 + int(c.G) * amt / 255
	b := int(out.Pix[i+2]) * (255 - amt) / 255 + int(c.B) * amt / 255
	a := int(out.Pix[i+3])
	if amt > 0 {
		a = 255
	}
	out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = uint8(r), uint8(g), uint8(b), uint8(a)
}

func drawLine(out *image.RGBA, x0, y0, x1, y1 int, c color.RGBA) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		setPx(out, x0, y0, c)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func drawCircle(out *image.RGBA, cx, cy, r int, c color.RGBA) {
	for y := -r; y <= r; y++ {
		for x := -r; x <= r; x++ {
			if x*x+y*y <= r*r {
				setPx(out, cx+x, cy+y, c)
			}
		}
	}
}

func setPx(out *image.RGBA, x, y int, c color.RGBA) {
	if x < 0 || y < 0 || x >= out.Rect.Dx() || y >= out.Rect.Dy() {
		return
	}
	i := y*out.Stride + x*4
	out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = c.R, c.G, c.B, 255
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
