package main

import (
	"math"

	"github.com/go-gl/gl/v3.3-core/gl"
)

// OrbitCamera is a yaw/pitch/distance orbit rig looking at a target.
type OrbitCamera struct {
	Yaw, Pitch, Distance float64
	Target              [3]float64
	Width, Height       int
	FovY                float64 // vertical fov, radians
}

func (c *OrbitCamera) normalize() {
	if c.Pitch > 1.55 {
		c.Pitch = 1.55
	}
	if c.Pitch < -1.55 {
		c.Pitch = -1.55
	}
	if c.Distance < 0.2 {
		c.Distance = 0.2
	}
	if c.Width < 1 {
		c.Width = 1
	}
	if c.Height < 1 {
		c.Height = 1
	}
}

// view returns a column-major 4x4 view matrix (world -> camera, camera looks
// down -Z, up = +Y).
func (c *OrbitCamera) view() [16]float32 {
	c.normalize()
	cy, sy := math.Cos(c.Yaw), math.Sin(c.Yaw)
	cp, sp := math.Cos(c.Pitch), math.Sin(c.Pitch)
	eye := [3]float64{
		c.Target[0] + c.Distance*cy*cp,
		c.Target[1] + c.Distance*sp,
		c.Target[2] + c.Distance*sy*cp,
	}
	f := [3]float64{c.Target[0] - eye[0], c.Target[1] - eye[1], c.Target[2] - eye[2]}
	fl := math.Sqrt(f[0]*f[0] + f[1]*f[1] + f[2]*f[2])
	f[0], f[1], f[2] = f[0]/fl, f[1]/fl, f[2]/fl
	// right = norm(cross(f, up)), up=(0,1,0)
	s := [3]float64{-f[2], 0, f[0]} // cross(f,(0,1,0))
	sl := math.Sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2])
	s[0], s[1], s[2] = s[0]/sl, s[1]/sl, s[2]/sl
	u := [3]float64{s[1]*f[2] - s[2]*f[1], s[2]*f[0] - s[0]*f[2], s[0]*f[1] - s[1]*f[0]}
	// column-major view matrix
	return [16]float32{
		float32(s[0]), float32(u[0]), float32(-f[0]), 0,
		float32(s[1]), float32(u[1]), float32(-f[1]), 0,
		float32(s[2]), float32(u[2]), float32(-f[2]), 0,
		float32(-(s[0]*eye[0] + s[1]*eye[1] + s[2]*eye[2])),
		float32(-(u[0]*eye[0] + u[1]*eye[1] + u[2]*eye[2])),
		float32(f[0]*eye[0] + f[1]*eye[1] + f[2]*eye[2]), 1,
	}
}

// proj returns a column-major 4x4 perspective matrix.
func (c *OrbitCamera) proj() [16]float32 {
	c.normalize()
	aspect := float64(c.Width) / float64(c.Height)
	f := 1.0 / math.Tan(c.FovY*0.5)
	near, far := 0.01, 100.0
	return [16]float32{
		float32(f / aspect), 0, 0, 0,
		0, float32(f), 0, 0,
		0, 0, float32((far + near) / (near - far)), -1,
		0, 0, float32(2 * far * near / (near - far)), 0,
	}
}

// pointSizeFor computes the gl_PointSize for a splat whose covariance 2D
// determinant / radius we compute in the shader; on CPU we only need view z for
// sorting, handled by depth().
func (c *OrbitCamera) depthZ(pos [3]float32) float32 {
	v := c.view()
	return v[2]*pos[0] + v[6]*pos[1] + v[10]*pos[2] + v[14]
}

var _ = gl.POINTS
