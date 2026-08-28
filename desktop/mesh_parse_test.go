package main

import (
	"os"
	"path/filepath"
	"testing"
)

// parseObjMesh must handle Open3D-style OBJs: "v x y z r g b" vertex lines,
// interleaved "vn" normals, and "f v//vn" slash-indexed faces. Previously:
// (1) "vn" lines were parsed as vertices (30K bogus unit-sphere verts that
// wrecked centerAndScale) and (2) "f 6//6" faces collapsed to (0,0,0) because
// %d stopped at the slash -- the whole mesh rendered as one degenerate
// triangle (invisible).
func TestParseObjMeshOpen3D(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "m.obj")
	obj := `v 0 0 0 0 0 0
vn 0 0 1
v 1 0 0 0 0 0
vn 0 0 1
v 0 1 0 0 0 0
vn 0 0 1
f 1//1 2//2 3//3
`
	if err := os.WriteFile(p, []byte(obj), 0644); err != nil {
		t.Fatal(err)
	}
	md, err := parseObjMesh(p)
	if err != nil {
		t.Fatal(err)
	}
	if md.NV != 3 {
		t.Fatalf("NV = %d, want 3 (vn lines must not count as vertices)", md.NV)
	}
	if len(md.Idx) != 3 {
		t.Fatalf("len(Idx) = %d, want 3", len(md.Idx))
	}
	if md.Idx[0] != 0 || md.Idx[1] != 1 || md.Idx[2] != 2 {
		t.Fatalf("Idx = %v, want [0 1 2]", md.Idx)
	}
}
