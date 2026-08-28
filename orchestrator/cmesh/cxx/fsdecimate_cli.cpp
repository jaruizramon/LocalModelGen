// fsdecimate_cli.cpp — standalone quadric decimator using the vendored
// fast_simplification (Forstmann/Rorden rewrite). Runs as its own process:
// the rewrite has refs/flipped logic that is memory-layout sensitive and
// corrupts the heap when called in-process via cgo, but is rock-solid as a
// standalone executable (this is the same algorithm the pipeline used via
// Python, so output quality is a drop-in match).
//
// Usage: fsdecimate <in.bin> <out.bin> <target_faces>
// Binary format: magic "CMESH", int32 nverts, int32 nfaces,
// nverts*3 float32 verts, nfaces*3 uint32 faces.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../vendor/wrapper.h"

static void read_all(FILE *f, void *dst, size_t n) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "read error\n"); exit(1); }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: fsdecimate <in.bin> <out.bin> <target_faces>\n");
        return 2;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) { perror("open in"); return 1; }
    char magic[6] = {0};
    read_all(in, magic, 5);
    if (magic[0] != 'C') { fprintf(stderr, "bad magic\n"); return 1; }
    int nverts = 0, nfaces = 0;
    read_all(in, &nverts, 4);
    read_all(in, &nfaces, 4);

    std::vector<double> pts((size_t)nverts * 3);
    std::vector<float> vf((size_t)nverts * 3);
    read_all(in, vf.data(), (size_t)nverts * 3 * 4);
    for (size_t i = 0; i < pts.size(); i++) pts[i] = vf[i];
    std::vector<int> faces((size_t)nfaces * 3);
    std::vector<uint32_t> ff((size_t)nfaces * 3);
    read_all(in, ff.data(), (size_t)nfaces * 3 * 4);
    for (size_t i = 0; i < faces.size(); i++) faces[i] = (int)ff[i];
    fclose(in);

    int target = atoi(argv[3]);
    Simplify::load_arrays_int32(nverts, nfaces, pts.data(), faces.data());
    Simplify::simplify_mesh(target, 7.0, false);
    int np = Simplify::n_points();
    Simplify::get_points(pts.data());
    /* get_faces_int32 emits VTK-PADDED 4-int cells [3,v0,v1,v2]; this driver
       consumes stride 3. Use the unpadded sibling and ITS live-face count --
       n_triangles() is only correct after compact_mesh and says nothing about
       the stride. This is what the pip wheel does (fast_simplification/
       simplify.py: return_faces_int32_no_padding().reshape(-1,3)). */
    int nt = Simplify::get_faces_int32_no_padding(faces.data());

    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror("open out"); return 1; }
    fwrite("CMESH", 1, 5, out);
    fwrite(&np, 4, 1, out);
    fwrite(&nt, 4, 1, out);
    for (int i = 0; i < np; i++) {
        float v[3] = {(float)pts[i * 3], (float)pts[i * 3 + 1], (float)pts[i * 3 + 2]};
        fwrite(v, 4, 3, out);
    }
    for (int i = 0; i < nt; i++) {
        uint32_t f[3] = {(uint32_t)faces[i * 3], (uint32_t)faces[i * 3 + 1],
                         (uint32_t)faces[i * 3 + 2]};
        fwrite(f, 4, 3, out);
    }
    fclose(out);
    return 0;
}
