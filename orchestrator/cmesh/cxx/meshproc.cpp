// meshproc.cpp — the ENTIRE mesh pipeline as one C/C++ subprocess: repair,
// degenerate-drop, quadric decimation, component cleanup, bilateral smooth.
// Runs as its own process because in-process cgo C calls corrupt the heap on
// this toolchain (proven empirically); a pure C/C++ process is reliable.
//
// Usage: meshproc <in.bin> <out.bin> [--repair] [--dedegen] [--decimate N]
//                 [--cleanup MINFRAC] [--smooth ITERS]
// Binary format: magic "CMESH", int32 nverts, int32 nfaces,
// nverts*3 float32 verts, nfaces*3 uint32 faces.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../vendor/wrapper.h"      // decimator (fast_simplification rewrite)
extern "C" {
#include "../cmesh.h"               // C kernels: nonmanifold, degenerate, components, smooth
}

static void read_all(FILE *f, void *dst, size_t n) {
    if (n && fread(dst, 1, n, f) != n) { fprintf(stderr, "read error\n"); exit(1); }
}

static void drop_faces(float *verts, uint32_t *faces, int *nverts, int *nfaces,
                       const uint8_t *drop) {
    int nf = *nfaces, w = 0;
    for (int i = 0; i < nf; i++)
        if (!drop[i]) {
            faces[w*3] = faces[i*3]; faces[w*3+1] = faces[i*3+1]; faces[w*3+2] = faces[i*3+2];
            w++;
        }
    *nfaces = w;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: meshproc <in> <out> [flags]\n"); return 2; }
    FILE *in = fopen(argv[1], "rb");
    if (!in) { perror("open in"); return 1; }
    char magic[6] = {0};
    read_all(in, magic, 5);
    if (magic[0] != 'C') { fprintf(stderr, "bad magic\n"); return 1; }
    int nverts = 0, nfaces = 0;
    read_all(in, &nverts, 4);
    read_all(in, &nfaces, 4);
    std::vector<float> verts((size_t)nverts * 3);
    read_all(in, verts.data(), (size_t)nverts * 3 * 4);
    std::vector<uint32_t> faces((size_t)nfaces * 3);
    read_all(in, faces.data(), (size_t)nfaces * 3 * 4);
    fclose(in);

    bool repair = false, dedegen = false, cleanup = false, smooth = false;
    int decimate = 0, iters = 0;
    double minfrac = 0.01;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--repair") repair = true;
        else if (a == "--dedegen") dedegen = true;
        else if (a == "--cleanup") { cleanup = true; if (i+1 < argc) minfrac = atof(argv[++i]); }
        else if (a == "--decimate") { if (i+1 < argc) decimate = atoi(argv[++i]); }
        else if (a == "--smooth") { smooth = true; if (i+1 < argc) iters = atoi(argv[++i]); }
    }

    std::vector<uint8_t> drop;
    if (repair) {
        drop.assign((size_t)nfaces, 0);
        cm_nonmanifold(faces.data(), nfaces, nverts, drop.data());
        drop_faces(verts.data(), faces.data(), &nverts, &nfaces, drop.data());
        fprintf(stderr, "repair -> %d faces\n", nfaces);
    }
    if (dedegen) {
        drop.assign((size_t)nfaces, 0);
        cm_degenerate(verts.data(), faces.data(), nfaces, drop.data());
        drop_faces(verts.data(), faces.data(), &nverts, &nfaces, drop.data());
        fprintf(stderr, "dedegen -> %d faces\n", nfaces);
    }
    if (decimate > 0 && nfaces > decimate) {
        std::vector<double> pts((size_t)nverts * 3);
        for (size_t i = 0; i < pts.size(); i++) pts[i] = verts[i];
        std::vector<int> fc((size_t)nfaces * 3);
        for (size_t i = 0; i < fc.size(); i++) fc[i] = (int)faces[i];
        Simplify::load_arrays_int32(nverts, nfaces, pts.data(), fc.data());
        Simplify::simplify_mesh(decimate, 7.0, false);
        nverts = Simplify::n_points();
        /* unpadded readback (stride 3) + its live-face count: get_faces_int32
           emits VTK 4-int cells [3,v0,v1,v2] and would corrupt 3 of every 4
           faces here, and overflow fc when the target exceeds 75% of input. */
        Simplify::get_points(pts.data());
        nfaces = Simplify::get_faces_int32_no_padding(fc.data());
        verts.resize((size_t)nverts * 3);
        faces.resize((size_t)nfaces * 3);
        for (int i = 0; i < nverts * 3; i++) verts[i] = (float)pts[i];
        for (int i = 0; i < nfaces * 3; i++) faces[i] = (uint32_t)fc[i];
        fprintf(stderr, "decimate -> %d faces\n", nfaces);
    }
    if (cleanup) {
        int nv2 = nverts;
        int nf2 = cm_keep_components(verts.data(), faces.data(), &nv2, nfaces,
                                     (float)minfrac);
        nverts = nv2; nfaces = nf2;
        fprintf(stderr, "cleanup -> %d verts %d faces\n", nverts, nfaces);
    }
    if (decimate > 0) {
        /* the decimator can leave degenerate/non-manifold structure */
        drop.assign((size_t)nfaces, 0);
        cm_nonmanifold(faces.data(), nfaces, nverts, drop.data());
        std::vector<uint8_t> dg((size_t)nfaces, 0);
        cm_degenerate(verts.data(), faces.data(), nfaces, dg.data());
        for (size_t i = 0; i < dg.size(); i++) if (dg[i]) drop[i] = 1;
        drop_faces(verts.data(), faces.data(), &nverts, &nfaces, drop.data());
        int nv2 = nverts;
        nfaces = cm_keep_components(verts.data(), faces.data(), &nv2, nfaces, 0.01f);
        nverts = nv2;
        fprintf(stderr, "post-decimate cleanup -> %d verts %d faces\n", nverts, nfaces);
    }
    if (smooth && iters > 0) {
        /* AFTER the post-decimate cleanup: smoothing against garbage
           (non-manifold/degenerate) topology contaminates the surviving
           vertex positions, and those positions then get dropped anyway. */
        cm_smooth_bilateral(verts.data(), faces.data(), nverts, nfaces,
                            iters, 0.4f, 0.5f);
    }
    /* consistent winding: xatlas charting (the GLB bake) requires it */
    {
        int nflip = cm_orient(faces.data(), nfaces);
        if (nflip) fprintf(stderr, "orient: flipped %d faces\n", nflip);
    }

    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror("open out"); return 1; }
    fwrite("CMESH", 1, 5, out);
    fwrite(&nverts, 4, 1, out);
    fwrite(&nfaces, 4, 1, out);
    fwrite(verts.data(), 4, (size_t)nverts * 3, out);
    fwrite(faces.data(), 4, (size_t)nfaces * 3, out);
    fclose(out);
    return 0;
}
