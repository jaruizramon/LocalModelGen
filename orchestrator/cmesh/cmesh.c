/* cmesh.c — C implementations. Flat arrays, caller-owned memory. */
#include "cmesh.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------- union-find ---------------- */
static int uf_find(int *p, int x) {
    int r = x;
    while (p[r] != r) r = p[r];
    while (p[x] != r) { int n = p[x]; p[x] = r; x = n; }
    return r;
}
static void uf_union(int *p, int a, int b) {
    int ra = uf_find(p, a), rb = uf_find(p, b);
    if (ra != rb) p[ra] = rb;
}
/* ---------------- hash: int64 key -> int value ---------------- */
typedef struct { int64_t key; int val; int used; } hslot;
typedef struct { hslot *t; size_t cap, n; } hmap;
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27;
    x *= 0x94d049bb133111ebULL; x ^= x >> 31; return x;
}
static int hmap_get(hmap *m, int64_t key) {
    if (!m->t) return -1;
    size_t mask = m->cap - 1, i = mix64((uint64_t)key) & mask;
    for (;;) {
        if (!m->t[i].used) return -1;
        if (m->t[i].key == key) return m->t[i].val;
        i = (i + 1) & mask;
    }
}
static void hmap_put(hmap *m, int64_t key, int val) {
    if (m->n * 4 >= m->cap * 3) { /* grow */
        size_t ncap = m->cap ? m->cap * 2 : 4096;
        hslot *nt = calloc(ncap, sizeof(hslot));
        size_t mask = ncap - 1;
        for (size_t i = 0; i < m->cap; i++)
            if (m->t[i].used) {
                size_t j = mix64((uint64_t)m->t[i].key) & mask;
                while (nt[j].used) j = (j + 1) & mask;
                nt[j] = m->t[i];
            }
        free(m->t); m->t = nt; m->cap = ncap;
    }
    size_t mask = m->cap - 1, i = mix64((uint64_t)key) & mask;
    /* update in place if the key exists (open addressing) */
    while (m->t[i].used && m->t[i].key != key) i = (i + 1) & mask;
    if (!m->t[i].used) m->n++;
    m->t[i].key = key; m->t[i].val = val; m->t[i].used = 1;
}
static void hmap_free(hmap *m) { free(m->t); m->t = NULL; m->cap = m->n = 0; }

/* ---------------- connected components ---------------- */
int cm_components(const uint32_t *faces, int nfaces, int nverts, int *comp_out) {
    int *p = malloc((size_t)nverts * sizeof(int));
    for (int i = 0; i < nverts; i++) p[i] = i;
    for (int i = 0; i < nfaces; i++) {
        uf_union(p, (int)faces[i * 3 + 0], (int)faces[i * 3 + 1]);
        uf_union(p, (int)faces[i * 3 + 0], (int)faces[i * 3 + 2]);
    }
    /* relabel roots 0..ncomp-1 */
    int *label = malloc((size_t)nverts * sizeof(int));
    for (int i = 0; i < nverts; i++) label[i] = -1;
    int ncomp = 0;
    for (int i = 0; i < nverts; i++) {
        int r = uf_find(p, i);
        if (label[r] < 0) label[r] = ncomp++;
        comp_out[i] = label[r];
    }
    free(p); free(label);
    return ncomp;
}

/* ---------------- non-manifold ---------------- */
int cm_nonmanifold(const uint32_t *faces, int nfaces, int nverts, uint8_t *bad_out) {
    (void)nverts;
    memset(bad_out, 0, (size_t)nfaces);
    hmap counts = {0};
    int nbad = 0;
    for (int i = 0; i < nfaces; i++) {
        uint32_t a = faces[i * 3 + 0], b = faces[i * 3 + 1], c = faces[i * 3 + 2];
        uint32_t e[3][2] = { {a, b}, {b, c}, {c, a} };
        for (int j = 0; j < 3; j++) {
            uint32_t lo = e[j][0] < e[j][1] ? e[j][0] : e[j][1];
            uint32_t hi = e[j][0] < e[j][1] ? e[j][1] : e[j][0];
            int64_t key = ((int64_t)lo << 32) | hi;
            int c = hmap_get(&counts, key);
            hmap_put(&counts, key, c < 0 ? 1 : c + 1);
        }
    }
    for (int i = 0; i < nfaces; i++) {
        uint32_t a = faces[i * 3 + 0], b = faces[i * 3 + 1], c = faces[i * 3 + 2];
        uint32_t e[3][2] = { {a, b}, {b, c}, {c, a} };
        for (int j = 0; j < 3; j++) {
            uint32_t lo = e[j][0] < e[j][1] ? e[j][0] : e[j][1];
            uint32_t hi = e[j][0] < e[j][1] ? e[j][1] : e[j][0];
            int64_t key = ((int64_t)lo << 32) | hi;
            if (hmap_get(&counts, key) > 2) { if (!bad_out[i]) nbad++; bad_out[i] = 1; break; }
        }
    }
    hmap_free(&counts);
    return nbad;
}

/* ---------------- bilateral smoothing ---------------- */
/* edge key -> chain of faces sharing it */
typedef struct { int64_t key; int head; } ehead;
void cm_smooth_bilateral(float *verts, const uint32_t *faces, int nverts,
                         int nfaces, int iters, float sigma_n, float sigma_c) {
    if (nfaces <= 0 || iters <= 0) return;
    float *fn = malloc((size_t)nfaces * 3 * sizeof(float));
    float *cents = malloc((size_t)nfaces * 3 * sizeof(float));
    float *nfn = malloc((size_t)nfaces * 3 * sizeof(float));
    /* adjacency: for each face, list of neighbor faces (edge-shared) */
    int *neighbors = malloc((size_t)nfaces * 12 * sizeof(int)); /* up to 12 */
    int *ncount = calloc((size_t)nfaces, sizeof(int));
    hmap first = {0}; /* edge key -> face index */
    int *next_face = malloc((size_t)nfaces * 3 * sizeof(int));
    for (int i = 0; i < nfaces * 3; i++) next_face[i] = -1;

    for (int i = 0; i < nfaces; i++) {
        uint32_t a = faces[i * 3 + 0], b = faces[i * 3 + 1], c = faces[i * 3 + 2];
        const float *va = verts + a * 3, *vb = verts + b * 3, *vc = verts + c * 3;
        float e0[3] = { vb[0]-va[0], vb[1]-va[1], vb[2]-va[2] };
        float e1[3] = { vc[0]-va[0], vc[1]-va[1], vc[2]-va[2] };
        float nx = e0[1]*e1[2]-e0[2]*e1[1], ny = e0[2]*e1[0]-e0[0]*e1[2], nz = e0[0]*e1[1]-e0[1]*e1[0];
        float l = sqrtf(nx*nx+ny*ny+nz*nz) + 1e-12f;
        if (!isfinite(nx) || !isfinite(ny) || !isfinite(nz) || l > 1e30f) {
            fn[i*3+0] = 0; fn[i*3+1] = 0; fn[i*3+2] = 1;  /* degenerate face: neutral normal */
        } else {
            fn[i*3+0] = nx/l; fn[i*3+1] = ny/l; fn[i*3+2] = nz/l;
        }
        cents[i*3+0] = (va[0]+vb[0]+vc[0])/3.0f;
        cents[i*3+1] = (va[1]+vb[1]+vc[1])/3.0f;
        cents[i*3+2] = (va[2]+vb[2]+vc[2])/3.0f;
        uint32_t e[3][2] = { {a, b}, {b, c}, {c, a} };
        for (int j = 0; j < 3; j++) {
            uint32_t lo = e[j][0] < e[j][1] ? e[j][0] : e[j][1];
            uint32_t hi = e[j][0] < e[j][1] ? e[j][1] : e[j][0];
            int64_t key = ((int64_t)lo << 32) | hi;
            int h = hmap_get(&first, key);
            if (h < 0) hmap_put(&first, key, i);
            else {
                /* add i to h's neighbor list and h to i's */
                int *nh = neighbors + h * 12, *ni = neighbors + i * 12;
                if (ncount[h] < 12 && ncount[i] < 12) {
                    int dup = 0;
                    for (int k = 0; k < ncount[h]; k++) if (nh[k] == i) { dup = 1; break; }
                    if (!dup) { nh[ncount[h]++] = i; ni[ncount[i]++] = h; }
                }
            }
        }
    }
    hmap_free(&first);
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < nfaces; i++) {
            const float *ni_ = fn + i * 3;
            float sx = 0, sy = 0, sz = 0, sw = 0;
            int nc = ncount[i];
            int *nb = neighbors + i * 12;
            for (int k = 0; k < nc; k++) {
                int j = nb[k];
                const float *nj = fn + j * 3;
                float dot = ni_[0]*nj[0] + ni_[1]*nj[1] + ni_[2]*nj[2];
                if (dot > 1.0f) { dot = 1.0f; }
                if (dot < -1.0f) { dot = -1.0f; }
                float ddx = cents[i*3+0]-cents[j*3+0], ddy = cents[i*3+1]-cents[j*3+1], ddz = cents[i*3+2]-cents[j*3+2];
                float w = expf(-(1.0f-dot)/(sigma_n*sigma_n)) * expf(-(ddx*ddx+ddy*ddy+ddz*ddz)/(sigma_c*sigma_c));
                sx += nj[0]*w; sy += nj[1]*w; sz += nj[2]*w; sw += w;
            }
            if (sw > 1e-12f) {
                float l = sqrtf(sx*sx+sy*sy+sz*sz) + 1e-12f;
                nfn[i*3+0] = sx/l; nfn[i*3+1] = sy/l; nfn[i*3+2] = sz/l;
            } else {
                nfn[i*3+0] = ni_[0]; nfn[i*3+1] = ni_[1]; nfn[i*3+2] = ni_[2];
            }
        }
        memcpy(fn, nfn, (size_t)nfaces * 3 * sizeof(float));
    }
    /* vertex update: average projections onto incident smoothed planes */
    float *acc = calloc((size_t)nverts * 3, sizeof(float));
    int *cnt = calloc((size_t)nverts, sizeof(int));
    for (int i = 0; i < nfaces; i++) {
        const float *n = fn + i * 3, *c = cents + i * 3;
        for (int j = 0; j < 3; j++) {
            uint32_t v = faces[i * 3 + j];
            const float *p = verts + v * 3;
            float t = (p[0]-c[0])*n[0] + (p[1]-c[1])*n[1] + (p[2]-c[2])*n[2];
            acc[v*3+0] += c[0] + n[0]*t;
            acc[v*3+1] += c[1] + n[1]*t;
            acc[v*3+2] += c[2] + n[2]*t;
            cnt[v]++;
        }
    }
    for (int v = 0; v < nverts; v++)
        if (cnt[v]) {
            float invc = 1.0f / cnt[v];
            float nx = acc[v*3+0]*invc, ny = acc[v*3+1]*invc, nz = acc[v*3+2]*invc;
            if (isfinite(nx) && isfinite(ny) && isfinite(nz))
                { verts[v*3+0]=nx; verts[v*3+1]=ny; verts[v*3+2]=nz; }
        }
    free(fn); free(cents); free(nfn); free(neighbors); free(ncount);
    free(next_face); free(acc); free(cnt);
}

/* ---------------- keep components >= min_frac ---------------- */
int cm_keep_components(float *verts, uint32_t *faces, int *nverts, int nfaces,
                       float min_frac) {
    if (nfaces <= 0) return 0;
    int nv = *nverts;
    int *comp = malloc((size_t)nv * sizeof(int));
    int ncomp = cm_components(faces, nfaces, nv, comp);
    int *cfaces = calloc((size_t)ncomp, sizeof(int));
    for (int i = 0; i < nfaces; i++)
        cfaces[comp[faces[i*3]]]++;
    int minf = (int)(nfaces * min_frac);
    if (minf < 1) minf = 1;
    int keep = 0, largest = 0;
    for (int i = 0; i < ncomp; i++) {
        if (cfaces[i] >= minf) keep++;
        if (cfaces[i] > cfaces[largest]) largest = i;
    }
    if (keep == 0) {
        /* Shattered mesh: every component is below min_frac. Fall back to the
           single largest component instead of annihilating the mesh to
           0 verts / 0 faces (which would be written out as a valid-looking
           empty CMESH with exit code 0). Raising its count to nfaces (>= any
           minf) makes the threshold tests below keep exactly that component. */
        keep = 1;
        cfaces[largest] = nfaces;
        for (int i = 0; i < ncomp; i++)
            if (i != largest) cfaces[i] = 0;
    }
    if (keep == ncomp) { free(comp); free(cfaces); return nfaces; }
    /* mark kept vertices */
    int *vkeep = calloc((size_t)nv, sizeof(int));
    for (int i = 0; i < nfaces; i++)
        if (cfaces[comp[faces[i*3]]] >= minf)
            for (int j = 0; j < 3; j++) vkeep[faces[i*3+j]] = 1;
    int *vmap = malloc((size_t)nv * sizeof(int));
    int nv2 = 0;
    for (int i = 0; i < nv; i++) vmap[i] = vkeep[i] ? nv2++ : -1;
    /* compact verts */
    for (int i = 0; i < nv; i++)
        if (vmap[i] >= 0 && vmap[i] != i) {
            verts[vmap[i]*3] = verts[i*3];
            verts[vmap[i]*3+1] = verts[i*3+1];
            verts[vmap[i]*3+2] = verts[i*3+2];
        }
    /* compact faces */
    int nf2 = 0;
    for (int i = 0; i < nfaces; i++)
        if (cfaces[comp[faces[i*3]]] >= minf) {
            faces[nf2*3] = (uint32_t)vmap[faces[i*3]];
            faces[nf2*3+1] = (uint32_t)vmap[faces[i*3+1]];
            faces[nf2*3+2] = (uint32_t)vmap[faces[i*3+2]];
            nf2++;
        }
    free(comp); free(cfaces); free(vkeep); free(vmap);
    *nverts = nv2;
    return nf2;
}


/* ---------------- degenerate (zero-area) face detection ---------------- */
int cm_degenerate(const float *verts, const uint32_t *faces, int nfaces,
                  uint8_t *bad_out) {
    int n = 0;
    for (int i = 0; i < nfaces; i++) {
        uint32_t a = faces[i*3], b = faces[i*3+1], c = faces[i*3+2];
        if (a == b || b == c || a == c) { bad_out[i] = 1; n++; continue; }
        const float *va = verts + a*3, *vb = verts + b*3, *vc = verts + c*3;
        float e0x = vb[0]-va[0], e0y = vb[1]-va[1], e0z = vb[2]-va[2];
        float e1x = vc[0]-va[0], e1y = vc[1]-va[1], e1z = vc[2]-va[2];
        float nx = e0y*e1z - e0z*e1y, ny = e0z*e1x - e0x*e1z, nz = e0x*e1y - e0y*e1x;
        if (nx*nx + ny*ny + nz*nz < 1e-16f) { bad_out[i] = 1; n++; }
    }
    return n;
}

/* ---------------- winding orientation (flood fill) ---------------- */
/* Orient all faces so adjacent faces traverse shared edges in opposite
 * directions (consistent winding). xatlas charting depends on this;
 * inconsistent winding forces per-face charts. Returns the number of
 * flipped faces. */
int cm_orient(uint32_t *faces, int nfaces) {
    if (nfaces <= 0) return 0;
    /* edge records: key -> up to two faces + traversal direction bit
     * (1 = the face traverses the edge min->max) */
    typedef struct { int64_t key; int f1, d1, f2, d2; int used; } erec;
    int ecap = nfaces * 3, ne = 0;
    erec *rec = calloc((size_t)ecap, sizeof(erec));
    hmap emap = {0};
    for (int i = 0; i < nfaces; i++) {
        uint32_t a = faces[i*3], b = faces[i*3+1], c = faces[i*3+2];
        uint32_t e[3][2] = { {a,b}, {b,c}, {c,a} };
        for (int j = 0; j < 3; j++) {
            uint32_t lo = e[j][0] < e[j][1] ? e[j][0] : e[j][1];
            uint32_t hi = e[j][0] < e[j][1] ? e[j][1] : e[j][0];
            int d = (e[j][0] == lo) ? 1 : 0;
            int64_t key = ((int64_t)lo << 32) | hi;
            int ri = hmap_get(&emap, key);
            if (ri < 0) {
                if (ne >= ecap) { ecap *= 2; rec = realloc(rec, (size_t)ecap * sizeof(erec)); }
                hmap_put(&emap, key, ne);
                rec[ne].key = key; rec[ne].f1 = i; rec[ne].d1 = d;
                rec[ne].f2 = -1; rec[ne].d2 = 0; rec[ne].used = 1;
                ne++;
            } else {
                if (rec[ri].f2 < 0) { rec[ri].f2 = i; rec[ri].d2 = d; }
                /* third face on an edge (non-manifold): leave it unconstrained */
            }
        }
    }
    uint8_t *flip = calloc((size_t)nfaces, 1);
    uint8_t *seen = calloc((size_t)nfaces, 1);
    int *queue = malloc((size_t)nfaces * sizeof(int));
    int nflipped = 0;
    for (int start = 0; start < nfaces; start++) {
        if (seen[start]) continue;
        int qh = 0, qt = 0;
        queue[qt++] = start; seen[start] = 1;
        while (qh < qt) {
            int f = queue[qh++];
            uint32_t a = faces[f*3], b = faces[f*3+1], c = faces[f*3+2];
            uint32_t e[3][2] = { {a,b}, {b,c}, {c,a} };
            for (int j = 0; j < 3; j++) {
                uint32_t lo = e[j][0] < e[j][1] ? e[j][0] : e[j][1];
                uint32_t hi = e[j][0] < e[j][1] ? e[j][1] : e[j][0];
                int64_t key = ((int64_t)lo << 32) | hi;
                int ri = hmap_get(&emap, key);
                if (ri >= 0) {
                    int g = (rec[ri].f1 == f) ? rec[ri].f2 : rec[ri].f1;
                    if (g >= 0 && g != f) {
                        int df = (rec[ri].f1 == f) ? rec[ri].d1 : rec[ri].d2;
                        int dg = (rec[ri].f1 == g) ? rec[ri].d1 : rec[ri].d2;
                        int want = flip[f] ^ df ^ dg ^ 1;
                        if (!seen[g]) {
                            flip[g] = (uint8_t)want;
                            seen[g] = 1;
                            queue[qt++] = g;
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < nfaces; i++)
        if (flip[i]) {
            uint32_t t = faces[i*3+1];
            faces[i*3+1] = faces[i*3+2];
            faces[i*3+2] = t;
            nflipped++;
        }
    hmap_free(&emap);
    free(rec); free(flip); free(seen); free(queue);
    return nflipped;
}
