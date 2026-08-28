/* cmesh.h — C mesh kernels for the 3DModelGen Go mesh pipeline.
 * Flat arrays only: verts are xyz float32 triples, faces are uint32 triples.
 * No allocation inside the hot kernels beyond explicit scratch passed by the
 * caller (or malloc'd + freed here). All memory is caller-owned. */
#ifndef CMESH_H
#define CMESH_H

#include <stdint.h>

/* Connected components of the face graph (union-find over vertex edges).
 * comp_out receives a component id per vertex (0..ncomp-1). Returns ncomp. */
int cm_components(const uint32_t *faces, int nfaces, int nverts, int *comp_out);

/* Flag faces incident to a non-manifold edge (an edge shared by > 2 faces).
 * bad_out is nfaces bytes (1 = drop). Returns the number of bad faces. */
int cm_nonmanifold(const uint32_t *faces, int nfaces, int nverts, uint8_t *bad_out);

/* Bilateral normal smoothing in place: each face's normal is pulled toward
 * the similarity-weighted mean of its edge-neighbors' normals (weighted by
 * normal agreement and centroid distance), then vertices are re-projected
 * onto the smoothed face planes. Preserves curvature. */
void cm_smooth_bilateral(float *verts, const uint32_t *faces, int nverts,
                         int nfaces, int iters, float sigma_n, float sigma_c);

/* Drop faces whose connected component is smaller than min_frac of all faces,
 * then compact verts + faces in place. Returns the new face count;
 * *nverts is updated. */
int cm_keep_components(float *verts, uint32_t *faces, int *nverts, int nfaces,
                       float min_frac);

/* Flag zero-area / duplicate-index faces; bad_out is nfaces bytes. */
int cm_degenerate(const float *verts, const uint32_t *faces, int nfaces,
                  uint8_t *bad_out);

/* Orient all faces to consistent winding (adjacent faces traverse shared
 * edges oppositely). Returns the number of flipped faces. */
int cm_orient(uint32_t *faces, int nfaces);

#endif
