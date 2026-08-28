"""Low-VRAM mesh decode for TRELLIS v1.

decode_mesh_low_vram: run torso + upsample + out_layer GLOBALLY on the 64^3
latent (sparse 256^3 surface output, ~600K cells -- memory-safe), subsample
the surface field to a 128^3 lattice, then run ONE single-grid FlexiCubes
extraction. The 128^3 dense grids (~40MB) fit in 8GB VRAM, and a single grid
means NO tile seams: the mesh is one connected manifold.

(Tiled extraction was tried and abandoned: overlapping tiles produce near-
duplicate faces / non-manifold edges that no weld or repair could cleanly
resolve, and the quadric decimation shattered on them.)
"""
import torch

from trellis.representations.mesh.utils_cube import (
    construct_dense_grid, sparse_cube2verts, get_dense_attrs, get_defomed_verts)


def _rss_mb():
    """Current process DRAM RSS in MB (Linux /proc/self/status)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


_prev_mem = {}


def memlog(tag):
    """Memory snapshot: GPU allocated/reserved + DRAM RSS, deltas vs last."""
    ga = torch.cuda.memory_allocated() / 2 ** 20
    gr = torch.cuda.memory_reserved() / 2 ** 20
    rss = _rss_mb()
    d_ga = ga - _prev_mem.get('ga', ga)
    d_rss = rss - _prev_mem.get('rss', rss)
    _prev_mem.update(ga=ga, rss=rss)
    print(f'[mem] {tag}: gpu_alloc={ga:.0f}MB ({d_ga:+.0f}) '
          f'gpu_resv={gr:.0f}MB dram_rss={rss:.0f}MB ({d_rss:+.0f})', flush=True)


def _layout(use_color):
    L = {'sdf': (8, 1), 'deform': (8, 3), 'weights': (21,)}
    if use_color:
        L['color'] = (8, 6)
    ranges, start = {}, 0
    for k, shape in L.items():
        size = 1
        for s in shape:
            size *= s
        ranges[k] = (start, start + size)
        start += size
    return ranges


def _split(feats, ranges):
    sdf = feats[:, ranges['sdf'][0]:ranges['sdf'][1]].reshape(-1, 8, 1)
    deform = feats[:, ranges['deform'][0]:ranges['deform'][1]].reshape(-1, 8, 3)
    weights = feats[:, ranges['weights'][0]:ranges['weights'][1]].reshape(-1, 21)
    color = None
    if 'color' in ranges:
        color = feats[:, ranges['color'][0]:ranges['color'][1]].reshape(-1, 8, 6)
    return sdf, deform, color, weights


def extract_local_region(extractor, coords, feats, res=128, device='cuda'):
    """Single-region FlexiCubes extraction (the official SparseFeatures2Mesh
    step) on the grid spanned by `coords`. `coords` are lattice indices in
    [0, res); vertices are returned in model space [-0.5, 0.5].

    The deform displacement uses the trained 256-grid scale (1/512) so the
    geometry matches the official decode regardless of the extraction res.
    """
    ranges = _layout(getattr(extractor, 'use_color', True))
    sdf_bias = getattr(extractor, 'sdf_bias', -1.0 / res)
    use_color = getattr(extractor, 'use_color', True)
    flex = extractor.mesh_extractor

    cmin = coords.min(0).values
    cmax = coords.max(0).values
    local = coords - cmin
    L = int((cmax - cmin + 1).max().item())  # cubic local grid of the extent

    sdf, deform, color, weights = _split(feats, ranges)
    sdf = sdf + sdf_bias
    v_attrs = [sdf, deform, color] if use_color else [sdf, deform]
    v_pos, v_attrs_t, _ = sparse_cube2verts(local, torch.cat(v_attrs, dim=-1))
    v_attrs_d = get_dense_attrs(v_pos, v_attrs_t, res=L + 1, sdf_init=True)
    weights_d = get_dense_attrs(local, weights, res=L, sdf_init=False)
    sdf_d = v_attrs_d[..., 0]
    deform_d = v_attrs_d[..., 1:4]
    colors_d = v_attrs_d[..., 4:] if use_color else None

    verts_local, cube_local = construct_dense_grid(L, device)
    # global lattice positions: (local + cmin) / res - 0.5
    verts_global = (verts_local.to(feats.dtype) + cmin) / res - 0.5
    x_nx3 = verts_global + (1 - 1e-8) / 512.0 * torch.tanh(deform_d)
    vertices, faces, _, colors_t = flex(
        voxelgrid_vertices=x_nx3, scalar_field=sdf_d, cube_idx=cube_local,
        resolution=L, beta=weights_d[:, :12], alpha=weights_d[:, 12:20],
        gamma_f=weights_d[:, 20], voxelgrid_colors=colors_d)
    if vertices.shape[0] == 0:
        return None
    return vertices, faces, colors_t


def decode_mesh_low_vram(dec, slat, res=128, device='cuda'):
    """Global torso + upsample + out_layer, subsample the 256^3 surface field
    to `res`^3, then ONE single-grid FlexiCubes extraction -> a clean,
    connected manifold (no tile seams).

    dec: an SLatMeshDecoder (has .input_layer, .pos_embedder, .blocks,
    .upsample, .out_layer, .dtype, .pe_mode, .mesh_extractor). slat: the 64^3
    slat (fp32 recommended).
    """
    from trellis.modules import sparse as sp
    from trellis.representations.mesh.cube2mesh import MeshExtractResult

    h = dec.input_layer(slat)
    if dec.pe_mode == 'ape':
        h = h + dec.pos_embedder(slat.coords[:, 1:])
    h = h.type(dec.dtype)
    for block in dec.blocks:
        h = block(h)
    # upsample + out_layer per 64^3 chunk (halo for feature consistency at the
    # borders, which the 128^3 subsample averages out). Running them on the
    # full latent OOMs spconv/cumm's workspace on large surfaces.
    hc = h.coords[:, 1:].long()
    hf = h.feats
    N = dec.resolution  # 64
    parts = []
    for ox in range(0, N, 32):
        for oy in range(0, N, 32):
            for oz in range(0, N, 32):
                lo = torch.tensor([max(0, ox - 2), max(0, oy - 2),
                                   max(0, oz - 2)], device=device)
                hi = torch.tensor([min(N, ox + 34), min(N, oy + 34),
                                   min(N, oz + 34)], device=device)
                mask = ((hc >= lo) & (hc < hi)).all(dim=1)
                if mask.sum() == 0:
                    continue
                sub = sp.SparseTensor(feats=hf[mask], coords=h.coords[mask])
                for blk in dec.upsample:
                    sub = blk(sub)
                sub = sub.type(slat.dtype)
                parts.append(dec.out_layer(sub))
    coords = torch.cat([p.coords[:, 1:] for p in parts], 0).long()  # 256-space
    feats = torch.cat([p.feats for p in parts], 0)
    # merge exact-dup cells from the chunk halos (mean features)
    uniq, inv = torch.unique(coords, dim=0, return_inverse=True)
    if uniq.shape[0] < coords.shape[0]:
        fsum = torch.zeros(uniq.shape[0], feats.shape[1],
                           device=feats.device, dtype=feats.dtype)
        fsum.index_add_(0, inv, feats)
        fsum /= torch.bincount(inv, minlength=uniq.shape[0]).unsqueeze(1)
        coords, feats = uniq, fsum
    memlog(f'chunked out_layer done: {coords.shape[0]} cells')
    # subsample 256^3 -> res^3 so a single-grid extraction fits in VRAM
    # coords * res // 256, NOT coords // (256 // res): the latter is only exact
    # when 256 % res == 0 (128/64/32); 96 -> 128-lattice / 96 = 1.33x inflated
    # mesh, 160 -> 256-lattice / 160 = 1.6x. Any res in (0, 256] is correct.
    csub = coords * res // 256
    uniq, inv = torch.unique(csub, dim=0, return_inverse=True)
    if uniq.shape[0] < coords.shape[0]:
        fsub = torch.zeros(uniq.shape[0], feats.shape[1],
                           device=feats.device, dtype=feats.dtype)
        fsub.index_add_(0, inv, feats)
        fsub /= torch.bincount(inv, minlength=uniq.shape[0]).unsqueeze(1)
        coords, feats = uniq, fsub
        memlog(f'field subsampled: {feats.shape[0]} cells @ {res}^3')

    r = extract_local_region(dec.mesh_extractor, coords, feats,
                             res=res, device=device)
    if r is None:
        return MeshExtractResult(vertices=torch.zeros(0, 3, device=device),
                                 faces=torch.zeros(0, 3, dtype=torch.long,
                                                   device=device),
                                 vertex_attrs=None, res=res)
    v, f, c = r
    return MeshExtractResult(vertices=v, faces=f, vertex_attrs=c, res=res)
