#!/usr/bin/env python
"""convert_glb_to_obj.py - .glb -> .obj + .mtl + textures (Blender/WebGL-ready).

Produces one .obj per mesh in the GLB. Each .obj references an .mtl, and the
.mtl's map_Kd points at a PNG saved next to it - so Blender (Import > Wavefront
OBJ) and three.js (OBJLoader + MTLLoader) load textures with zero fixing.

Notes:
- glTF units are meters; OBJ has no unit. TRELLIS assets are normalized to
  roughly [-0.5, 0.5], which is fine for Blender and WebGL at 1:1 scale.
- For WebGL you can also load the original .glb directly via three.js
  GLTFLoader (no conversion needed) - OBJ is only required for Blender editing
  or legacy pipelines.
- RGBA textures keep their alpha channel (TRELLIS uses it for translucency).

Output is ALWAYS written next to the input .glb (its directory), so
the OBJ/MTL/texture live with the source model. With --zip, one
model_obj.zip (obj + mtl + textures) is written instead - unzip it and import
the .obj in Blender; the material and texture resolve automatically.

Usage:
    python convert_glb_to_obj.py /path/to/model.glb [--zip]
"""
import os
import re
import sys

import trimesh
from PIL import Image


def _save_texture(mesh, mtl_dir, name):
    """Save the mesh's texture under `name` inside mtl_dir. Returns bytes or 0."""
    mat = getattr(mesh.visual, 'material', None)
    if mat is None:
        return 0
    tex = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
    if tex is None:
        return 0
    path = os.path.join(mtl_dir, name)
    if hasattr(tex, 'save'):
        tex.save(path)
    else:
        Image.fromarray(tex).save(path)
    return os.path.getsize(path)


def _export_mesh(mesh, obj_path):
    """Export one mesh to OBJ and wire its textures. Returns (obj, tex_count)."""
    os.makedirs(os.path.dirname(obj_path) or '.', exist_ok=True)
    mesh.export(obj_path)
    # resolve the mtl path from the obj's mtllib line
    mtl_name = None
    for line in open(obj_path):
        if line.startswith('mtllib'):
            mtl_name = line.split()[1]
            break
    saved = 0
    if mtl_name:
        mtl_path = os.path.join(os.path.dirname(obj_path), mtl_name)
        names = re.findall(r'map_Kd\s+(\S+)', open(mtl_path).read())
        for name in names:
            saved += _save_texture(mesh, os.path.dirname(obj_path), name)
    return obj_path, saved


def _collect_outputs(results):
    files = []
    for obj in results:
        d = os.path.dirname(obj)
        files.append(obj)
        for f in sorted(os.listdir(d)):
            if f.endswith(('.mtl', '.png', '.jpg', '.jpeg')):
                files.append(os.path.join(d, f))
    return files


def _make_zip(out_dir, base, files):
    import zipfile
    zip_path = os.path.join(out_dir, f'{base}_obj.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.relpath(f, out_dir))
    return zip_path


def convert(glb_path, out_dir=None, as_zip=False, target_tris=0):
    glb_path = os.path.abspath(glb_path)
    if not os.path.exists(glb_path):
        raise SystemExit(f'not found: {glb_path}')
    out_dir = os.path.abspath(out_dir or os.path.dirname(glb_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(glb_path))[0]

    scene = trimesh.load(glb_path, force='scene')
    geoms = list(scene.geometry.values())
    if not geoms:
        raise SystemExit(f'no mesh geometry in {glb_path}')

    # optional quadric decimation for low-poly / flat-faceted output
    if target_tris:
        import numpy as np
        import fast_simplification as fs
        dec = []
        for gm in geoms:
            nv, nf = fs.simplify(gm.vertices.astype(np.float64),
                                 gm.faces.astype(np.int64),
                                 target_count=target_tris)
            gm = trimesh.Trimesh(vertices=nv, faces=nf, process=True)
            gm.visual = gm.visual  # drop stale visual if geometry changed
            dec.append(gm)
        geoms = dec
    results = []
    if len(geoms) == 1:
        obj, saved = _export_mesh(geoms[0], os.path.join(out_dir, f'{base}.obj'))
        results.append(obj)
        print(f'{obj}: {len(geoms[0].vertices)} verts, {len(geoms[0].faces)} faces, {saved} texture(s)')
    else:
        # multiple geometries with different materials: one obj each in a subdir
        # (trimesh names every texture material_N.png, so subdirs avoid clashes)
        for i, mesh in enumerate(geoms):
            sub = os.path.join(out_dir, f'{base}_{i}')
            os.makedirs(sub, exist_ok=True)
            obj, saved = _export_mesh(mesh, os.path.join(sub, f'{base}.obj'))
            results.append(obj)
            print(f'{obj}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, {saved} texture(s)')
    if as_zip:
        zip_path = _make_zip(out_dir, base, _collect_outputs(results))
        print(f'{zip_path}')
        return zip_path
    return results[0] if len(results) == 1 else results


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if not args:
        raise SystemExit(__doc__)
    tris = 0
    if '--tris' in sys.argv:
        i = sys.argv.index('--tris')
        tris = int(sys.argv[i + 1])
    convert(args[0], as_zip='--zip' in sys.argv, target_tris=tris)
