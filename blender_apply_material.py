"""Import a generated 3DModelGen asset into Blender, apply the PBR material
(texture included), set smooth shading, and save a ready-to-open .blend.

Usage:
    blender -b -P blender_apply_material.py -- <input.glb|.obj> <output.blend>

- GLB: texture is embedded; the importer already builds the material, this
  script guarantees smooth shading and saves the file.
- OBJ: keep the .mtl + texture .png next to the .obj (as in the _obj.zip);
  the script wires the first PNG found next to it into the Principled BSDF if
  the importer did not.
"""
import os
import sys

# Blender 5.0.1 on this box runs on the conda interpreter but strips
# site-packages from sys.path, so bundled addons (gltf importer) cannot find
# numpy. Re-add it explicitly; override with BLENDER_PYTHONPATH if relocated.
_sp = os.environ.get('BLENDER_PYTHONPATH', '/home/pipo/miniconda3/lib/python3.14/site-packages')
if os.path.isdir(_sp) and _sp not in sys.path:
    sys.path.append(_sp)

import bpy


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    if len(argv) < 2:
        sys.exit('usage: blender -b -P blender_apply_material.py -- <asset> <out.blend>')
    src, dst = argv[0], argv[1]
    if not os.path.exists(src):
        sys.exit(f'not found: {src}')

    bpy.ops.wm.read_factory_settings(use_empty=True)

    ext = os.path.splitext(src)[1].lower()
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=src)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=src)
    else:
        sys.exit(f'unsupported extension: {ext}')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.shade_smooth()

    # PNG next to the OBJ (material_0.png from the _obj.zip), used only if the
    # importer left the material without an image.
    img = None
    if ext == '.obj':
        for fn in sorted(os.listdir(os.path.dirname(os.path.abspath(src)))):
            if fn.lower().endswith('.png'):
                img = bpy.data.images.load(os.path.join(os.path.dirname(os.path.abspath(src)), fn))
                break

    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        if not obj.data.materials:
            obj.data.materials.append(bpy.data.materials.new('3DModelGen'))
        for mat in obj.data.materials:
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get('Principled BSDF')
            tex = None
            for n in mat.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image is not None:
                    tex = n
                    break
            if tex is None and img is not None and bsdf is not None:
                tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
                tex.image = img
            if tex is not None and bsdf is not None:
                mat.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])

    bpy.ops.wm.save_as_mainfile(filepath=dst)
    print(f'saved: {dst}')


if __name__ == '__main__':
    main()
