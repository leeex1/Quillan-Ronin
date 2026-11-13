import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=r"C:\02_QUILLAN\09 - Projects\thronglets-ronin\game\assets\ronin.glb")
import mathutils
minv = mathutils.Vector((1e9,1e9,1e9)); maxv = mathutils.Vector((-1e9,-1e9,-1e9))
for obj in bpy.context.scene.objects:
    if obj.type in ("MESH","ARMATURE"):
        bb = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box] if obj.type=="MESH" else [obj.location]
        for v in bb:
            minv = mathutils.Vector(map(min, minv, v)); maxv = mathutils.Vector(map(max, maxv, v))
        print("OBJ:", obj.name, obj.type, "loc=", tuple(round(x,3) for x in obj.location))
print("WORLD_MIN:", tuple(round(x,3) for x in minv))
print("WORLD_MAX:", tuple(round(x,3) for x in maxv))
print("SIZE:", tuple(round(maxv[i]-minv[i],3) for i in range(3)))
