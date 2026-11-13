"""Ronin Thronglets — rigged ronin builder.
Creates an armature-rigged ronin with 4 animation clips (Idle, Walk, Attack, Celebrate)
and exports ronin.glb (with animations) for Godot.
Run: blender -b -P rig_export.py -- <out_glb>
"""

import bpy, sys, math, os

OUT = sys.argv[sys.argv.index("--") + 1]

# ---------- clean ----------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for a in list(bpy.data.actions) + list(bpy.data.armatures):
    pass


def mat(name, color, metallic=0.9, rough=0.35, emissive=(0, 0, 0), strength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    if any(c > 0 for c in emissive):
        bsdf.inputs["Emission Color"].default_value = (*emissive, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m


OBSIDIAN = mat("Obsidian", (0.05, 0.05, 0.07), 0.95, 0.25)
CRIMSON = mat("CrimsonLacquer", (0.45, 0.02, 0.03), 0.4, 0.3)
GOLD = mat("GoldTrim", (0.8, 0.55, 0.1), 1.0, 0.2)
STEEL = mat("Steel", (0.35, 0.36, 0.4), 1.0, 0.4)
VISOR = mat(
    "RedVisor", (0.1, 0.0, 0.0), 0.2, 0.4, emissive=(1.0, 0.05, 0.05), strength=6.0
)
TEAL = mat(
    "TealCircuit", (0.0, 0.1, 0.1), 0.3, 0.5, emissive=(0.0, 0.9, 0.85), strength=3.5
)
CLOTH = mat("DarkSash", (0.12, 0.03, 0.03), 0.0, 0.8)


def add_mesh(primitive, name, loc, rot=(0, 0, 0), scale=(1, 1, 1), material=None, **kw):
    if primitive == "uv_sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=24, ring_count=16, radius=1, location=loc
        )
    elif primitive == "cone":
        bpy.ops.mesh.primitive_cone_add(
            vertices=24,
            radius1=kw.get("radius1", 1),
            radius2=kw.get("radius2", 0),
            depth=kw.get("depth", 1),
            location=loc,
        )
    elif primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=24,
            radius=kw.get("radius", 1),
            depth=kw.get("depth", 1),
            location=loc,
        )
    elif primitive == "box":
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    elif primitive == "torus":
        bpy.ops.mesh.primitive_torus_add(
            major_radius=kw.get("major", 1),
            minor_radius=kw.get("minor", 0.15),
            major_segments=24,
            minor_segments=8,
            location=loc,
        )
    ob = bpy.context.active_object
    ob.name = name
    ob.rotation_euler = rot
    ob.scale = scale
    if material:
        ob.data.materials.append(material)
    return ob


parts = {}
parts["Belt"] = add_mesh(
    "cylinder", "Belt", (0, 0, 0.68), radius=0.42, depth=0.16, material=CLOTH
)
parts["Body"] = add_mesh(
    "uv_sphere", "Body", (0, 0, 0.95), scale=(0.52, 0.45, 0.62), material=CRIMSON
)
parts["Chest"] = add_mesh(
    "uv_sphere", "Chest", (0, -0.28, 1.02), scale=(0.4, 0.18, 0.42), material=OBSIDIAN
)
parts["Emblem"] = add_mesh(
    "cylinder",
    "Emblem",
    (0, -0.44, 1.05),
    rot=(math.radians(90), 0, 0),
    radius=0.09,
    depth=0.02,
    material=TEAL,
)
parts["Head"] = add_mesh(
    "uv_sphere", "Head", (0, 0, 1.72), scale=(0.34, 0.32, 0.32), material=OBSIDIAN
)
parts["Visor"] = add_mesh(
    "box", "Visor", (0, -0.30, 1.72), scale=(0.42, 0.06, 0.09), material=VISOR
)
parts["Brim"] = add_mesh(
    "cylinder", "Brim", (0, 0, 1.86), radius=0.42, depth=0.09, material=CRIMSON
)
parts["Dome"] = add_mesh(
    "cone",
    "Dome",
    (0, 0, 2.05),
    radius1=0.33,
    radius2=0.06,
    depth=0.35,
    material=OBSIDIAN,
)
parts["Maedate"] = add_mesh(
    "torus",
    "Maedate",
    (0, -0.02, 2.22),
    rot=(math.radians(20), 0, 0),
    major=0.16,
    minor=0.045,
    material=GOLD,
)
parts["ShoulderL"] = add_mesh(
    "uv_sphere",
    "ShoulderL",
    (-0.52, 0, 1.28),
    scale=(0.24, 0.22, 0.2),
    material=CRIMSON,
)
parts["ShoulderR"] = add_mesh(
    "uv_sphere", "ShoulderR", (0.52, 0, 1.28), scale=(0.24, 0.22, 0.2), material=CRIMSON
)
parts["ArmL"] = add_mesh(
    "cylinder",
    "ArmL",
    (-0.6, 0, 0.92),
    rot=(0, math.radians(12), 0),
    radius=0.13,
    depth=0.55,
    material=OBSIDIAN,
)
parts["ArmR"] = add_mesh(
    "cylinder",
    "ArmR",
    (0.6, 0, 0.92),
    rot=(0, math.radians(-12), 0),
    radius=0.13,
    depth=0.55,
    material=OBSIDIAN,
)
parts["LegL"] = add_mesh(
    "cylinder", "LegL", (-0.24, 0, 0.28), radius=0.16, depth=0.5, material=OBSIDIAN
)
parts["LegR"] = add_mesh(
    "cylinder", "LegR", (0.24, 0, 0.28), radius=0.16, depth=0.5, material=OBSIDIAN
)
parts["Katana"] = add_mesh(
    "cylinder",
    "Katana",
    (0.1, 0.28, 1.25),
    rot=(math.radians(35), 0, math.radians(35)),
    radius=0.05,
    depth=1.15,
    material=STEEL,
)
parts["Hilt"] = add_mesh(
    "cylinder",
    "Hilt",
    (-0.28, 0.1, 1.66),
    rot=(math.radians(35), 0, math.radians(35)),
    radius=0.07,
    depth=0.22,
    material=GOLD,
)

# ---------- armature ----------
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
arm = bpy.context.active_object
arm.name = "RoninRig"
ebm = arm.data
# strip default bone
for b in list(ebm.edit_bones):
    ebm.edit_bones.remove(b)


def bone(name, head, tail, parent=None):
    b = ebm.edit_bones.new(name)
    b.head = head
    b.tail = tail
    if parent:
        b.parent = ebm.edit_bones[parent]
    return b


bone("Hips", (0, 0, 0.45), (0, 0, 0.75))
bone("Spine", (0, 0, 0.75), (0, 0, 1.15), "Hips")
bone("Chest", (0, 0, 1.15), (0, 0, 1.42), "Spine")
bone("Head", (0, 0, 1.42), (0, 0, 1.80), "Chest")
bone("ArmL", (-0.52, 0, 1.28), (-0.62, 0, 0.72), "Chest")
bone("ArmR", (0.52, 0, 1.28), (0.62, 0, 0.72), "Chest")
bone("LegL", (-0.24, 0, 0.55), (-0.24, 0, 0.02), "Hips")
bone("LegR", (0.24, 0, 0.55), (0.24, 0, 0.02), "Hips")
bpy.ops.object.mode_set(mode="OBJECT")

# ---------- parent parts to bones ----------
BONE_OF = {
    "Belt": "Hips",
    "Body": "Spine",
    "Chest": "Chest",
    "Emblem": "Chest",
    "Head": "Head",
    "Visor": "Head",
    "Brim": "Head",
    "Dome": "Head",
    "Maedate": "Head",
    "ShoulderL": "ArmL",
    "ShoulderR": "ArmR",
    "ArmL": "ArmL",
    "ArmR": "ArmR",
    "LegL": "LegL",
    "LegR": "LegR",
    "Katana": "Spine",
    "Hilt": "Spine",
}
for pname, bname in BONE_OF.items():
    p = parts[pname]
    p.parent = arm
    p.parent_type = "BONE"
    p.parent_bone = bname


# ---------- animations ----------
def new_action(name):
    act = bpy.data.actions.new(name)
    arm.animation_data_create()
    arm.animation_data.action = act
    return act


def key_bone(bname, frame, rot=None, loc=None, scale=None):
    pb = arm.pose.bones[bname]
    if rot is not None:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = rot
        pb.keyframe_insert("rotation_euler", frame=frame)
    if loc is not None:
        pb.location = loc
        pb.keyframe_insert("location", frame=frame)
    if scale is not None:
        pb.scale = scale
        pb.keyframe_insert("scale", frame=frame)


def pose_reset():
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
        pb.location = (0, 0, 0)
        pb.scale = (1, 1, 1)


DEG = math.radians

# ---- CLIP: Idle (32 frames, loops) ----
pose_reset()
act = new_action("Idle")
for f, breathe in [(1, 1.0), (9, 1.045), (17, 1.0), (25, 1.045), (33, 1.0)]:
    key_bone(
        "Spine",
        f,
        rot=(DEG(2 * (breathe - 1) / 0.045 * 0.5), 0, 0),
        scale=(1, 1, breathe),
    )
    key_bone("Chest", f, rot=(DEG(-1.5 * (breathe - 1) / 0.045 * 0.5), 0, 0))
key_bone("Head", 1, rot=(DEG(0), 0, 0))
key_bone("Head", 17, rot=(DEG(2), 0, DEG(1.2)))
key_bone("Head", 33, rot=(DEG(0), 0, 0))
key_bone("ArmL", 1, rot=(0, 0, DEG(4)))
key_bone("ArmR", 1, rot=(0, 0, DEG(-4)))
key_bone("ArmL", 17, rot=(0, 0, DEG(7)))
key_bone("ArmR", 17, rot=(0, 0, DEG(-7)))
key_bone("ArmL", 33, rot=(0, 0, DEG(4)))
key_bone("ArmR", 33, rot=(0, 0, DEG(-4)))
arm.animation_data.action = act

# ---- CLIP: Walk (24 frames, loops) ----
pose_reset()
act = new_action("Walk")
stride = [0, 12, 24]
for i, f in enumerate(stride):
    phase = [1, -1, 1][i]
    key_bone("LegL", f, rot=(DEG(32 * phase), 0, 0))
    key_bone("LegR", f, rot=(DEG(-32 * phase), 0, 0))
    key_bone("ArmL", f, rot=(DEG(-24 * phase), 0, DEG(4)))
    key_bone("ArmR", f, rot=(DEG(24 * phase), 0, DEG(-4)))
    key_bone("Hips", f, loc=(0, 0, 0.06 if phase == 1 else 0.0))
    key_bone("Chest", f, rot=(DEG(3), 0, DEG(2.5 * phase)))
    key_bone("Head", f, rot=(DEG(-2), 0, DEG(-1.5 * phase)))
arm.animation_data.action = act

# ---- CLIP: Attack (18 frames, one-shot) ----
pose_reset()
act = new_action("Attack")
key_bone("ArmR", 1, rot=(DEG(-15), 0, DEG(-4)))
key_bone("ArmR", 6, rot=(DEG(-150), 0, DEG(-30)))
key_bone("Chest", 6, rot=(DEG(-6), 0, DEG(22)))
key_bone("Head", 6, rot=(DEG(4), 0, DEG(-8)))
key_bone("ArmR", 11, rot=(DEG(55), 0, DEG(10)))
key_bone("Chest", 11, rot=(DEG(8), 0, DEG(-18)))
key_bone("Head", 11, rot=(DEG(-2), 0, DEG(6)))
key_bone("ArmR", 18, rot=(DEG(-15), 0, DEG(-4)))
key_bone("Chest", 18, rot=(0, 0, 0))
key_bone("Head", 18, rot=(0, 0, 0))
arm.animation_data.action = act

# ---- CLIP: Celebrate (28 frames) ----
pose_reset()
act = new_action("Celebrate")
key_bone("ArmL", 1, rot=(0, 0, DEG(4)))
key_bone("ArmR", 1, rot=(0, 0, DEG(-4)))
key_bone("ArmL", 8, rot=(0, 0, DEG(160)))
key_bone("ArmR", 8, rot=(0, 0, DEG(-160)))
key_bone("Spine", 8, scale=(1, 1, 1.06))
key_bone("LegL", 8, loc=(0, 0, 0.14))
key_bone("LegR", 8, loc=(0, 0, 0.14))
for f in (10, 14, 18):
    key_bone("Hips", f, loc=(0, 0, 0.16 if f == 10 else (0.0 if f == 14 else 0.16)))
key_bone("ArmL", 24, rot=(0, 0, DEG(20)))
key_bone("ArmR", 24, rot=(0, 0, DEG(-20)))
key_bone("ArmL", 28, rot=(0, 0, DEG(4)))
key_bone("ArmR", 28, rot=(0, 0, DEG(-4)))
key_bone("Spine", 28, scale=(1, 1, 1.0))
key_bone("Hips", 1, loc=(0, 0, 0))
key_bone("Hips", 28, loc=(0, 0, 0))
key_bone("LegL", 28, loc=(0, 0, 0))
key_bone("LegR", 28, loc=(0, 0, 0))
arm.animation_data.action = act

# ---------- push actions into NLA tracks so glTF exports them as clips ----------
arm.animation_data.action = None
for clip in ["Idle", "Walk", "Attack", "Celebrate"]:
    track = arm.animation_data.nla_tracks.new()
    track.name = clip
    a = bpy.data.actions[clip]
    strip = track.strips.new(clip, 1, a)
    track.mute = False

# ---------- export ----------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(
    filepath=OUT,
    export_format="GLB",
    use_selection=True,
    export_animations=True,
    export_nla_strips=True,
    export_animation_mode="ACTIONS",
)
print("RIGGED EXPORT DONE ->", OUT)
