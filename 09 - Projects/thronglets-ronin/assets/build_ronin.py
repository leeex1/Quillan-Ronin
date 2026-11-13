"""Blender headless builder for Ronin Thronglets assets.
Run: blender -b -P build_ronin.py -- <out_dir>
Creates: ronin.glb (game model), portrait.png, keyart.png
"""

import bpy, sys, math, os, random
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "."
os.makedirs(OUT, exist_ok=True)

# ---------- clean scene ----------
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


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


def make_mats():
    global OBSIDIAN, CRIMSON, GOLD, STEEL, VISOR, TEAL, CLOTH
    OBSIDIAN = mat("Obsidian", (0.05, 0.05, 0.07), 0.95, 0.25)
    CRIMSON = mat("CrimsonLacquer", (0.45, 0.02, 0.03), 0.4, 0.3)
    GOLD = mat("GoldTrim", (0.8, 0.55, 0.1), 1.0, 0.2)
    STEEL = mat("Steel", (0.35, 0.36, 0.4), 1.0, 0.4)
    VISOR = mat(
        "RedVisor", (0.1, 0.0, 0.0), 0.2, 0.4, emissive=(1.0, 0.05, 0.05), strength=6.0
    )
    TEAL = mat(
        "TealCircuit",
        (0.0, 0.1, 0.1),
        0.3,
        0.5,
        emissive=(0.0, 0.9, 0.85),
        strength=3.5,
    )
    CLOTH = mat("DarkSash", (0.12, 0.03, 0.03), 0.0, 0.8)


make_mats()


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


def build_ronin(prefix="Ronin", at=(0, 0, 0), s=1.0, visor_mat=VISOR):
    g = bpy.data.objects.new(prefix + "_Root", None)
    g.empty_display_type = "PLAIN_AXES"
    bpy.context.scene.collection.objects.link(g)
    parts = []
    # body: squat rounded torso
    parts.append(
        add_mesh(
            "uv_sphere",
            prefix + "_Body",
            (0, 0, 0.95),
            scale=(0.52, 0.45, 0.62),
            material=CRIMSON,
        )
    )
    # chest plate
    parts.append(
        add_mesh(
            "uv_sphere",
            prefix + "_Chest",
            (0, -0.28, 1.02),
            scale=(0.4, 0.18, 0.42),
            material=OBSIDIAN,
        )
    )
    # chest teal emblem
    parts.append(
        add_mesh(
            "cylinder",
            prefix + "_Emblem",
            (0, -0.44, 1.05),
            rot=(math.radians(90), 0, 0),
            radius=0.09,
            depth=0.02,
            material=TEAL,
        )
    )
    # sash / belt
    parts.append(
        add_mesh(
            "cylinder",
            prefix + "_Belt",
            (0, 0, 0.68),
            radius=0.42,
            depth=0.16,
            material=CLOTH,
        )
    )
    # head
    parts.append(
        add_mesh(
            "uv_sphere",
            prefix + "_Head",
            (0, 0, 1.72),
            scale=(0.34, 0.32, 0.32),
            material=OBSIDIAN,
        )
    )
    # kabuto brim
    parts.append(
        add_mesh(
            "cylinder",
            prefix + "_Brim",
            (0, 0, 1.86),
            radius=0.42,
            depth=0.09,
            material=CRIMSON,
        )
    )
    # kabuto dome
    parts.append(
        add_mesh(
            "cone",
            prefix + "_Dome",
            (0, 0, 2.05),
            radius1=0.33,
            radius2=0.06,
            depth=0.35,
            material=OBSIDIAN,
        )
    )
    # maedate: golden crescent (torus)
    parts.append(
        add_mesh(
            "torus",
            prefix + "_Maedate",
            (0, -0.02, 2.22),
            rot=(math.radians(20), 0, 0),
            major=0.16,
            minor=0.045,
            material=GOLD,
        )
    )
    # visor slit
    parts.append(
        add_mesh(
            "box",
            prefix + "_Visor",
            (0, -0.30, 1.72),
            scale=(0.42, 0.06, 0.09),
            material=visor_mat,
        )
    )
    # shoulder plates (halved spheres faked with scaled spheres)
    for sx in (-1, 1):
        parts.append(
            add_mesh(
                "uv_sphere",
                prefix + f"_Shoulder{sx}",
                (0.52 * sx, 0, 1.28),
                scale=(0.24, 0.22, 0.2),
                material=CRIMSON,
            )
        )
        # arm
        parts.append(
            add_mesh(
                "cylinder",
                prefix + f"_Arm{sx}",
                (0.6 * sx, 0, 0.92),
                rot=(0, math.radians(12 * -sx), 0),
                radius=0.13,
                depth=0.55,
                material=OBSIDIAN,
            )
        )
    # legs
    for sx in (-1, 1):
        parts.append(
            add_mesh(
                "cylinder",
                prefix + f"_Leg{sx}",
                (0.24 * sx, 0, 0.28),
                radius=0.16,
                depth=0.5,
                material=OBSIDIAN,
            )
        )
    # katana across back (scabbard)
    parts.append(
        add_mesh(
            "cylinder",
            prefix + "_Katana",
            (0.1, 0.28, 1.25),
            rot=(math.radians(35), 0, math.radians(35)),
            radius=0.05,
            depth=1.15,
            material=STEEL,
        )
    )
    parts.append(
        add_mesh(
            "cylinder",
            prefix + "_KatanaHilt",
            (-0.28, 0.1, 1.66),
            rot=(math.radians(35), 0, math.radians(35)),
            radius=0.07,
            depth=0.22,
            material=GOLD,
        )
    )
    for p in parts:
        p.parent = g
    g.location = at
    g.scale = (s, s, s)
    return g


def add_light(name, loc, energy, size=2.0, color=(1, 1, 1)):
    bpy.ops.object.light_add(type="AREA", location=loc)
    L = bpy.context.active_object
    L.name = name
    L.data.energy = energy
    L.data.size = size
    L.data.color = color
    return L


def setup_world(color_top=(0.02, 0.02, 0.05), strength=0.6):
    world = bpy.data.worlds.new("World") if not bpy.data.worlds else bpy.data.worlds[0]
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (*color_top, 1.0)
    bg.inputs[1].default_value = strength


# =========================================================
# 1) GAME MODEL EXPORT
# =========================================================
setup_world()
r = build_ronin("Ronin")
# simple rig-friendly origin at feet
bpy.ops.object.select_all(action="DESELECT")
for o in bpy.data.objects:
    if o.name.startswith("Ronin"):
        o.select_set(True)
bpy.ops.export_scene.gltf(
    filepath=os.path.join(OUT, "ronin.glb"), export_format="GLB", use_selection=False
)

# wipe for renders
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


# =========================================================
# 2) PORTRAIT RENDER
# =========================================================
setup_world((0.01, 0.01, 0.03), 0.3)
r = build_ronin("PortraitRonin", (0, 0, 0))
add_light("KeyLight", (2.2, -2.5, 3.0), 900, 2.5, (1.0, 0.85, 0.75))
add_light("RimLight", (-2.5, 2.0, 2.5), 1200, 2.0, (0.4, 0.8, 1.0))
add_light("FillRed", (0.5, -1.0, 1.2), 250, 3.0, (1.0, 0.15, 0.12))
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 1.35))
chest_target = bpy.context.active_object
bpy.ops.object.camera_add(location=(0, -3.6, 1.6), rotation=(math.radians(88), 0, 0))
cam = bpy.context.active_object
cam.data.lens = 60
bpy.context.scene.camera = cam
# aim camera
track = cam.constraints.new(type="TRACK_TO")
track.target = chest_target
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x = 1024
sc.render.resolution_y = 1280
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = os.path.join(OUT, "portrait.png")
bpy.ops.render.render(write_still=True)

# =========================================================
# 3) KEYART RENDER (island dojo + swarm)
# =========================================================
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

setup_world((0.03, 0.01, 0.06), 0.5)
# island base (inverted cone rock)
bpy.ops.mesh.primitive_cone_add(
    vertices=32, radius1=6.5, radius2=2.0, depth=7, location=(0, 0, -3.4)
)
rock = bpy.context.active_object
rock.data.materials.append(mat("Rock", (0.09, 0.07, 0.1), 0.1, 0.9))
# terraced platform
for i, (rad, z) in enumerate([(5.5, 0.0), (4.0, 0.9), (2.5, 1.8)]):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=rad, depth=0.8, location=(0, 0, z)
    )
    t = bpy.context.active_object
    t.name = f"Terrace{i}"
    t.data.materials.append(mat(f"Terrace{i}", (0.16, 0.08, 0.08), 0.3, 0.7))
# torii gate
CRIM = mat("ToriiRed", (0.5, 0.03, 0.02), 0.2, 0.4)
for sx in (-1, 1):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=0.35, depth=3.4, location=(1.6 * sx, 0, 3.6)
    )
    p = bpy.context.active_object
    p.data.materials.append(CRIM)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 5.45))
beam = bpy.context.active_object
beam.scale = (4.6, 0.4, 0.32)
beam.data.materials.append(CRIM)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 4.6))
beam2 = bpy.context.active_object
beam2.scale = (3.4, 0.35, 0.24)
beam2.data.materials.append(CRIM)
# swarm of ronins
random.seed(7)
for i in range(46):
    ang = random.uniform(0, 2 * math.pi)
    rr = random.uniform(0.6, 5.0)
    z = 0.4 + (0.9 if rr < 4.2 else 0) + (0.9 if rr < 2.5 else 0)
    build_ronin(
        f"Thrall{i}",
        (rr * math.cos(ang), rr * math.sin(ang), z),
        s=random.uniform(0.55, 0.75),
    )
# giant moon
bpy.ops.mesh.primitive_uv_sphere_add(
    segments=32, ring_count=16, radius=4, location=(6, 14, 8)
)
moon = bpy.context.active_object
moon.data.materials.append(
    mat("Moon", (0.9, 0.88, 0.8), 0.0, 0.9, emissive=(0.9, 0.88, 0.75), strength=0.9)
)
# lights
add_light("Storm", (0, -6, 9), 1500, 8, (0.7, 0.75, 1.0))
add_light("EmberKey", (3, -3, 5), 800, 4, (1.0, 0.5, 0.3))
add_light("TealGlow", (0, 0, 2.5), 400, 5, (0.1, 0.9, 0.8))
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 2.3))
k_target = bpy.context.active_object
bpy.ops.object.camera_add(location=(0, -16.5, 6.2), rotation=(math.radians(72), 0, 0))
cam = bpy.context.active_object
cam.data.lens = 30
bpy.context.scene.camera = cam
track = cam.constraints.new(type="TRACK_TO")
track.target = k_target
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x = 1920
sc.render.resolution_y = 1080
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = os.path.join(OUT, "keyart.png")
bpy.ops.render.render(write_still=True)
print("DONE", os.listdir(OUT))
