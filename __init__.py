bl_info = {
    "name": "Maze",
    "author": "The French Monkey",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Maze",
    "description": "Generate parametric maze",
    "category": "Add Mesh",
    "license": "GPL-3.0-or-later",
}

import bpy, bmesh, random
from math import cos, sin, tau
from collections import defaultdict

_SYNCING_FROM_OBJECT = False

# ------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------

def center_and_normalize(verts):
    if not verts:
        return verts
    min_x = min(v[0] for v in verts)
    max_x = max(v[0] for v in verts)
    min_y = min(v[1] for v in verts)
    max_y = max(v[1] for v in verts)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    size_x = (max_x - min_x)
    size_y = (max_y - min_y)
    mx = max(size_x, size_y)
    if mx <= 1e-12:
        return [(0.0, 0.0, v[2]) for v in verts]
    s = 1.0 / mx
    return [((x - cx) * s, (y - cy) * s, z) for (x, y, z) in verts]

def build_mesh_from_edges(verts, edges, wall_h):
    bm = bmesh.new()
    bm_verts = [bm.verts.new((x, y, 0.0)) for x, y, _ in verts]
    bm.verts.ensure_lookup_table()

    seen = set()
    for a, b in edges:
        if a == b: 
            continue
        k = (min(a, b), max(a, b))
        if k in seen: 
            continue
        seen.add(k)
        try:
            bm.edges.new((bm_verts[a], bm_verts[b]))
        except ValueError:
            pass

    if wall_h > 0.0 and bm.edges:
        ret = bmesh.ops.extrude_edge_only(bm, edges=bm.edges)
        ex_verts = [v for v in ret["geom"] if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, verts=ex_verts, vec=(0, 0, wall_h))

    mesh = bpy.data.meshes.new("MazeMesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh

def apply_or_update_solidify(obj, wall_size):
    if wall_size <= 0.0:
        for m in list(obj.modifiers):
            if m.type == "SOLIDIFY" and m.name.startswith("MG_Solidify"):
                obj.modifiers.remove(m)
        return
    mod = None
    for m in obj.modifiers:
        if m.type == "SOLIDIFY" and m.name.startswith("MG_Solidify"):
            mod = m
            break
    if not mod:
        mod = obj.modifiers.new("MG_Solidify", "SOLIDIFY")
    mod.name = "MG_Solidify"
    mod.thickness = wall_size
    mod.offset = 0.0
    mod.solidify_mode = "NON_MANIFOLD"

# ------------------------------------------------------------------------
# Graph Generation
# ------------------------------------------------------------------------

def spacing_to_stride(cs: float) -> int:
    cs = max(0.0, min(1.0, cs))
    return max(1, int(round(1 + cs * 8)))

def graph_rect_grid(w, h, iw=0.0, ih=0.0, corridor_spacing=0.0):
    w, h = max(1, int(w)), max(1, int(h))
    stride = spacing_to_stride(corridor_spacing)
    ox, oy = -(w - 1) * 0.5, -(h - 1) * 0.5
    verts, id_at, k = [], {}, 0
    for j in range(0, h, stride):
        for i in range(0, w, stride):
            x, y = ox + i, oy + j
            if (iw > 0.0 and ih > 0.0) and (-iw * 0.5 <= x <= iw * 0.5) and (-ih * 0.5 <= y <= ih * 0.5):
                continue
            id_at[(i, j)] = k
            verts.append((x, y, 0.0))
            k += 1
    graph = defaultdict(list)
    for j in range(0, h, stride):
        for i in range(0, w, stride):
            if (i, j) not in id_at:
                continue
            a = id_at[(i, j)]
            for di, dj in ((stride, 0), (-stride, 0), (0, stride), (0, -stride)):
                ni, nj = i + di, j + dj
                if (ni, nj) in id_at:
                    graph[a].append(id_at[(ni, nj)])
    verts = center_and_normalize(verts)
    return verts, graph, list(graph.keys())

def graph_circular_polar(outer_d, inner_d, corridor_spacing=0.0):
    r_out = max(0.25, float(outer_d) * 0.5)
    r_in = max(0.0, min(r_out - 0.1, float(inner_d) * 0.5))
    base_rings = max(2, int(round((r_out - r_in) * 3)))
    base_spokes = max(8, int(round(tau * r_out * 2)))
    stride = spacing_to_stride(corridor_spacing)
    ring_idx = list(range(0, base_rings + 1, stride))
    if ring_idx[-1] != base_rings:
        ring_idx.append(base_rings)
    spoke_idx = list(range(0, base_spokes, stride))
    if len(spoke_idx) < 3:
        spoke_idx = list(range(base_spokes))
    verts, id_at, k = [], {}, 0
    for ri in ring_idx:
        r = r_in + (r_out - r_in) * (ri / base_rings)
        for si in spoke_idx:
            a = tau * (si / base_spokes)
            verts.append((r * cos(a), r * sin(a), 0.0))
            id_at[(ri, si)] = k
            k += 1
    graph = defaultdict(list)
    for ri in ring_idx:
        for si in spoke_idx:
            a = id_at[(ri, si)]
            si_next = spoke_idx[(spoke_idx.index(si) + 1) % len(spoke_idx)]
            b = id_at[(ri, si_next)]
            graph[a].append(b); graph[b].append(a)
            if ri != ring_idx[-1]:
                ri_next = ring_idx[ring_idx.index(ri) + 1]
                c = id_at[(ri_next, si)]
                graph[a].append(c); graph[c].append(a)
    verts = center_and_normalize(verts)
    return verts, graph, list(graph.keys())

# ------------------------------------------------------------------------
# Maze Algorithms (DPS / Kruskal)
# ------------------------------------------------------------------------

class UnionFind:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}
    def find(self, n):
        if self.parent[n] != n:
            self.parent[n] = self.find(self.parent[n])
        return self.parent[n]
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: 
            return False
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True

def tree_dps(graph, nodes, rng):
    start = rng.choice(nodes)
    stack, visited, parent = [start], {start}, {start: None}
    while stack:
        v = stack[-1]
        unvis = [u for u in graph[v] if u not in visited]
        if unvis:
            w = rng.choice(unvis)
            parent[w] = v
            visited.add(w)
            stack.append(w)
        else:
            stack.pop()
    return parent

def tree_kruskal(graph, nodes, rng):
    edges = [(a, b) for a in nodes for b in graph[a] if a < b]
    rng.shuffle(edges)
    uf = UnionFind(nodes)
    parent = {n: None for n in nodes}
    for a, b in edges:
        if uf.union(a, b):
            parent[b] = a
    return parent

def build_maze_edges(graph, nodes, style, seed):
    rng = random.Random(int(seed))
    parent = tree_dps(graph, nodes, rng) if style == "DPS" else tree_kruskal(graph, nodes, rng)
    return [(p, v) for v, p in parent.items() if p is not None]

# ------------------------------------------------------------------------
# Rebuild / Generate
# ------------------------------------------------------------------------

def rebuild_maze(obj):
    if not obj or obj.type != "MESH": return False
    st, mt = obj["maze_style"], obj["maze_type"]
    cs, rw, rh, riw, rih, cod, cid = obj["corridor_spacing"], obj["rect_width"], obj["rect_height"], obj["rect_inner_w"], obj["rect_inner_h"], obj["circ_outer_d"], obj["circ_inner_d"]
    ms, wh, ws, sd = obj["maze_size"], obj["wall_height"], obj["wall_size"], obj["seed"]
    if mt == "RECT": verts, g, nodes = graph_rect_grid(rw, rh, riw, rih, cs)
    else: verts, g, nodes = graph_circular_polar(cod, cid, cs)
    edges = build_maze_edges(g, nodes, st, sd)
    verts_scaled = [(x * ms, y * ms, z) for (x, y, z) in verts]
    mesh = build_mesh_from_edges(verts_scaled, edges, wh)
    old = obj.data; obj.data = mesh
    if old and old.users == 0: bpy.data.meshes.remove(old)
    apply_or_update_solidify(obj, ws)
    return True

def generate_maze(context):
    p, obj = context.scene.mazegen_props, context.active_object
    keys = ("maze_style","maze_type","maze_size","corridor_spacing","wall_height","wall_size","seed",
            "rect_width","rect_height","rect_inner_w","rect_inner_h","circ_outer_d","circ_inner_d")
    if obj and obj.type == "MESH" and "maze_generator" in obj and obj.select_get():
        for k in keys: obj[k] = getattr(p, k)
        rebuild_maze(obj)
        obj.name = obj.data.name = f"Maze_{p.maze_type}_{p.maze_style}"
        return obj
    mesh = bpy.data.meshes.new("MazeMesh")
    new_obj = bpy.data.objects.new("Maze", mesh)
    context.collection.objects.link(new_obj)
    context.view_layer.objects.active = new_obj
    new_obj.select_set(True)
    for k in keys: new_obj[k] = getattr(p, k)
    new_obj["maze_generator"] = True
    rebuild_maze(new_obj)
    new_obj.name = new_obj.data.name = f"Maze_{p.maze_type}_{p.maze_style}"
    return new_obj

# ------------------------------------------------------------------------
# Live Update
# ------------------------------------------------------------------------

def update_maze(self, context):
    if globals().get("_SYNCING_FROM_OBJECT", False):
        return

    p = context.scene.mazegen_props
    if not p.live_update:
        return
    obj = context.active_object
    if not obj or obj.type != "MESH" or "maze_generator" not in obj:
        return
    for k in (
        "maze_style","maze_type","maze_size","corridor_spacing","wall_height","wall_size","seed",
        "rect_width","rect_height","rect_inner_w","rect_inner_h","circ_outer_d","circ_inner_d"
    ):
        obj[k] = getattr(p, k)
    rebuild_maze(obj)
    obj.name = obj.data.name = f"Maze_{p.maze_type}_{p.maze_style}"
    
def _sync_maze_to_props(obj, scene):
    if not obj or obj.type != "MESH" or "maze_generator" not in obj:
        return
    p = scene.mazegen_props
    try:
        globals()["_SYNCING_FROM_OBJECT"] = True 
        for k in (
            "maze_style","maze_type","maze_size","corridor_spacing","wall_height","wall_size","seed",
            "rect_width","rect_height","rect_inner_w","rect_inner_h","circ_outer_d","circ_inner_d"
        ):
            if k in obj:
                setattr(p, k, obj[k])
    finally:
        globals()["_SYNCING_FROM_OBJECT"] = False

# ------------------------------------------------------------------------
# Selection Change Listener 
# ------------------------------------------------------------------------

_MSGBUS_OWNER = object()

def _on_active_object_change():
    scene = bpy.context.scene
    obj = bpy.context.view_layer.objects.active
    _sync_maze_to_props(obj, scene)

# ------------------------------------------------------------------------
# Properties
# ------------------------------------------------------------------------

class MazeGenProperties(bpy.types.PropertyGroup):
    live_update: bpy.props.BoolProperty(name="Live Update", default=False)

    maze_size: bpy.props.FloatProperty(name="Maze Size", default=10.0, min=0.01, max=1000.0, subtype='FACTOR', update=update_maze)
    corridor_spacing: bpy.props.FloatProperty(name="Corridor Spacing", default=0.0, min=0.0, max=1.0, subtype='FACTOR', update=update_maze)
    wall_size: bpy.props.FloatProperty(name="Wall Size", default=0.12, min=0.0, max=1.0, subtype='FACTOR', update=update_maze)
    wall_height: bpy.props.FloatProperty(name="Wall Height", default=2.0, min=0.01, max=10.0, subtype='FACTOR', update=update_maze)
    seed: bpy.props.IntProperty(name="Seed", default=1, min=0, max=1000, subtype='FACTOR', update=update_maze)

    maze_type: bpy.props.EnumProperty(name="Shape", items=[("RECT", "Rectangular", ""), ("CIRC", "Circular", "")], default="RECT", update=update_maze)
    maze_style: bpy.props.EnumProperty(name="Style", items=[("DPS", "DPS", "Depth-First Search"), ("KRUSKAL", "Kruskal", "Kruskal's MST")], default="DPS", update=update_maze)

    rect_width: bpy.props.IntProperty(name="Width", default=20, min=1, max=500, subtype='FACTOR', update=update_maze)
    rect_height: bpy.props.IntProperty(name="Height", default=20, min=1, max=500, subtype='FACTOR', update=update_maze)
    rect_inner_w: bpy.props.FloatProperty(name="Inner Width", default=0.0, min=0.0, max=500.0, subtype='FACTOR', update=update_maze)
    rect_inner_h: bpy.props.FloatProperty(name="Inner Height", default=0.0, min=0.0, max=500.0, subtype='FACTOR', update=update_maze)

    circ_outer_d: bpy.props.FloatProperty(name="Outer Diameter", default=10.0, min=0.1, max=100.0, subtype='FACTOR', update=update_maze)
    circ_inner_d: bpy.props.FloatProperty(name="Inner Diameter", default=0.0, min=0.0, max=100.0, subtype='FACTOR', update=update_maze)

# ------------------------------------------------------------------------
# Operators
# ------------------------------------------------------------------------

class MAZEGEN_OT_generate(bpy.types.Operator):
    bl_idname = "maze.generate"
    bl_label = "Generate Maze"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        generate_maze(context)
        return {"FINISHED"}
    
class MAZEGEN_OT_reset(bpy.types.Operator):
    bl_idname = "maze.reset"
    bl_label = "Reset Sliders"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        p = context.scene.mazegen_props

        live = p.live_update

        try:
            globals()["_SYNCING_FROM_OBJECT"] = True 

            prop_defs = type(p).bl_rna.properties

            reset_keys = (
                "maze_size", "corridor_spacing", "wall_size", "wall_height", "seed",
                "rect_width", "rect_height", "rect_inner_w", "rect_inner_h",
                "circ_outer_d", "circ_inner_d"
            )

            for key in reset_keys:
                prop = prop_defs.get(key)
                if prop is not None:
                    default_val = prop.default
                    setattr(p, key, default_val)
        finally:
            globals()["_SYNCING_FROM_OBJECT"] = False

        if live:
            update_maze(p, context)

        self.report({'INFO'}, "Maze sliders reset to default values")
        return {'FINISHED'}
    
class MAZEGEN_OT_delete(bpy.types.Operator):
    bl_idname = "maze.delete"
    bl_label = "Delete Selected Maze"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != "MESH" or "maze_generator" not in obj:
            self.report({'WARNING'}, "No maze selected to delete")
            return {"CANCELLED"}

        bpy.data.objects.remove(obj, do_unlink=True)
        self.report({'INFO'}, "Selected maze deleted")
        return {"FINISHED"}

# ------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------

class VIEW3D_PT_maze_generator(bpy.types.Panel):
    bl_label = "Maze"
    bl_idname = "VIEW3D_PT_maze_generator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Maze"

    def draw(self, context):
        layout = self.layout
        p = context.scene.mazegen_props
        o = context.object

        row = layout.row(align=True)
        row.operator("maze.generate", icon="MESH_GRID")

        row = layout.row(align=True)
        row.prop(p, "live_update", toggle=True)
        row.operator("maze.reset", text="Reset")

        layout.separator(factor=0.5)

        layout.prop(p, "maze_type", expand=True)
        layout.prop(p, "maze_style", expand=True)
        layout.prop(p, "seed", slider=True)

        layout.separator(factor=0.5)

        layout.prop(p, "maze_size", slider=True)
        layout.prop(p, "corridor_spacing", slider=True)
        layout.prop(p, "wall_size", slider=True)
        layout.prop(p, "wall_height", slider=True)

        layout.separator(factor=0.5)

        if p.maze_type == "RECT":
            layout.prop(p, "rect_width", slider=True)
            layout.prop(p, "rect_height", slider=True)
            layout.prop(p, "rect_inner_w", slider=True)
            layout.prop(p, "rect_inner_h", slider=True)

        elif p.maze_type == "CIRC":
            layout.prop(p, "circ_outer_d", slider=True)
            layout.prop(p, "circ_inner_d", slider=True)

        layout.separator(factor=0.5)

        if o and o.type == "MESH" and "maze_generator" in o and o.select_get():
            row = layout.row(align=True)
            row.operator("maze.delete", icon="TRASH", text="Delete Maze")
            layout.label(text=f"Editing Maze: {o.name}", icon="EDITMODE_HLT")
        else:
            layout.label(text="No generated maze selected", icon="INFO")

# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (
    MazeGenProperties,
    MAZEGEN_OT_generate,
    MAZEGEN_OT_reset,
    MAZEGEN_OT_delete,
    VIEW3D_PT_maze_generator, 
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.mazegen_props = bpy.props.PointerProperty(type=MazeGenProperties)

    bpy.msgbus.subscribe_rna(
        key=(bpy.types.LayerObjects, "active"),
        owner=_MSGBUS_OWNER,
        args=(),
        notify=_on_active_object_change,
    )

def unregister():

    bpy.msgbus.clear_by_owner(_MSGBUS_OWNER)

    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.mazegen_props

if __name__ == "__main__":
    register()
