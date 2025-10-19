SHOW_SEA = False

import pygame, math, random
BASE_PRE_RIVER_STATE = None
SHOW_WATER = True
from roads import RoadSystem
import roads
from buildings import BuildingSystem
from water import WaterSystem
import water as water_mod
from sea import SeaSystem
from decorations import DecorationsSystem
from ui import SidebarUI, Slider
import bridges
from facilities import FacilitiesSystem, FacilitiesUI
from bridges import commit_preview_bridge, start_bridge_preview_blend as start_bridge_preview, update_bridge_preview_blend, cancel_preview_bridge
import walls

pygame.init(); pygame.font.init()

SCALE = 1.5
def Si(x): return int(round(x*SCALE))

# Reserve a fixed amount of horizontal space on the left side of the
# application window for the facilities control panel.  This panel
# contains the checkboxes and sliders used to toggle and adjust the
# various civic structures.  Moving the panel outside of the map
# ensures that it no longer obscures the map itself.  The width is
# specified in scaled units so that it grows proportionally when
# adjusting the global SCALE.  All screen‑space conversions account
# for this offset.
LEFT_PANEL_WIDTH = Si(220)

MAP_WIDTH    = Si(936)
PANEL_WIDTH  = Si(320)
# Total window width is the sum of the left facilities panel, the map
# region and the right‑hand parameter sliders panel.
WIDTH        = LEFT_PANEL_WIDTH + MAP_WIDTH + PANEL_WIDTH
HEIGHT       = Si(664)

# keep world like before; this does NOT change road growth behavior
WORLD_X = -MAP_WIDTH // 2; WORLD_Y = -HEIGHT // 2
WORLD_WIDTH, WORLD_HEIGHT = MAP_WIDTH*4, HEIGHT*4  # big world to reveal by dragging bounds

cam_zoom = 1.0
cam_offset = [0.0, 0.0]

GEN_BOUNDS = [0.0, 0.0, float(MAP_WIDTH), float(HEIGHT)]  # x0,y0,x1,y1
_drag_saved_overrides = None

def world_to_screen(pt):
    """Convert a world coordinate to screen space.

    The x coordinate is shifted to the right by the width of the
    facilities panel so that the map region begins immediately to
    the right of the panel.  Scaling by cam_zoom is applied
    uniformly to both axes.
    """
    return (
        LEFT_PANEL_WIDTH + (pt[0] - cam_offset[0]) * cam_zoom,
        (pt[1] - cam_offset[1]) * cam_zoom,
    )

def screen_to_world(pt):
    """Convert a screen coordinate back to world space.

    The x coordinate is adjusted by subtracting the facilities panel
    width before unapplying the current zoom.  This maps the
    screen's map region back to world coordinates.  Points in the
    facilities panel (to the left of the map) will yield world
    coordinates extending into the negative region, which is
    acceptable for panning and other operations.
    """
    return (
        (pt[0] - LEFT_PANEL_WIDTH) / cam_zoom + cam_offset[0],
        pt[1] / cam_zoom + cam_offset[1],
    )

PARAMS = {
    "PRIMARY_SEGMENT_LENGTH": 137,
    "SECONDARY_SEGMENT_LENGTH": 56,
    "TERTIARY_SEGMENT_LENGTH": 50,
    "ALLEY_SEGMENT_LENGTH": Si(33),
    "CURVE": {0:39.70, 1:12.30, 2:7.80, 3:5.76},
    "ROAD_WIDTH": {0:Si(10), 1:Si(8), 2:Si(6), 3:Si(3)},
    "MIN_PARALLEL_SPACING": Si(18),
    "MIN_PARALLEL_ANGLE_DEG": 45,
    "SNAP_RADIUS_NODE": Si(19),
    "SNAP_RADIUS_SEG": Si(18),
    "LENGTH_RANGES": {0:(Si(30),Si(50)),1:(Si(16),Si(30)),2:(Si(10),Si(20)),3:(Si(10),Si(20))},
    "BRANCH_BASE_ANGLE": 90, "BRANCH_JITTER": 2, "BIDIR_BRANCH_PROB": 1.0,
    "BRANCH_DELAY": {0:0,1:1,2:2,3:3}, "BRANCH_PROB": {0:1.0,1:0.9,2:0.7},
    "HOUSE_SCALE": 0.10, "HOUSE_FRONTAGE_BASE": Si(40), "HOUSE_DEPTH_BASE": Si(80),
    "HOUSE_OFFSET_BASE": Si(1), "HOUSE_STEP_MIN": Si(2),
    "FACILITY_SCALE": 1.0,
    "LEVEL_REMAP": {0:0, 1:0, 2:1, 3:2},
    "QUADTREE_MAX_OBJECTS": 32,
    "QUADTREE_MAX_LEVELS": 10,
    "SEGMENT_COUNT_LIMIT": 5000,
    "STEP_LIMIT": 2000,
}

BG_COLOR   = (247, 246, 242)
GRID_COLOR = (232, 232, 232)
ROAD_COLOR = (164, 168, 176)

ADDRESS_NAMES = ['Main Street','Broadway','Oak Lane','Maple Avenue','Elm Street','Pine Road','Cedar Court','Willow Way','River Drive','Hillcrest']

roads = RoadSystem(map_size=(MAP_WIDTH, HEIGHT), world_bounds=(WORLD_X, WORLD_Y, WORLD_WIDTH, WORLD_HEIGHT), params=PARAMS, address_names=ADDRESS_NAMES)
roads.set_generation_bounds(tuple(GEN_BOUNDS))
builds = BuildingSystem(map_size=(MAP_WIDTH, HEIGHT), params=PARAMS)
water  = WaterSystem(map_size=(MAP_WIDTH, HEIGHT))
sea = SeaSystem((MAP_WIDTH, HEIGHT))

decor = DecorationsSystem((MAP_WIDTH, HEIGHT))
# Attach to systems for optional hooks
water.decor = decor
sea.decor = decor
SHOW_ROAD_NAMES = False

# ---- undo stack ----
UNDO_STACK=[]; MAX_UNDO=50
BASE_PRE_RIVER_STATE = None
def snapshot_state():
    state = {
        "roads": roads.serialize_state(),
        "water": water.serialize_state(),
        "sea": sea.serialize_state(),
        "buildings": builds.serialize_state(),
        "GEN_BOUNDS": list(GEN_BOUNDS),
        "SHOW_ROAD_NAMES": SHOW_ROAD_NAMES,
        "cam_offset": tuple(cam_offset),
        "cam_zoom": cam_zoom,
    }
    UNDO_STACK.append(state)
    if len(UNDO_STACK) > MAX_UNDO:
        del UNDO_STACK[0]

def restore_state_pop():
    global SHOW_ROAD_NAMES, cam_zoom
    if not UNDO_STACK: return
    state = UNDO_STACK.pop()
    roads.restore_state(state["roads"])
    water.restore_state(state["water"])
    sea.restore_state(state.get("sea", {}))
    builds.restore_state(state["buildings"])
    GEN_BOUNDS[:] = state.get("GEN_BOUNDS", GEN_BOUNDS)
    roads.set_generation_bounds(tuple(GEN_BOUNDS))
    SHOW_ROAD_NAMES = state.get("SHOW_ROAD_NAMES", False)
    cam_offset[:] = list(state.get("cam_offset", tuple(cam_offset)))
    cam_zoom = state.get("cam_zoom", cam_zoom)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("city weaver — vector (manual spacing override only)")
font = pygame.font.Font(None, Si(20))
# Position the sidebar to the right of the map region.  Its x
# coordinate is offset by the facilities panel width plus the map
# width so that it does not overlap either the map or the left panel.
sidebar = SidebarUI((LEFT_PANEL_WIDTH + MAP_WIDTH, 0, PANEL_WIDTH, HEIGHT)); sidebar.set_font(font)

# Instantiate the facilities system and its UI panel.  The facilities system
# manages placement and rendering of civic structures (schools, banks, squares,
# etc.) on top of the house and road network.  The UI exposes a checkbox
# and slider for each facility type allowing users to toggle its
# generation and adjust how many appear.  The panel lives on the left
# side of the window and shares the global font so text remains
# consistent across UI elements.
facilities = FacilitiesSystem(map_size=(MAP_WIDTH, HEIGHT), params=PARAMS)
facilities_ui = FacilitiesUI((0, 0, LEFT_PANEL_WIDTH, HEIGHT), facility_names=list(facilities.specs), specs=facilities.specs)
facilities_ui.set_font(font)
facilities.set_font(font)

# ---- sliders (scrollable) ----
def add_slider(label, rng, key_path, step, fmt="{:.2f}"):
    def get_ref():
        d = PARAMS
        for k in key_path[:-1]: d = d[k]
        return d, key_path[-1]
    d,k = get_ref(); cur = d[k]
    def on_change(v):
        d,k = get_ref()
        d[k] = type(cur)(v) if not isinstance(cur, tuple) else v
    sidebar.add_slider(Slider(label, rng[0], rng[1], cur, step=step, fmt=fmt, on_change=on_change))


# ---- system sliders (appear before other sliders) ----
def add_int_slider(label, rng, key_path, step=1, fmt="{:.0f}", on_after=None):
    def get_ref():
        d = PARAMS
        for k in key_path[:-1]: d = d[k]
        return d, key_path[-1]
    d,k = get_ref(); cur = int(d[k])
    def on_change(v):
        d,k = get_ref()
        d[k] = int(v)
        if on_after:
            try: on_after(int(v))
            except Exception: pass
    sidebar.add_slider(Slider(label, rng[0], rng[1], cur, step=step, fmt=fmt, on_change=on_change))

def _set_quadtree_max_objects(v):
    try:
        import quadtree as _qt
        _qt.QT_MAX_OBJECTS = int(v)
    except Exception:
        pass

def _set_quadtree_max_levels(v):
    try:
        import quadtree as _qt
        _qt.QT_MAX_LEVELS = int(v)
    except Exception:
        pass

add_int_slider("Quadtree Max Objects", (4, 128), ("QUADTREE_MAX_OBJECTS",), 1, "{:.0f}", _set_quadtree_max_objects)
add_int_slider("Quadtree Max Levels", (4, 16), ("QUADTREE_MAX_LEVELS",), 1, "{:.0f}", _set_quadtree_max_levels)
add_int_slider("Segment Count Limit", (100, 50000), ("SEGMENT_COUNT_LIMIT",), 10, "{:.0f}")
add_int_slider("Step Limit", (100, 20000), ("STEP_LIMIT",), 50, "{:.0f}")

add_slider("Primary curve", (0.0, 40.0), ("CURVE",0), 0.1)
add_slider("Secondary curve", (0.0, 20.0), ("CURVE",1), 0.1)
add_slider("Tertiary curve", (0.0, 15.0), ("CURVE",2), 0.1)
add_slider("Primary seg len", (Si(10), Si(120)), ("PRIMARY_SEGMENT_LENGTH",), Si(1), "{:.0f}")
add_slider("Secondary seg len", (Si(8), Si(80)), ("SECONDARY_SEGMENT_LENGTH",), Si(1), "{:.0f}")
add_slider("Tertiary seg len", (Si(6), Si(80)), ("TERTIARY_SEGMENT_LENGTH",), Si(1), "{:.0f}")
add_slider("Primary width", (Si(2), Si(20)), ("ROAD_WIDTH",0), 1, "{:.0f}")
add_slider("Secondary width", (Si(2), Si(16)), ("ROAD_WIDTH",1), 1, "{:.0f}")
add_slider("Tertiary width", (Si(2), Si(14)), ("ROAD_WIDTH",2), 1, "{:.0f}")
add_slider("Min parallel spacing", (Si(4), Si(60)), ("MIN_PARALLEL_SPACING",), 1, "{:.0f}")
add_slider("Min parallel angle", (0, 90), ("MIN_PARALLEL_ANGLE_DEG",), 1, "{:.0f}")
add_slider("Snap radius (node)", (Si(4), Si(60)), ("SNAP_RADIUS_NODE",), 1, "{:.0f}")
add_slider("Snap radius (seg)", (Si(4), Si(60)), ("SNAP_RADIUS_SEG",), 1, "{:.0f}")
add_slider("Branch angle", (0, 180), ("BRANCH_BASE_ANGLE",), 1, "{:.0f}")
add_slider("Branch jitter", (0, 30), ("BRANCH_JITTER",), 1, "{:.0f}")
add_slider("Bidirectional prob", (0.0, 1.0), ("BIDIR_BRANCH_PROB",), 0.01)
add_slider("Branch prob L1", (0.0, 1.0), ("BRANCH_PROB",1), 0.01)
add_slider("Branch prob L2", (0.0, 1.0), ("BRANCH_PROB",2), 0.01)
add_slider("House scale", (0.05, 0.50), ("HOUSE_SCALE",), 0.01)
add_slider("House frontage", (Si(10), Si(120)), ("HOUSE_FRONTAGE_BASE",), 1, "{:.0f}")
add_slider("House depth", (Si(10), Si(160)), ("HOUSE_DEPTH_BASE",), 1, "{:.0f}")
add_slider("House offset", (0, Si(20)), ("HOUSE_OFFSET_BASE",), 1, "{:.0f}")
add_slider("Plot gap", (0, Si(20)), ("HOUSE_STEP_MIN",), 1, "{:.0f}")
def _on_facility_scale_change(v):
    PARAMS["FACILITY_SCALE"] = float(v)
    # Regenerate facilities immediately so changes are visible
    try:
        enabled = [name for name, on in facilities_ui.enabled.items() if on]
        facilities.generate(roads, builds, water, enabled_names=enabled)
        _cleanup_buildings_vs_squares()
    except Exception:
        pass
sidebar.add_slider(Slider("Facility scale", 0.5, 2.0, PARAMS.get("FACILITY_SCALE", 1.0), step=0.1, fmt="{:.2f}", on_change=_on_facility_scale_change))

def _normalize_bounds():
    x0,y0,x1,y1 = GEN_BOUNDS
    if x0>x1: x0,x1 = x1,x0
    if y0>y1: y0,y1 = y1,y0
    GEN_BOUNDS[0],GEN_BOUNDS[1],GEN_BOUNDS[2],GEN_BOUNDS[3] = x0,y0,x1,y1

def draw_map_bounds():
    _normalize_bounds()
    sx0, sy0 = world_to_screen((GEN_BOUNDS[0], GEN_BOUNDS[1]))
    sx1, sy1 = world_to_screen((GEN_BOUNDS[2], GEN_BOUNDS[3]))
    pygame.draw.rect(screen, (60,60,60), pygame.Rect(min(sx0,sx1), min(sy0,sy1), abs(sx1-sx0), abs(sy1-sy0)), max(1, int(1*cam_zoom)))
    handle = max(4, int(8*cam_zoom))
    for px, py in [(sx0,sy0),(sx1,sy0),(sx1,sy1),(sx0,sy1)]:
        pygame.draw.rect(screen, (60,60,60), pygame.Rect(int(px-handle/2), int(py-handle/2), handle, handle), 2)


# ----------------------- road regeneration helpers -----------------------
# These helper functions implement regeneration of the road network so
# that primary highways follow the orientation of bridge decks and the
# remainder of the road network branches off accordingly.  They are
# invoked after river and bridge generation to rebuild `roads`.

def _clip_line_to_gen(a, b):
    """
    Clip the infinite line segment defined by endpoints `a` and `b` to the
    current generation bounds (GEN_BOUNDS).  Returns a tuple of
    clipped endpoints (A,B) or None if the line does not intersect
    the generation rectangle.  The clipping uses the helper
    `_clip_segment_to_rect` from the `roads` module and returns
    world-space coordinates.
    """
    # Compute the axis-aligned generation rectangle
    rect = (GEN_BOUNDS[0], GEN_BOUNDS[1], GEN_BOUNDS[2], GEN_BOUNDS[3])
    # Use roads module's clipping function
    try:
        clipped = roads._clip_segment_to_rect(a, b, rect)
    except Exception:
        clipped = None
    return clipped


def rebuild_roads_from_bridges(existing_roads, water, sample_spacing=None, seed_every=None):
    """
    Rebuild an entire `RoadSystem` so that highways follow the centerlines
    of the current bridge overlay and the remainder of the road network
    grows from those highways.  A new `RoadSystem` is returned and must
    be assigned back to the global `roads` variable by the caller.

    Parameters
    ----------
    existing_roads : RoadSystem
        The current road system (used only to access parameters and map
        dimensions).  Its graph is not modified.
    water : WaterSystem
        The current water system, used for water-aware segment clipping.
    sample_spacing : float, optional
        Distance between highway seed points.  Defaults to three times
        the primary segment length.
    seed_every : unused (reserved for future).  If provided it will
        override the number of seeds per highway direction.

    Returns
    -------
    RoadSystem
        A freshly constructed `RoadSystem` containing only the new
        highways and seeds for branching growth.
    """
    # Retrieve configuration from the existing road system
    map_size = (MAP_WIDTH, HEIGHT)
    world_bounds = (WORLD_X, WORLD_Y, WORLD_WIDTH, WORLD_HEIGHT)
    params = PARAMS
    address_names = ADDRESS_NAMES

    # Create a new road system with the same parameters
    new_roads = RoadSystem(map_size=map_size, world_bounds=world_bounds, params=params, address_names=address_names)
    new_roads.set_generation_bounds(tuple(GEN_BOUNDS))

    # Determine seed spacing based on primary segment length if not specified
    primary_len = params.get("PRIMARY_SEGMENT_LENGTH", 40)
    if sample_spacing is None:
        sample_spacing = max(primary_len * 3, 100)

    # Fetch the bridge deck geometries from the overlay; if none, return an empty road system
    deck_geoms = bridges.get_bridge_geoms() if hasattr(bridges, 'get_bridge_geoms') else []
    if not deck_geoms:
        return new_roads

    import math, random
    # For each deck, extend a long highway through its midpoint along its axis
    for (A, B, lvl) in deck_geoms:
        # Compute unit vector along the deck
        vx, vy = (B[0] - A[0], B[1] - A[1])
        L = (vx * vx + vy * vy) ** 0.5 or 1.0
        ux, uy = vx / L, vy / L
        # Midpoint for symmetric extension
        mx, my = ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)
        # Create a very long segment along the deck direction
        far = max(GEN_BOUNDS[2] - GEN_BOUNDS[0], GEN_BOUNDS[3] - GEN_BOUNDS[1]) * 2.0
        raw_a = (mx - ux * far, my - uy * far)
        raw_b = (mx + ux * far, my + uy * far)
        clipped = _clip_line_to_gen(raw_a, raw_b)
        if not clipped:
            continue
        HA, HB = clipped
        # Add the highway segment as a manual level-1 segment.  Use half the min parallel spacing to avoid
        # immediate rejection from near_parallel_too_close.
        try:
            mps = params.get("MIN_PARALLEL_SPACING", 0)
            override = float(mps) * 0.5
        except Exception:
            override = None
        # For highway segments along bridge decks we deliberately pass
        # ``water=None`` so that the manual segment crosses the river
        # without being truncated by the river clipping logic.  The
        # cross-river behaviour of these highways is governed solely
        # by the bridge geometry.
        new_roads.add_manual_segment(HA, HB, level=1, water=None, min_spacing_override=override)

        # Determine normals (perpendicular) to seed secondary branches
        # Left and right normal vectors (90 deg rotation)
        left_angle = (math.degrees(math.atan2(-uy, -ux)) + 90.0) % 360.0
        right_angle = (left_angle + 180.0) % 360.0

        # Compute the number of sample points along the highway
        seg_len = math.hypot(HB[0] - HA[0], HB[1] - HA[1]) or 1.0
        if sample_spacing <= 0:
            n_pts = 0
        else:
            n_pts = max(0, int(seg_len // sample_spacing))

        # Place seeds along the highway excluding endpoints
        for i in range(1, n_pts + 1):
            t = i / float(n_pts + 1)
            px = HA[0] + (HB[0] - HA[0]) * t
            py = HA[1] + (HB[1] - HA[1]) * t
            # Optionally jitter heading a little to avoid unnatural symmetry
            jitter = 0.0  # could be random.uniform(-5.0,5.0) for organic feel
            new_roads.add_active_seed((px, py), left_angle + jitter, level=2)
            new_roads.add_active_seed((px, py), right_angle + jitter, level=2)

    return new_roads

def reset_everything():
    roads.nodes.clear(); roads.segments.clear(); roads.road_graph.clear()
    roads.qt = roads.qt.__class__((WORLD_X,WORLD_Y,WORLD_WIDTH,HEIGHT*2))
    roads.active.clear(); roads.segment_names.clear()
    roads.river_bridge_segments.clear(); roads.river_skip_segments.clear(); roads.sea_skip_segments.clear()
    roads.river_cross_count=0
    builds.reset(); water.reset(); sea.reset(); sea.reset()
    roads.set_generation_bounds(tuple(GEN_BOUNDS))
    # Clear all existing facilities when the city is reset.  Facilities
    # will regenerate automatically when houses are rebuilt via the
    # [B] key or when the user toggles them in the facilities panel.
    try:
        facilities.facilities = []
        facilities.fac_qt = None
    except Exception:
        pass

panning=False; pan_start_screen=None; pan_start_offset=None
placing_bridge=False
dbg_prev_visible=False  # debug: track first-time preview acquisition
dragging_handle=None; dragging_road=False; road_drag_start=None; road_drag_curr=None

reset_text = font.render("Reset", True, (0,0,0))
pad_x, pad_y = Si(8), Si(4)
button_w, button_h = reset_text.get_width()+pad_x*2, reset_text.get_height()+pad_y*2
reset_btn_rect = pygame.Rect(LEFT_PANEL_WIDTH + MAP_WIDTH - button_w - Si(10), Si(10), button_w, button_h)
reset_bg_color = (230,230,230)

# ---------- Export Button Setup ----------
# Add a second button labelled "Export" to the top right of the map.  The
# export button sits immediately to the left of the existing reset
# button.  Clicking this button will save the current view of the map
# (excluding UI panels) to both a PNG and a PDF file in the working
# directory.  The PDF contains invisible link annotations over each
# facility footprint; when clicked these annotations reveal the
# facility name.  See the export_city() function below for details.
export_text = font.render("Export", True, (0,0,0))
# Use the same padding as the reset button so both buttons share the
# same height.  Compute the total width of the export button from
# the rendered text plus horizontal padding.  Position it with a
# 10‑pixel gap to the left of the reset button.  The y coordinate and
# height match the reset button.
export_btn_w = export_text.get_width() + pad_x*2
export_btn_h = button_h
export_btn_rect = pygame.Rect(reset_btn_rect.x - export_btn_w - Si(10), reset_btn_rect.y, export_btn_w, export_btn_h)
export_bg_color = (230,230,230)

def _escape_pdf_string(s: str) -> str:
    """
    Escape parentheses and backslashes in a string for safe insertion
    into a PDF literal.  PDF uses parentheses to delimit strings, so
    these characters must be prefixed with a backslash.

    Parameters
    ----------
    s : str
        The raw text to escape.

    Returns
    -------
    str
        The escaped string.
    """
    return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

def _write_pdf_with_annotations(img_path: str, pdf_path: str, annots: list[tuple[float,float,float,float,str]]) -> None:
    """
    Write a simple single‑page PDF containing the supplied image and
    rectangular link annotations.  Each annotation is defined by a
    bounding box and a name; the annotation's /Contents field is set
    to the facility name so PDF viewers can reveal it on click or
    hover.  The PDF is written manually without any third‑party
    dependencies.  Image data are compressed using zlib and stored
    as a FlateDecode XObject.  Coordinates are in pixel units; the
    page size is equal to the image size.

    Parameters
    ----------
    img_path : str
        Path to the PNG image to embed.
    pdf_path : str
        Destination path for the generated PDF file.
    annots : list of tuples
        A list of annotation definitions.  Each entry is a
        four‑element tuple (llx, lly, urx, ury, name) specifying
        the lower‑left and upper‑right corners of the annotation
        rectangle in image pixel coordinates and the associated
        facility name.
    """
    import zlib
    from PIL import Image

    # Load the image and convert to raw RGB bytes
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    raw = img.tobytes()
    # Compress the RGB data with zlib.  A PNG predictor improves
    # compression for continuous tone images by computing row
    # differences.  See the PDF specification for details.
    compressed = zlib.compress(raw)
    # Prepare the image XObject.  Use predictor parameters to inform
    # the PDF reader how to unpack the compressed RGB data.
    img_stream = (
        f"<<\n"
        f"/Type /XObject\n"
        f"/Subtype /Image\n"
        f"/Width {width}\n"
        f"/Height {height}\n"
        f"/ColorSpace /DeviceRGB\n"
        f"/BitsPerComponent 8\n"
        f"/Filter /FlateDecode\n"
        f"/DecodeParms << /Predictor 15 /Colors 3 /BitsPerComponent 8 /Columns {width} >>\n"
        f"/Length {len(compressed)}\n"
        f">>\n"
        f"stream\n".encode('utf-8') + compressed + b"\nendstream"
    )
    # Prepare the content stream that draws the image.  The sequence
    # saves graphics state (q), sets up a transformation that scales
    # the unit square to the full page size (width height) and draws
    # the image by name (Im0).  Finally, restore graphics state (Q).
    content_commands = f"q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n"
    compressed_content = zlib.compress(content_commands.encode('utf-8'))
    content_stream = (
        f"<<\n"
        f"/Length {len(compressed_content)}\n"
        f"/Filter /FlateDecode\n"
        f">>\n"
        f"stream\n".encode('utf-8') + compressed_content + b"\nendstream"
    )
    # Allocate objects in the following order:
    # 1. Catalog (root)
    # 2. Pages
    # 3. Page
    # 4. Image XObject
    # 5. Content stream
    # 6..(6+n-1). Annotations
    objects: list[bytes] = []
    # We will fill annotation objects later; keep placeholders
    n_ann = len(annots)
    # Placeholder for object indices: we'll append content in the order
    # defined above.
    # Object 4: image XObject
    objects.append(img_stream)
    # Object 5: content stream
    objects.append(content_stream)
    # Objects 6..: annotation dictionaries
    for (llx, lly, urx, ury, name) in annots:
        # Escape the name for PDF
        esc_name = _escape_pdf_string(name)
        annot_dict = (
            f"<<\n"
            f"/Type /Annot\n"
            f"/Subtype /Link\n"
            f"/Rect [{llx:.2f} {lly:.2f} {urx:.2f} {ury:.2f}]\n"
            f"/Border [0 0 0]\n"
            f"/Contents ({esc_name})\n"
            f">>"
        ).encode('utf-8')
        objects.append(annot_dict)
    # Prepare the page dictionary; annotation references computed below
    # Note: object numbering offsets: catalog (1), pages (2), page (3),
    # image (4), content (5), annotations start at 6.
    # Build annotation references list
    if n_ann > 0:
        annot_refs = ' '.join([f"{6 + i} 0 R" for i in range(n_ann)])
        annots_entry = f"/Annots [{annot_refs}]\n"
    else:
        annots_entry = ''
    page_dict = (
        f"<<\n"
        f"/Type /Page\n"
        f"/Parent 2 0 R\n"
        f"/MediaBox [0 0 {width} {height}]\n"
        f"/Resources << /XObject << /Im0 4 0 R >> >>\n"
        f"/Contents 5 0 R\n"
        f"{annots_entry}"
        f">>"
    ).encode('utf-8')
    # Pages dictionary referencing the single page
    pages_dict = (
        f"<<\n"
        f"/Type /Pages\n"
        f"/Kids [3 0 R]\n"
        f"/Count 1\n"
        f">>"
    ).encode('utf-8')
    # Catalog dictionary referencing pages
    catalog_dict = (
        f"<<\n"
        f"/Type /Catalog\n"
        f"/Pages 2 0 R\n"
        f">>"
    ).encode('utf-8')
    # Prepend catalog, pages and page to the objects list in reverse
    # order to maintain numbering alignment.  Object indices will be:
    # 1: catalog, 2: pages, 3: page, 4: image, 5: content,
    # 6..: annotations
    objects = [catalog_dict, pages_dict, page_dict] + objects
    # Begin assembling the PDF file
    parts: list[bytes] = []
    parts.append(b"%PDF-1.4\n")
    # offsets record the byte offset of each object in the output
    offsets = []
    # Write each object
    for obj_index, content in enumerate(objects, start=1):
        offsets.append(sum(len(p) for p in parts))
        parts.append(f"{obj_index} 0 obj\n".encode('utf-8'))
        parts.append(content)
        parts.append(b"\nendobj\n")
    # Cross‑reference table
    xref_start = sum(len(p) for p in parts)
    parts.append(b"xref\n")
    parts.append(f"0 {len(objects)+1}\n".encode('utf-8'))
    parts.append(b"0000000000 65535 f \n")
    for offset in offsets:
        parts.append(f"{offset:010d} 00000 n \n".encode('utf-8'))
    # Trailer with root (catalog)
    parts.append(b"trailer\n")
    parts.append(f"<< /Size {len(objects)+1} /Root 1 0 R >>\n".encode('utf-8'))
    parts.append(b"startxref\n")
    parts.append(f"{xref_start}\n".encode('utf-8'))
    parts.append(b"%%EOF")
    # Write to disk
    with open(pdf_path, 'wb') as f:
        for part in parts:
            f.write(part)

# -----------------------------------------------------------------------------
# Additional helper: write a PDF from raw RGB bytes
# -----------------------------------------------------------------------------
def _write_pdf_with_annotations_data(rgb_bytes: bytes, size: tuple[int,int], pdf_path: str, annots: list[tuple[float,float,float,float,str]]) -> None:
    """
    Like `_write_pdf_with_annotations` but accepts raw RGB pixel data and image
    dimensions instead of loading from disk.  This avoids a dependency on
    Pillow.  The pixel buffer must contain unfiltered RGB bytes in row‑major
    order (top‑left first).  The PDF is written directly to `pdf_path`.

    Parameters
    ----------
    rgb_bytes : bytes
        Raw RGB pixel data (3 bytes per pixel) arranged by row from
        top‑left to bottom‑right.
    size : (width, height)
        Dimensions of the image in pixels.
    pdf_path : str
        Destination path for the generated PDF.
    annots : list of tuples
        Annotation definitions: (llx, lly, urx, ury, name) in image
        coordinates where (0,0) is the top‑left corner.  These will be
        converted to PDF coordinates internally.
    """
    import zlib
    width, height = size
    # Compress the raw data.  Because we do not apply PNG predictors,
    # the PDF image dictionary omits the DecodeParms entry.
    compressed = zlib.compress(rgb_bytes)
    # Build the image XObject stream
    img_stream = (
        f"<<\n"
        f"/Type /XObject\n"
        f"/Subtype /Image\n"
        f"/Width {width}\n"
        f"/Height {height}\n"
        f"/ColorSpace /DeviceRGB\n"
        f"/BitsPerComponent 8\n"
        f"/Filter /FlateDecode\n"
        f"/Length {len(compressed)}\n"
        f">>\n".encode('utf-8') + b"stream\n" + compressed + b"\nendstream"
    )
    # Prepare the content stream to draw the image
    import zlib as _zlib
    content_commands = f"q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n"
    compressed_content = _zlib.compress(content_commands.encode('utf-8'))
    content_stream = (
        f"<<\n"
        f"/Length {len(compressed_content)}\n"
        f"/Filter /FlateDecode\n"
        f">>\n".encode('utf-8') + b"stream\n" + compressed_content + b"\nendstream"
    )
    # Build annotation objects
    objects: list[bytes] = []
    for (llx, lly, urx, ury, name) in annots:
        esc_name = _escape_pdf_string(name)
        annot_dict = (
            f"<<\n"
            f"/Type /Annot\n"
            f"/Subtype /Link\n"
            f"/Rect [{llx:.2f} {lly:.2f} {urx:.2f} {ury:.2f}]\n"
            f"/Border [0 0 0]\n"
            f"/Contents ({esc_name})\n"
            f">>"
        ).encode('utf-8')
        objects.append(annot_dict)
    # Page dictionary referencing annotations (will be inserted later)
    n_ann = len(objects)
    if n_ann > 0:
        annot_refs = ' '.join([f"{6 + i} 0 R" for i in range(n_ann)])
        annots_entry = f"/Annots [{annot_refs}]\n"
    else:
        annots_entry = ''
    page_dict = (
        f"<<\n"
        f"/Type /Page\n"
        f"/Parent 2 0 R\n"
        f"/MediaBox [0 0 {width} {height}]\n"
        f"/Resources << /XObject << /Im0 4 0 R >> >>\n"
        f"/Contents 5 0 R\n"
        f"{annots_entry}"
        f">>"
    ).encode('utf-8')
    pages_dict = (
        f"<<\n"
        f"/Type /Pages\n"
        f"/Kids [3 0 R]\n"
        f"/Count 1\n"
        f">>"
    ).encode('utf-8')
    catalog_dict = (
        f"<<\n"
        f"/Type /Catalog\n"
        f"/Pages 2 0 R\n"
        f">>"
    ).encode('utf-8')
    # Assemble objects in order: catalog, pages, page, image, content, annotations
    pdf_objects: list[bytes] = [catalog_dict, pages_dict, page_dict, img_stream, content_stream] + objects
    # Build PDF
    parts: list[bytes] = []
    parts.append(b"%PDF-1.4\n")
    offsets = []
    for obj_index, content in enumerate(pdf_objects, start=1):
        offsets.append(sum(len(p) for p in parts))
        parts.append(f"{obj_index} 0 obj\n".encode('utf-8'))
        parts.append(content)
        parts.append(b"\nendobj\n")
    xref_start = sum(len(p) for p in parts)
    parts.append(b"xref\n")
    parts.append(f"0 {len(pdf_objects)+1}\n".encode('utf-8'))
    parts.append(b"0000000000 65535 f \n")
    for off in offsets:
        parts.append(f"{off:010d} 00000 n \n".encode('utf-8'))
    parts.append(b"trailer\n")
    parts.append(f"<< /Size {len(pdf_objects)+1} /Root 1 0 R >>\n".encode('utf-8'))
    parts.append(b"startxref\n")
    parts.append(f"{xref_start}\n".encode('utf-8'))
    parts.append(b"%%EOF")
    with open(pdf_path, 'wb') as f:
        for p in parts:
            f.write(p)

def export_city() -> None:
    """
    Export the current view of the map to both a PNG and a PDF.  The
    exported PNG covers only the map area (excluding the UI panels
    and control strips) and preserves the current camera zoom and
    panning.  The PDF embeds the same image and adds invisible link
    annotations over each facility footprint.  Clicking these
    annotations in a PDF viewer reveals the facility name via the
    annotation's /Contents.
    """
    try:
        # Create an off‑screen surface matching the map region.  Exclude
        # the facilities panel and right‑hand sliders.  Only the map
        # (roads, buildings, water, sea, walls, bridges and facilities)
        # will be drawn to this surface.
        export_surface = pygame.Surface((MAP_WIDTH, HEIGHT))
        # Fill background
        export_surface.fill(BG_COLOR)
        # Draw grid just like the on‑screen version
        grid_step = Si(24)
        # Determine world bounds visible in the current viewport
        world_x0 = cam_offset[0]
        world_y0 = cam_offset[1]
        world_x1 = cam_offset[0] + MAP_WIDTH / cam_zoom
        world_y1 = cam_offset[1] + HEIGHT / cam_zoom
        vx = int(math.floor(world_x0 / grid_step) * grid_step)
        while vx <= world_x1 + grid_step:
            sx = (vx - cam_offset[0]) * cam_zoom
            if -grid_step <= sx <= MAP_WIDTH + grid_step:
                pygame.draw.line(export_surface, GRID_COLOR, (sx, 0), (sx, HEIGHT), 1)
            vx += grid_step
        vy = int(math.floor(world_y0 / grid_step) * grid_step)
        while vy <= world_y1 + grid_step:
            sy = (vy - cam_offset[1]) * cam_zoom
            if -grid_step <= sy <= HEIGHT + grid_step:
                pygame.draw.line(export_surface, GRID_COLOR, (0, sy), (MAP_WIDTH, sy), 1)
            vy += grid_step
        # Synchronise decoration and roads classification
        try:
            decor.ensure_synced(water, sea)
        except Exception:
            pass
        try:
            roads.get_classed()
        except Exception:
            pass
        # Define a world_to_export_screen conversion that omits the
        # facilities panel offset.  This mirrors world_to_screen but
        # starts from the left edge of the exported surface (x=0).
        def world_to_export_screen(pt: tuple[float, float]) -> tuple[float, float]:
            return (
                (pt[0] - cam_offset[0]) * cam_zoom,
                (pt[1] - cam_offset[1]) * cam_zoom,
            )
        # Draw walls network and map bounds (generation rectangle)
        try:
            walls.draw_network_clipped(export_surface, world_to_export_screen, cam_zoom, roads)
        except Exception:
            pass
        # Draw generation bounds rectangle (respect borders)
        try:
            # Normalise bounds as in draw_map_bounds()
            x0, y0, x1, y1 = GEN_BOUNDS
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            px0, py0 = world_to_export_screen((x0, y0))
            px1, py1 = world_to_export_screen((x1, y1))
            rect = pygame.Rect(int(min(px0, px1)), int(min(py0, py1)), int(abs(px1 - px0)), int(abs(py1 - py0)))
            pygame.draw.rect(export_surface, (60, 60, 60), rect, max(1, int(1 * cam_zoom)))
        except Exception:
            pass
        # Draw primary content: roads, buildings, facilities, sea, water and decoration
        try:
            roads.draw_roads(
                export_surface,
                world_to_export_screen,
                cam_zoom,
                colors={"ROAD": ROAD_COLOR},
                water=water,
                aa=True,
                show_names=SHOW_ROAD_NAMES if 'SHOW_ROAD_NAMES' in globals() else False,
            )
        except Exception:
            pass
        try:
            builds.draw(export_surface, world_to_export_screen, cam_zoom)
        except Exception:
            pass
        # Draw facilities and record bounding boxes for annotations
        facility_annots: list[tuple[float, float, float, float, str]] = []
        try:
            # Draw facilities on the export surface and compute their
            # bounding boxes in image coordinates
            for fac in facilities.facilities:
                poly = fac.get("poly")
                spec = fac.get("spec")
                # Skip if no polygon
                if not poly:
                    continue
                pts_screen = [world_to_export_screen(p) for p in poly]
                # Draw facility polygon and outline similar to draw()
                colour = spec.color if spec else (180, 180, 180)
                shadow_offset = (2, 2)
                shadow_color = (100, 100, 100)
                sh = [(x + shadow_offset[0], y + shadow_offset[1]) for (x, y) in pts_screen]
                pygame.draw.polygon(export_surface, shadow_color, sh)
                pygame.draw.polygon(export_surface, colour, pts_screen)
                pygame.draw.polygon(export_surface, (80, 80, 80), pts_screen, max(1, int(2 * cam_zoom)))
                # Bounding box for annotation
                xs = [p[0] for p in pts_screen]
                ys = [p[1] for p in pts_screen]
                x0_b, x1_b = min(xs), max(xs)
                y0_b, y1_b = min(ys), max(ys)
                name = spec.name if spec else "Facility"
                facility_annots.append((x0_b, y0_b, x1_b, y1_b, name))
            # After drawing facility polygons, draw the markers as in original draw()
            for fac in facilities.facilities:
                spec = fac.get("spec")
                if spec and spec.kind == "square":
                    continue
                # Compute centroid for marker
                poly = fac.get("poly")
                if not poly:
                    continue
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
                sx, sy = world_to_export_screen((cx, cy))
                radius = 5
                r, g, b = spec.color if spec else (200, 200, 200)
                border_col = (max(r - 60, 0), max(g - 60, 0), max(b - 60, 0))
                centre_col = (max(r - 80, 0), max(g - 80, 0), max(b - 80, 0))
                fill_col = spec.color if spec else (200, 200, 200)
                pygame.draw.circle(export_surface, border_col, (int(sx), int(sy)), radius + 2)
                pygame.draw.circle(export_surface, fill_col, (int(sx), int(sy)), radius)
                pygame.draw.circle(export_surface, centre_col, (int(sx), int(sy)), max(1, radius // 2))
        except Exception:
            pass
        # Draw water overlays and grass, sea and decorations
        try:
            sea.draw_overlay(export_surface, world_to_export_screen, cam_zoom)
        except Exception:
            pass
        if SHOW_WATER:
            try:
                water.draw_overlay(export_surface, world_to_export_screen, cam_zoom)
            except Exception:
                pass
            try:
                water_mod.draw_grass_top(export_surface, world_to_export_screen, cam_zoom, water)
            except Exception:
                pass
        try:
            decor.draw(export_surface, world_to_export_screen, cam_zoom)
        except Exception:
            pass
        try:
            bridges.draw_overlay(export_surface, world_to_export_screen, cam_zoom, roads, water)
        except Exception:
            pass
        # Draw walls overlay above all
        try:
            walls.draw_overlay(export_surface, world_to_export_screen, cam_zoom, roads, water)
        except Exception:
            pass
        # Save the PNG
        import os
        png_path = os.path.join(os.getcwd(), "exported_city.png")
        pygame.image.save(export_surface, png_path)
        # Prepare annotation rectangles for PDF.  Convert from image
        # coordinate system (origin top‑left) to PDF coordinate system
        # (origin bottom‑left).  The PDF coordinate system uses the
        # same units as the image (points per pixel).
        pdf_annots: list[tuple[float,float,float,float,str]] = []
        for (x0_b, y0_b, x1_b, y1_b, name) in facility_annots:
            # Ensure the rectangle is properly ordered
            llx = max(0.0, min(x0_b, x1_b))
            urx = min(MAP_WIDTH, max(x0_b, x1_b))
            # Convert y from top‑down to bottom‑up
            lly = max(0.0, min(HEIGHT - max(y0_b, y1_b), HEIGHT))
            ury = min(HEIGHT, max(HEIGHT - min(y0_b, y1_b), 0.0))
            pdf_annots.append((llx, lly, urx, ury, name))
        # Write the PDF file.  Use raw pixel data from the current
        # export surface rather than reloading via Pillow.  This
        # avoids dependency on external image libraries.  The raw
        # RGB bytes are obtained from the export surface; size is
        # determined from the surface dimensions.
        pdf_path = os.path.join(os.getcwd(), "exported_city.pdf")
        try:
            # Retrieve raw RGB bytes from the surface.  The string
            # returned by pygame.image.tostring() is row‑major from
            # top‑left and uses 3 bytes per pixel (RGB).
            rgb_bytes = pygame.image.tostring(export_surface, 'RGB')
            _write_pdf_with_annotations_data(rgb_bytes, (MAP_WIDTH, HEIGHT), pdf_path, pdf_annots)
        except Exception as _pdf_e:
            # Fallback: if raw export fails, attempt file‑based export
            _write_pdf_with_annotations(png_path, pdf_path, pdf_annots)
        print(f"[Export] Saved export to {png_path} and {pdf_path}")
    except Exception as _e:
        print("[Export] Failed:", _e)

clock = pygame.time.Clock()
running=True
cutting=False
cut_start=None
cut_curr=None

while running:
    clock.tick(60)
    for event in pygame.event.get():
        # ---- keyboard ----
        if event.type == pygame.KEYDOWN:
            # C -> hide & clear sea + river
            if event.key == pygame.K_c:
                SHOW_SEA = False
                SHOW_WATER = False
                try:
                    sea.reset()
                except Exception:
                    pass
                try:
                    water.reset()
                except Exception:
                    pass
            # S -> show & (re)generate sea
            elif event.key == pygame.K_s:
                try:
                    sea.reset()
                except Exception:
                    pass
                sea.generate('bottom')
                SHOW_SEA = True
            
            # Ctrl+W -> clear & disable walls
            elif (event.key == pygame.K_w) and (event.mod & pygame.KMOD_CTRL):
                try:
                    walls.disable(); walls.reset_walls()
                    print("[walls] disabled & cleared (Ctrl+W)")
                except Exception as _e:
                    print("[walls] clear error:", _e)

            # W -> enable & (re)generate walls
            elif (event.key == pygame.K_w):
                try:
                    if not walls.is_enabled():
                        walls.enable()
                    walls.reset_walls()
                    walls.generate_walls(roads, n_gates=3, n_verts=12, debug=False)
                    st = walls.get_wall_state()
                    print(f"[walls] generated (W): enabled={st.get('enabled')} verts={len(st.get('poly', []))}")
                except Exception as _e:
                    print("[walls] generate error:", _e)

# R -> full reset
            elif (event.mod & pygame.KMOD_CTRL and event.key == pygame.K_r) or event.key == pygame.K_r:
                SHOW_WATER = True
                builds.reset()
                water.reset()
                sea.reset()
                bridges.reset_bridges()
        if event.type == pygame.QUIT: running=False

        # Give the facilities UI a chance to consume UI events.  When the
        # user toggles a facility or adjusts its count the panel sets
        # an internal flag indicating that regeneration is required.  We
        # defer the actual regeneration until the user releases the
        # mouse button to avoid repeatedly rebuilding the facilities
        # during slider drags.  If the UI consumes the event, skip
        # further handling.
        if facilities_ui.handle_event(event):
            continue

        if sidebar.handle_event(event): continue

        if event.type == pygame.MOUSEWHEEL:
            mx,my = pygame.mouse.get_pos()
            # Only allow zooming when the cursor is over the map region.
            if LEFT_PANEL_WIDTH <= mx < LEFT_PANEL_WIDTH + MAP_WIDTH:
                anchor_world = screen_to_world((mx, my))
                zoom_factor = 1.1 if event.y > 0 else 1.0 / 1.1
                new_zoom = max(0.2, min(5.0, cam_zoom * zoom_factor))
                # Maintain the point under the cursor during zoom by
                # adjusting the camera offset.  The x coordinate is
                # corrected for the facilities panel width.
                cam_offset[0] = anchor_world[0] - ((mx - LEFT_PANEL_WIDTH) / new_zoom)
                cam_offset[1] = anchor_world[1] - (my / new_zoom)
                cam_zoom = new_zoom

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx,my = event.pos
            # Handle export button click before any other interaction.  If
            # the user clicks the export button, generate the PNG and
            # PDF and skip further processing of this mouse event.  This
            # prevents manual road drawing or facility selection from
            # interfering with the export.
            try:
                if export_btn_rect.collidepoint(mx, my):
                    export_city()
                    continue
            except Exception:
                pass
            # First handle facility bullet clicks.  Clicking on a facility
            # marker toggles its label and consumes the event so it
            # does not fall through to map editing.  Pass the
            # world_to_screen function and zoom so the facility system can
            # locate its markers.
            try:
                if facilities.handle_click((mx, my), world_to_screen, cam_zoom):
                    continue
            except Exception:
                pass
            # If a bridge preview is active, commit it on left-click and skip
            # manual road interactions.  This allows users to place the
            # bridge exactly where they click without interference.
            try:
                import bridges as _bridges_mod
                if getattr(_bridges_mod, '_PREVIEW_GEOM', None) is not None:
                    # commit and clear preview
                    _bridges_mod.commit_preview_bridge()
                    print('[DBG] commit preview -> bridge'); dbg_prev_visible=False; placing_bridge=False
                    # do not perform manual connect/seed
                    continue
            except Exception:
                pass
            if reset_btn_rect.collidepoint(mx,my):
                snapshot_state(); reset_everything(); continue
            if LEFT_PANEL_WIDTH <= mx < LEFT_PANEL_WIDTH + MAP_WIDTH:
                # handles
                hx, hy = 10, 10
                corners = [(GEN_BOUNDS[0], GEN_BOUNDS[1]), (GEN_BOUNDS[2], GEN_BOUNDS[1]), (GEN_BOUNDS[2], GEN_BOUNDS[3]), (GEN_BOUNDS[0], GEN_BOUNDS[3])]
                dragging_handle = None
                for i,(wxh,wyh) in enumerate(corners):
                    sxh, syh = world_to_screen((wxh, wyh))
                    rect = pygame.Rect(int(sxh-hx/2), int(syh-hy/2), hx, hy)
                    if rect.collidepoint(mx, my):
                        snapshot_state()
                        dragging_handle = i; break
                if dragging_handle is not None: continue
                # manual connect or seed
                wx,wy = screen_to_world((mx,my))
                proj, seg, t = roads.try_snap_segment((wx,wy))
                snapshot_state()
                if seg is not None:
                    dragging_road=True; road_drag_start=(proj, seg); road_drag_curr=proj
                    _drag_saved_overrides = {
                        'SNAP_RADIUS_NODE': PARAMS.get('SNAP_RADIUS_NODE'),
                        'SNAP_RADIUS_SEG':  PARAMS.get('SNAP_RADIUS_SEG'),
                        'HOUSE_STEP_MIN':   PARAMS.get('HOUSE_STEP_MIN'),
                    }
                    PARAMS['SNAP_RADIUS_NODE'] = Si(28)
                    PARAMS['SNAP_RADIUS_SEG']  = Si(28)
                    PARAMS['HOUSE_STEP_MIN']   = 0
                else:
                    roads.spawn_seed(wx, wy, start_level=1)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
            mx,my = event.pos
            if LEFT_PANEL_WIDTH <= mx < LEFT_PANEL_WIDTH + MAP_WIDTH:
                panning=True; pan_start_screen=(mx,my); pan_start_offset=(cam_offset[0], cam_offset[1])

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
            panning=False; pan_start_screen=None; pan_start_offset=None
            # If any facility UI changes were made during this interaction,
            # regenerate the facilities now.  This deferred regeneration
            # prevents expensive rebuilding on every slider motion.
            if getattr(facilities_ui, 'await_regen', False):
                enabled = [name for name, on in facilities_ui.enabled.items() if on]
                try:
                    facilities.generate(roads, builds, water, enabled_names=enabled)
                    _cleanup_buildings_vs_squares()
                except Exception:
                    pass
                facilities_ui.await_regen = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            mx, my = event.pos
            # If a bridge preview is active, cancel it on right-click and skip
            # entering cut mode.  This allows users to abort preview
            # placement without accidental cuts.
            try:
                import bridges as _bridges_mod
                if getattr(_bridges_mod, '_PREVIEW_GEOM', None) is not None:
                    _bridges_mod.cancel_preview_bridge()
                    continue
            except Exception:
                pass
            if LEFT_PANEL_WIDTH <= mx < LEFT_PANEL_WIDTH + MAP_WIDTH:
                cutting = True
                cut_start = screen_to_world((mx, my))
                cut_curr = cut_start

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            if cutting and (cut_start is not None) and (cut_curr is not None):
                # remove only main network segments (level 1)
                roads.remove_segments_crossing_line(cut_start, cut_curr, level_filter=None)
            cutting = False
            cut_start = None
            cut_curr = None
    
            panning=False; pan_start_screen=None; pan_start_offset=None

            # After handling right-button release, check if facility UI
            # requested a regeneration and perform it once.  This avoids
            # repeated regeneration while dragging sliders.
            if getattr(facilities_ui, 'await_regen', False):
                enabled = [name for name, on in facilities_ui.enabled.items() if on]
                try:
                    facilities.generate(roads, builds, water, enabled_names=enabled)
                    _cleanup_buildings_vs_squares()
                except Exception:
                    pass
                facilities_ui.await_regen = False

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if dragging_road and road_drag_start is not None:
                p1, s1 = road_drag_start
                wx, wy = screen_to_world((mx, my))
                # widen snapping for end too
                old_snap_node = PARAMS.get('SNAP_RADIUS_NODE'); old_snap_seg = PARAMS.get('SNAP_RADIUS_SEG')
                PARAMS['SNAP_RADIUS_NODE'] = Si(28); PARAMS['SNAP_RADIUS_SEG'] = Si(28)
                proj2, seg2, t2 = roads.try_snap_segment((wx, wy))
                # derive endpoints
                p_start = p1
                s_start = s1
                if s_start is None:
                    # try to snap start to the nearest segment
                    projS, segS, tS = roads.try_snap_segment(p1)
                    if segS is not None:
                        p_start = projS; s_start = segS
                p_end = proj2 if seg2 is not None else (wx, wy)
                s_end = seg2
                # perform splits where we have segments
                if s_start is not None:
                    roads.split_segment_at_point(s_start, p_start)
                if s_end is not None and s_end is not s_start:
                    roads.split_segment_at_point(s_end, p_end)
                # add the manual segment regardless of snap status
                ok = roads.add_manual_segment(p_start, p_end, level=2, water=water, min_spacing_override=0)
                # restore snap radii
                PARAMS['SNAP_RADIUS_NODE'] = old_snap_node
                PARAMS['SNAP_RADIUS_SEG']  = old_snap_seg
                # restore drag-only overrides
                if _drag_saved_overrides is not None:
                    PARAMS['SNAP_RADIUS_NODE'] = _drag_saved_overrides.get('SNAP_RADIUS_NODE', PARAMS['SNAP_RADIUS_NODE'])
                    PARAMS['SNAP_RADIUS_SEG']  = _drag_saved_overrides.get('SNAP_RADIUS_SEG',  PARAMS['SNAP_RADIUS_SEG'])
                    PARAMS['HOUSE_STEP_MIN']   = _drag_saved_overrides.get('HOUSE_STEP_MIN',   PARAMS['HOUSE_STEP_MIN'])
                    _drag_saved_overrides = None
                dragging_road=False; road_drag_start=None; road_drag_curr=None
                dragging_road=False; road_drag_start=None; road_drag_curr=None
            if dragging_handle is not None:
                roads.set_generation_bounds(tuple(GEN_BOUNDS)); dragging_handle=None

            # On left-button release, regenerate facilities if needed.  The
            # facilities UI sets a flag when the user toggles a checkbox or
            # adjusts a slider; we honour that flag here to rebuild
            # facilities exactly once per interaction.
            if getattr(facilities_ui, 'await_regen', False):
                enabled = [name for name, on in facilities_ui.enabled.items() if on]
                try:
                    facilities.generate(roads, builds, water, enabled_names=enabled)
                    _cleanup_buildings_vs_squares()
                except Exception:
                    pass
                facilities_ui.await_regen = False
        
        elif event.type == pygame.MOUSEMOTION:
            if cutting:
                mx, my = event.pos
                cut_curr = screen_to_world((mx, my))

            mx,my = event.pos
            if placing_bridge:
                try:
                    import bridges as _bridges_mod
                    if getattr(_bridges_mod, '_PREVIEW_GEOM', None) is None:
                        wx, wy = screen_to_world((mx, my))
                        try:
                            _bridges_mod.update_bridge_preview_blend((wx, wy), water, roads=roads, level=1, angle_blend=0.5)
                        except Exception:
                            _bridges_mod.update_bridge_preview((wx, wy), water, level=1)
                    if not dbg_prev_visible and getattr(_bridges_mod, '_PREVIEW_GEOM', None) is not None:
                        dbg_prev_visible=True; print('[DBG] preview acquired')
                except Exception:
                    pass
            # update red cut preview if active
            if cutting and cut_start is not None:
                cut_curr = screen_to_world((mx, my))

            mmb_down = pygame.mouse.get_pressed(3)[1]
            if panning and (pan_start_screen is None or pan_start_offset is None or not mmb_down):
                panning=False; pan_start_screen=None; pan_start_offset=None
            if panning and pan_start_screen is not None and pan_start_offset is not None:
                dx = mx - pan_start_screen[0]; dy = my - pan_start_screen[1]
                cam_offset[0] = pan_start_offset[0] - dx / cam_zoom
                cam_offset[1] = pan_start_offset[1] - dy / cam_zoom
            if dragging_road and road_drag_start is not None:
                wx,wy = screen_to_world((mx,my))
                proj, seg, t = roads.try_snap_segment((wx,wy))
                road_drag_curr = proj if seg is not None else (wx,wy)
            if dragging_handle is not None:
                wx,wy = screen_to_world((mx,my))
                wx = max(WORLD_X, min(WORLD_X + WORLD_WIDTH, wx)); wy = max(WORLD_Y, min(WORLD_Y + WORLD_HEIGHT, wy))
                if dragging_handle == 0: GEN_BOUNDS[0], GEN_BOUNDS[1] = wx, wy
                elif dragging_handle == 1: GEN_BOUNDS[2], GEN_BOUNDS[1] = wx, wy
                elif dragging_handle == 2: GEN_BOUNDS[2], GEN_BOUNDS[3] = wx, wy
                elif dragging_handle == 3: GEN_BOUNDS[0], GEN_BOUNDS[3] = wx, wy

            # If a bridge preview is active, update it to follow the mouse.  We
            # avoid spawning a preview on mere motion by checking if one
            # exists in the bridges module.
            try:
                # Access the preview state via bridges module
                import bridges as _bridges_mod
                if getattr(_bridges_mod, '_PREVIEW_GEOM', None) is not None:
                    wx, wy = screen_to_world((mx, my))
                    # Update using blended preview so the bridge follows nearby road direction
                    try:
                        _bridges_mod.update_bridge_preview_blend((wx, wy), water, roads=roads, level=1, angle_blend=0.5)
                    except Exception:
                        # Fallback to original preview update if blended version is unavailable
                        _bridges_mod.update_bridge_preview((wx, wy), water, level=1)
            except Exception:
                pass

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                restore_state_pop()
            if event.key == pygame.K_r:
                # Snapshot current state for undo
                snapshot_state()
                # Generate a new river only. Do not generate bridges or rebuild the road network here.
                roads.lock_mutations()
                roads.disable_constraints()
                water.generate_river()
                roads.enable_constraints()
                roads.unlock_mutations()
                # Clear any existing bridge overlay so it does not persist with the new river
                bridges.reset_bridges()
                # Clear walls when a new river is generated, as the city footprint may change
                try:
                    walls.reset_walls()
                except Exception:
                    pass
            if event.key == pygame.K_b:
                snapshot_state()
                # Regenerate all houses
                builds.generate(roads, water)
                # Regenerate facilities according to currently enabled specs
                enabled = [name for name, on in facilities_ui.enabled.items() if on]
                try:
                    facilities.generate(roads, builds, water, enabled_names=enabled)
                    _cleanup_buildings_vs_squares()
                except Exception:
                    pass
            if event.key == pygame.K_g:
                # Snapshot current state for undo
                snapshot_state()
                # Commit any existing preview before starting a new one
                try:
                    commit_preview_bridge()
                except Exception:
                    pass
                # Determine world position of mouse (clamp X inside map to avoid off-map preview)
                mx, my = pygame.mouse.get_pos()
                if mx >= LEFT_PANEL_WIDTH + MAP_WIDTH:
                    mx = LEFT_PANEL_WIDTH + MAP_WIDTH - 1
                wx, wy = screen_to_world((mx, my))
                # Start a blended-direction preview (no snapping, just a visible preview)
                ok = start_bridge_preview((wx, wy), water, roads, level=1, angle_blend=0.5)
                ok_dbg = ok if 'ok' in locals() else True; print(f"[DBG] G: start_preview ok={ok_dbg} river={bool(getattr(water,'river_poly',None))} len={len(water.river_poly) if getattr(water,'river_poly',None) else 0} mouse=({mx},{my}) world=({wx:.1f},{wy:.1f})")
                placing_bridge=True
            elif event.key == pygame.K_ESCAPE:
                try:
                    cancel_preview_bridge()
                except Exception:
                    pass
                placing_bridge=False
            
            if event.key == pygame.K_w:
                # Shift+W -> 2 gates, W -> 3 gates
                snapshot_state()
                n_gates = 2 if (event.mod & pygame.KMOD_SHIFT) else 3
                try:
                    walls.reset_walls()
                    walls.generate_walls(roads, water, n_gates=n_gates)
                except Exception:
                    pass
            # Cancel preview bridge with ESC key
            if event.key == pygame.K_ESCAPE:
                try:
                    import bridges as _bridges_mod
                    if getattr(_bridges_mod, '_PREVIEW_GEOM', None) is not None:
                        _bridges_mod.cancel_preview_bridge()
                except Exception:
                    pass
            if event.key == pygame.K_h: snapshot_state(); sea.generate_harbor()


    if roads.active:

        # --- growth limits (from sliders) ---
        if len(roads.segments) >= int(PARAMS.get("SEGMENT_COUNT_LIMIT", 100000)):
            roads.active = []
        step_cap = int(PARAMS.get("STEP_LIMIT", 100000))
        for br in roads.active[:]:
            if step_cap <= 0: break
            step_cap -= 1
            children = roads.step_branch(br, water=water)
            if children: roads.active.extend(children)
            if br["remaining"]<=0 and br in roads.active: roads.active.remove(br)

    screen.fill(BG_COLOR)

    # grid (light)
    grid_step = Si(24)
    world_x0 = cam_offset[0]; world_y0 = cam_offset[1]
    world_x1 = cam_offset[0] + MAP_WIDTH / cam_zoom
    world_y1 = cam_offset[1] + HEIGHT / cam_zoom
    vx = int(math.floor(world_x0 / grid_step) * grid_step)
    while vx <= world_x1 + grid_step:
        sx = LEFT_PANEL_WIDTH + (vx - cam_offset[0]) * cam_zoom
        if -grid_step <= sx - LEFT_PANEL_WIDTH <= MAP_WIDTH + grid_step:
            pygame.draw.line(screen, GRID_COLOR, (sx, 0), (sx, HEIGHT), 1)
        vx += grid_step
    vy = int(math.floor(world_y0 / grid_step) * grid_step)
    while vy <= world_y1 + grid_step:
        sy = (vy - cam_offset[1]) * cam_zoom
        if -grid_step <= sy <= HEIGHT + grid_step:
            pygame.draw.line(screen, GRID_COLOR, (LEFT_PANEL_WIDTH, sy), (LEFT_PANEL_WIDTH + MAP_WIDTH, sy), 1)
        vy += grid_step

    # water.draw(screen, world_to_screen, cam_zoom)
    # ensure decorations up-to-date
    decor.ensure_synced(water, sea)  # moved below
    roads.get_classed()
    walls.draw_network_clipped(screen, world_to_screen, cam_zoom, roads)
    draw_map_bounds()
    # Draw city walls above terrain and roads but below water and sea overlays
    try:
        walls.draw_overlay(screen, world_to_screen, cam_zoom, roads, water)
    except Exception:
        pass


    # preview cut line (right-drag)
    if cutting and cut_start is not None and cut_curr is not None:
        p1 = world_to_screen(cut_start); p2 = world_to_screen(cut_curr)
        pygame.draw.line(screen, (200,40,40), p1, p2, max(1, int(2*cam_zoom)))
    
    # preview manual line
    if dragging_road and road_drag_start is not None and road_drag_curr is not None:
        p1 = world_to_screen(road_drag_start[0]); p2 = world_to_screen(road_drag_curr)
        pygame.draw.line(screen, (40,170,70), p1, p2, max(1,int(2*cam_zoom)))
        # Clip water & bridges to the map rectangle so nothing renders outside the map
    prev_clip = screen.get_clip()
    # Clip drawing to the map region only; exclude the facilities panel on the left
    screen.set_clip(pygame.Rect(LEFT_PANEL_WIDTH, 0, MAP_WIDTH, HEIGHT))
    # Draw the built environment first: houses, then facilities.  By
    # drawing facilities here, we ensure their footprints are visible
    # beneath the upcoming water and sea overlays.  Facilities are
    # integrated into the landscape just like houses.
    
    # Draw roads (primary/secondary/tertiary/alley) before buildings and water
    try:
        roads.draw_roads(
            screen,
            world_to_screen,
            cam_zoom,
            colors={"ROAD": ROAD_COLOR},
            water=water,
            aa=True,
            show_names=SHOW_ROAD_NAMES if 'SHOW_ROAD_NAMES' in globals() else False
        )
    except Exception:
        pass

    builds.draw(screen, world_to_screen, cam_zoom)
    try:
        facilities.draw(screen, world_to_screen, cam_zoom)
    except Exception:
        pass
    # Now overlay the sea and river on top of the built environment.  The
    # sea should always appear above buildings and facilities.  Draw
    # water overlays only if they are enabled.  Grass and shoreline
    # embellishments follow after water overlays.
    sea.draw_overlay(screen, world_to_screen, cam_zoom)
    if SHOW_WATER:
        water.draw_overlay(screen, world_to_screen, cam_zoom)
    if SHOW_WATER:
        water_mod.draw_grass_top(screen, world_to_screen, cam_zoom, water)
    # connectors above everything except bridges
    # We intentionally disable drawing bridge connectors so that roads do not try to snap
    # to the bridges.  Only draw the bridge decks themselves.
    decor.draw(screen, world_to_screen, cam_zoom)
    bridges.draw_overlay(screen, world_to_screen, cam_zoom, roads, water)
    try:
        import bridges as _bridges_mod
        _pg = pygame
        if getattr(_bridges_mod, '_PREVIEW_GEOM', None) is not None:
            a, b, _lvl = _bridges_mod._PREVIEW_GEOM
            PREVIEW_LINE_SCALE = 0.6  # draw fallback line at 25% of raw preview length
            mxp = 0.5 * (a[0] + b[0]); myp = 0.5 * (a[1] + b[1])
            a = (mxp + (a[0]-mxp)*PREVIEW_LINE_SCALE, myp + (a[1]-myp)*PREVIEW_LINE_SCALE)
            b = (mxp + (b[0]-mxp)*PREVIEW_LINE_SCALE, myp + (b[1]-myp)*PREVIEW_LINE_SCALE)
            _pg.draw.line(screen, (220, 60, 60), world_to_screen(a), world_to_screen(b), max(2, int(2*cam_zoom)))
            # endpoints markers
            _pg.draw.circle(screen, (30,30,30), world_to_screen(a), max(2, int(2*cam_zoom)), 1)
            _pg.draw.circle(screen, (30,30,30), world_to_screen(b), max(2, int(2*cam_zoom)), 1)
    except Exception:
        pass
    screen.set_clip(prev_clip)

# helper notes
    note1 = font.render("[R] River  [S] Sea  [C] Clear water", True, (30,30,30))
    note2 = font.render("[B] Buildings  [O] Road names  [Ctrl+Z] Undo", True, (30,30,30))
    note3 = font.render("[G] Place bridge (click to set, Esc to cancel)", True, (30,30,30))
    note4 = font.render("[W] Walls", True, (30,30,30))
    screen.blit(note1, (LEFT_PANEL_WIDTH + 20, HEIGHT - 60))
    screen.blit(note2, (LEFT_PANEL_WIDTH + 20, HEIGHT - 35))
    # Walls note appears above other notes
    screen.blit(note4, (LEFT_PANEL_WIDTH + 20, HEIGHT - 80))
    screen.blit(note3, (LEFT_PANEL_WIDTH + 20, HEIGHT - 20))

    # Export and Reset buttons + Sidebar
    # Draw export button to the left of the reset button.  Use the
    # same styling as the reset button for visual consistency.
    pygame.draw.rect(screen, export_bg_color, export_btn_rect)
    pygame.draw.rect(screen, (0,0,0), export_btn_rect, 2)
    screen.blit(export_text, (export_btn_rect.x + pad_x, export_btn_rect.y + pad_y))
    # Draw reset button
    pygame.draw.rect(screen, reset_bg_color, reset_btn_rect)
    pygame.draw.rect(screen, (0,0,0), reset_btn_rect, 2)
    screen.blit(reset_text, (reset_btn_rect.x + pad_x, reset_btn_rect.y + pad_y))
    # Draw the right‑hand parameter sliders
    sidebar.draw_bg(screen); sidebar.draw(screen)
    # Draw the facilities UI panel on the left.  Draw after the map
    # and before final display flip so it appears on top of everything.
    facilities_ui.draw_bg(screen); facilities_ui.draw(screen)

    # Draw lightweight decorative instances
# (moved) decor.draw(screen, world_to_screen, cam_zoom)
    pygame.display.flip()

pygame.quit()

# --- helper: ensure squares displace buildings (post facilities.generate) ---
def _cleanup_buildings_vs_squares():
    print("[Squares] cleanup hook running")
    try:
        bs = globals().get("builds") or globals().get("buildings")
        fs = globals().get("facilities")
        if not bs or not fs:
            print("[Squares] no builds/facilities handle, skipping"); return
        polys = fs.square_polys() if hasattr(fs, "square_polys") else []
        if not polys:
            print("[Squares] no square polygons to clip"); return
        removed = bs.remove_inside_polys(polys) if hasattr(bs, "remove_inside_polys") else 0
        print(f"[Squares] removed {removed} buildings in squares (post-hook)")
    except Exception as _e:
        print("[Squares] cleanup failed", _e)
