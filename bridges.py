
# bridges.py
# Bridges generation + overlay rendering ABOVE the river layer.
# - generate_bridges(...) : places bridge road segments that cross the river
# - draw_overlay(...)     : draws bridge decks/rails on a separate layer above water
#
# Plug-in points in city_weaver.py:
#   after drawing water overlay -> call bridges.draw_overlay(screen, world_to_screen, cam_zoom, roads, water)
#
from __future__ import annotations
import math, random

# --- Visual-only mode to prevent road network mutations from bridges ---
VISUAL_ONLY = True
def _safe_add_manual_segment(roads, *a, **kw):
    if VISUAL_ONLY: return False
    return _safe_add_manual_segment(roads, *a, **kw)
def _safe_split_segment_at_point(roads, *a, **kw):
    if VISUAL_ONLY: return None
    return _safe_split_segment_at_point(roads, *a, **kw)
def _safe_split_segment_and_add(roads, *a, **kw):
    if VISUAL_ONLY: return False
    return _safe_split_segment_and_add(roads, *a, **kw)
from typing import List, Tuple, Optional, Iterable

from geometry import seg_intersection, road_tube_poly, point_in_poly, point_seg_dist

# Global length scaler for visual bridge deck (applied symmetrically around midpoint)
BRIDGE_LENGTH_SCALE_DEFAULT = 0.6  # quarter length

_GEN_ID = 0
def _new_overlay_state():
    return set(), [], [], []  # BRIDGES, BRIDGE_GEOMS, APPROACH_GEOMS, APPROACH_CURVES
def _assign_overlay_state():
    global _BRIDGES, _BRIDGE_GEOMS, _APPROACH_GEOMS, _APPROACH_CURVES
    _BRIDGES, _BRIDGE_GEOMS, _APPROACH_GEOMS, _APPROACH_CURVES = _new_overlay_state()

# Global preview state for interactive bridge placement.  When the user
# presses G in city_weaver.py, a bridge preview is created
# that follows the mouse cursor until the user clicks to place it.  The
# preview bridge is stored as a pair of endpoints in world space
# (_PREVIEW_GEOM) with an associated level.  None indicates no preview.
_PREVIEW_GEOM: Optional[Tuple[Point, Point, int]] = None



# --- Accessor for overlay bridge deck geoms (centerlines) ---
def get_bridge_geoms() -> list:
    """
    Return a list of (A, B, level) tuples describing each bridge deck
    generated in the current overlay.  Each tuple contains the
    world-space endpoints A and B of the uniform deck span as well
    as the level originally requested in `generate_bridges`.  Call
    `generate_bridges(...)` before using this function; otherwise
    the returned list may be empty.
    """
    global _BRIDGE_GEOMS
    # return a copy to avoid accidental mutations of the internal state
    return list(_BRIDGE_GEOMS)


def _clear_bridge_overlay_cache():
    global _GEN_ID
    _GEN_ID += 1
    _assign_overlay_state()

def _tube_polyline_world(points, width):
    """Return a polygon (list of world-space points) representing a tube around a polyline.
    Uses simple miter joins with a soft clamp. Expects >=2 points.
    """
    if not points or len(points) < 2:
        return []
    import math
    hw = float(width) * 0.5
    Ls = []
    Rs = []
    def _norm(a, b):
        ax, ay = a; bx, by = b
        vx, vy = (bx-ax, by-ay)
        L = math.hypot(vx, vy) or 1.0
        nx, ny = (vy/L, -vx/L)
        return nx, ny
    n = len(points)
    for i, P in enumerate(points):
        if i == 0:
            nx, ny = _norm(points[i], points[i+1])
        elif i == n-1:
            nx, ny = _norm(points[i-1], points[i])
        else:
            n1x, n1y = _norm(points[i-1], points[i])
            n2x, n2y = _norm(points[i], points[i+1])
            # miter with clamp
            mx, my = (n1x+n2x, n1y+n2y)
            mL = math.hypot(mx, my)
            if mL < 1e-6:
                nx, ny = n2x, n2y
            else:
                mx, my = mx/mL, my/mL
                # clamp miter length to avoid spikes
                dot = max(-1.0, min(1.0, n1x*n2x + n1y*n2y))
                miter_scale = 1.0 / max(0.3, (1.0 + dot))
                nx, ny = mx * miter_scale, my * miter_scale
        Ls.append((P[0] + nx*hw, P[1] + ny*hw))
        Rs.append((P[0] - nx*hw, P[1] - ny*hw))
    poly = Ls + list(reversed(Rs))
    return poly


Point = Tuple[float, float]

# Track only the manual bridges we create here
_assign_overlay_state()

# ----------------------- helpers used by generation -----------------------
def _arclen(poly: List[Point]) -> List[float]:
    L = [0.0]
    for i in range(1, len(poly)):
        ax, ay = poly[i-1]; bx, by = poly[i]
        d = math.hypot(bx-ax, by-ay)
        L.append(L[-1] + d)
    return L

def _resample_by_count(poly: List[Point], count: int) -> List[Point]:
    count = max(2, int(count))
    if len(poly) <= 2:
        return list(poly)
    segs = []
    total = 0.0
    for i in range(1, len(poly)):
        ax, ay = poly[i-1]; bx, by = poly[i]
        d = math.hypot(bx-ax, by-ay)
        segs.append((d, (ax,ay), (bx,by)))
        total += d
    if total <= 1e-6:
        return [poly[0], poly[-1]]
    step = total / (count - 1)
    pts = [poly[0]]
    dist = 0.0
    si = 0
    while len(pts) < count - 1 and si < len(segs):
        seg_len, A, B = segs[si]
        if dist + seg_len >= step:
            t = (step - dist) / max(1e-12, seg_len)
            px = A[0] + (B[0] - A[0]) * t
            py = A[1] + (B[1] - A[1]) * t
            pts.append((px, py))
            A = (px, py)
            segs[si] = (math.hypot(B[0]-A[0], B[1]-A[1]), A, B)
            dist = 0.0
        else:
            dist += seg_len
            si += 1
    pts.append(poly[-1])
    return pts

def _split_left_right(river_poly: List[Point]) -> Tuple[List[Point], List[Point]]:
    n = len(river_poly)
    if n < 6 or (n % 2) != 0:
        raise ValueError('river_poly must have an even # of points >= 6')
    half = n // 2
    left = river_poly[:half]
    right = list(reversed(river_poly[half:]))
    return left, right

def _centerline_from_banks(left: List[Point], right: List[Point], samples: int = 200) -> List[Point]:
    Ls = _resample_by_count(left, samples)
    Rs = _resample_by_count(right, samples)
    return [((a[0]+b[0])*0.5, (a[1]+b[1])*0.5) for a,b in zip(Ls, Rs)]

def _tangent(center: List[Point], i: int) -> Point:
    if i <= 0: i = 1
    if i >= len(center)-1: i = len(center)-2
    ax, ay = center[i-1]; bx, by = center[i+1]
    vx, vy = (bx-ax, by-ay)
    L = math.hypot(vx, vy) or 1.0
    return (vx/L, vy/L)

def _bridge_endpoints_through_polygon(center_pt: Point, normal: Point, poly: List[Point],
                                      land_offset: float, min_channel_width: float = 8.0) -> Optional[Tuple[Point, Point]]:
    """Shoot a long segment along +/-normal through the river polygon.
    Pick the *local* bank pair that brackets the center (near t=0.5).
    Extend endpoints onto land by land_offset. Return None if local channel is too narrow."""
    R = 100000.0
    a = (center_pt[0] - normal[0]*R, center_pt[1] - normal[1]*R)
    b = (center_pt[0] + normal[0]*R, center_pt[1] + normal[1]*R)
    hits: List[Tuple[float, Point]] = []
    n = len(poly)
    for i in range(n):
        c = poly[i]; d = poly[(i+1) % n]
        ok, P, ta, _ = seg_intersection(a, b, c, d)
        if ok and P is not None and 0.0 <= ta <= 1.0:
            hits.append((ta, P))
    if len(hits) < 2:
        return None
    hits.sort(key=lambda x: x[0])
    pair = None
    for i in range(len(hits)-1):
        t0, p0 = hits[i]; t1, p1 = hits[i+1]
        if t0 <= 0.5 <= t1:
            pair = (p0, p1)
            break
    if pair is None:
        best = None; best_gap = 1e9
        for i in range(len(hits)-1):
            t0,_ = hits[i]; t1,_ = hits[i+1]
            gap = abs((t0+t1)*0.5 - 0.5)
            if gap < best_gap: best_gap = gap; best = (hits[i][1], hits[i+1][1])
        pair = best
    if pair is None:
        return None
    p_in, q_in = pair
    # check width
    w = ((q_in[0]-p_in[0])**2 + (q_in[1]-p_in[1])**2)**0.5
    if w < float(min_channel_width):
        return None
    p_land = (p_in[0] - normal[0]*land_offset, p_in[1] - normal[1]*land_offset)
    q_land = (q_in[0] + normal[0]*land_offset, q_in[1] + normal[1]*land_offset)
    return p_land, q_land


def _bridge_endpoints_from_banks(center_pt: Point, left_pt: Point, right_pt: Point, land_offset: float) -> Tuple[Point, Point]:
    """Fallback: use paired left/right bank points and push outward by land_offset."""
    # outward from center to each bank
    lx, ly = left_pt; rx, ry = right_pt
    cx, cy = center_pt
    lvx, lvy = (lx - cx, ly - cy); rl = (lvx**2 + lvy**2) ** 0.5 or 1.0
    rvx, rvy = (rx - cx, ry - cy); rr = (rvx**2 + rvy**2) ** 0.5 or 1.0
    lp = (lx + (lvx/rl) * land_offset, ly + (lvy/rl) * land_offset)
    rp = (rx + (rvx/rr) * land_offset, ry + (rvy/rr) * land_offset)
    return lp, rp


def _internal_angle(a: Point, b: Point, c: Point) -> float:
    vx1, vy1 = a[0]-b[0], a[1]-b[1]
    vx2, vy2 = c[0]-b[0], c[1]-b[1]
    L1 = math.hypot(vx1, vy1) or 1.0
    L2 = math.hypot(vx2, vy2) or 1.0
    dot = max(-1.0, min(1.0, (vx1*vx2 + vy1*vy2) / (L1*L2)))
    return math.degrees(math.acos(dot))

# ----------------------- public API: generation -----------------------
def generate_bridges(roads, water, desired: int = 6, min_spacing: float = 220.0, land_offset: float = 24.0, level: int = 1):
    _clear_bridge_overlay_cache()
    return generate_bridges_uniform(
        roads, water,
        desired=desired,
        target_water_span=120.0,
        max_approach_len=4000.0,
        level=level
    )

def _connect_endpoint_to_nearest_road(roads, water, endpoint: Point, exclude_seg, level: int, max_dist: float = 130.0) -> bool:
    """Connect `endpoint` to the nearest *visible* road (skip decorative/water-suppressed).
    If no segment within range, try snapping to a nearby node. Returns True if connected."""
    best = None; best_d = 1e9; best_proj = None
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())
    for s in roads.segments:
        if not getattr(s, "active", True): continue
        if s is exclude_seg: continue
        if s in _BRIDGES: continue
        if s in deco or s in river_skip or s in sea_skip: continue
        d, proj, t = point_seg_dist(endpoint, s.a, s.b)
        if t is None: continue
        if d < best_d:
            best_d = d; best = s; best_proj = proj
    # Try segment connector first
    if best is not None and best_proj is not None and best_d <= max_dist:
        if getattr(water, "river_poly", None) and point_in_poly(best_proj, water.river_poly):
            pass  # fall through to node-snap
        elif getattr(water, "sea_poly", None) and point_in_poly(best_proj, water.sea_poly):
            pass
        else:
            _safe_split_segment_at_point(roads, best, best_proj)
            _safe_add_manual_segment(roads, endpoint, best_proj, level=level, water=None, min_spacing_override=0)
            return True
    # Fallback: nearest node snap (a bit larger radius than SNAP_RADIUS_NODE)
    try:
        snap_pt, idx = roads.try_snap_node(endpoint)
        if idx is not None:
            _safe_add_manual_segment(roads, endpoint, snap_pt, level=level, water=None, min_spacing_override=0)
            return True
    except Exception:
        pass
    return False

def _snap_to_visible_road(roads, water, endpoint: Point, exclude_seg=None, max_dist: float = 150.0):
    """Return a point snapped to the nearest *visible* road segment by splitting that segment.
    Skips decorative and water-suppressed segments. Returns (snapped_point, succeeded)."""
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())

    best = None; best_d = 1e9; best_proj = None
    for s in roads.segments:
        if not getattr(s, "active", True): continue
        if s is exclude_seg: continue
        if s in _BRIDGES: continue
        if s in deco or s in river_skip or s in sea_skip: continue
        d, proj, t = point_seg_dist(endpoint, s.a, s.b)
        if t is None: continue
        if d < best_d:
            best_d = d; best = s; best_proj = proj
    if best is not None and best_proj is not None and best_d <= max_dist:
        if getattr(water, "river_poly", None) and point_in_poly(best_proj, water.river_poly):
            pass
        elif getattr(water, "sea_poly", None) and point_in_poly(best_proj, water.sea_poly):
            pass
        else:
            _safe_split_segment_at_point(roads, best, best_proj)
            return best_proj, True, best
    try:
        snap_pt, idx = roads.try_snap_node(endpoint)
        if idx is not None:
            return snap_pt, True, None
    except Exception:
        pass
    return endpoint, False, None



def _connect_end_to_other_road(roads, water, start_pt: Point, avoid_seg=None, max_dist: float = 280.0) -> bool:
    """Add a straight connector from start_pt to a *different* visible road segment.
    Splits the target segment and adds the connector if the straight line stays on land.
    Returns True if a segment was created."""
    from geometry import point_in_poly, segment_intersects_polygon, seg_intersection, point_seg_dist
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())

    best = None; best_d = 1e9; best_proj = None
    for s in roads.segments:
        if not getattr(s, "active", True): continue
        if s is avoid_seg: continue
        if s in _BRIDGES: continue
        if s in deco or s in river_skip or s in sea_skip: continue
        d, proj, t = point_seg_dist(start_pt, s.a, s.b)
        if t is None: continue
        if d < best_d:
            best_d = d; best = s; best_proj = proj

    if best is None or best_proj is None or best_d > max_dist:
        return False

    # Line from start_pt to best_proj must be on land (no river/sea crossing)
    if getattr(water, "river_poly", None) and segment_intersects_polygon(start_pt, best_proj, water.river_poly):
        return False
    if getattr(water, "sea_poly", None) and segment_intersects_polygon(start_pt, best_proj, water.sea_poly):
        return False

    # Split and add
    _safe_split_segment_at_point(roads, best, best_proj)
    ok = _safe_add_manual_segment(roads, start_pt, best_proj, level=1, water=None, min_spacing_override=0)
    return bool(ok)


def _extend_collinear_to_next_road(roads, water, start_pt: Point, dir_vec: Point, level: int,
                                   avoid_seg=None, max_len: float = 800.0) -> bool:
    """Cast a ray from start_pt along dir_vec and connect straight to the first *visible* road it hits.
    Skips decorative/water-suppressed/bridge segments and rejects paths that cross water. Returns True if added."""
    from geometry import segment_intersects_polygon
    import math
    ux, uy = dir_vec
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux / L, uy / L
    ray_a = start_pt
    ray_b = (start_pt[0] + ux * max_len, start_pt[1] + uy * max_len)
    # iterate all segments, find closest intersection along ray
    best = None; best_t = 1e9; best_pt = None
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())
    for s in roads.segments:
        if not getattr(s, "active", True): continue
        if s is avoid_seg: continue
        if s in deco or s in river_skip or s in sea_skip: continue
        if s in _BRIDGES: continue
        ok, P, t_ray, t_seg = seg_intersection(ray_a, ray_b, s.a, s.b)
        if not ok or P is None or t_ray is None: continue
        if t_ray < 1e-4 or t_ray > 1.0: continue
        if t_seg is None or t_seg < 0.0 or t_seg > 1.0: continue
        if t_ray < best_t:
            best_t = t_ray; best = s; best_pt = P
    if best is None or best_pt is None:
        return False
    # ensure the straight path ray_a -> best_pt stays on land
    if getattr(water, "river_poly", None) and segment_intersects_polygon(ray_a, best_pt, water.river_poly):
        return False
    if getattr(water, "sea_poly", None) and segment_intersects_polygon(ray_a, best_pt, water.sea_poly):
        return False
    # split hit segment and add the straight extension
    _safe_split_segment_at_point(roads, best, best_pt)
    ok = _safe_add_manual_segment(roads, ray_a, best_pt, level=level, water=None, min_spacing_override=0)
    return bool(ok)



def _ray_hit_visible_road(roads, water, start: Point, dir_vec: Point, avoid_seg=None, max_len: float = 800.0):
    """Cast a ray from start in dir_vec, return the first hit point on a visible, active road segment.
    Skips decorative & water-suppressed & our own bridge segs. Returns (hit_point or None)."""
    from geometry import seg_intersection, segment_intersects_polygon
    import math
    ux, uy = dir_vec
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux/L, uy/L
    a = start
    b = (start[0] + ux*max_len, start[1] + uy*max_len)
    best_t = 1e9; best_pt = None; best_seg = None
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())
    for s in roads.segments:
        if not getattr(s, "active", True): continue
        if s is avoid_seg: continue
        if s in _BRIDGES: continue
        if s in deco or s in river_skip or s in sea_skip: continue
        ok, P, t_ray, t_seg = seg_intersection(a, b, s.a, s.b)
        if not ok or P is None or t_ray is None: continue
        if t_ray < 1e-4 or t_ray > 1.0: continue
        if t_seg is None or t_seg < 0.0 or t_seg > 1.0: continue
        if t_ray < best_t:
            best_t, best_pt, best_seg = t_ray, P, s
    if best_pt is None:
        return None
    # ensure the straight path stays on land
    if getattr(water, "river_poly", None) and segment_intersects_polygon(a, best_pt, water.river_poly):
        return None
    if getattr(water, "sea_poly", None) and segment_intersects_polygon(a, best_pt, water.sea_poly):
        return None
    _safe_split_segment_at_point(roads, best_seg, best_pt)
    return best_pt


def _ray_first_hit_on_road(roads, water, start: Point, dir_vec: Point, avoid_seg=None, max_len: float = 1200.0):
    """Return the first intersection point of a ray with any *visible* road segment, or None.
    Does NOT modify the roads graph; overlay-only logic. Skips decorative & water-suppressed & our bridge segs."""
    from geometry import seg_intersection, segment_intersects_polygon
    import math
    ux, uy = dir_vec
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux/L, uy/L
    a = start
    b = (start[0] + ux*max_len, start[1] + uy*max_len)
    best_t = 1e9; best_pt = None
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())
    for s in roads.segments:
        if not getattr(s, "active", True): continue
        if s is avoid_seg: continue
        if s in _BRIDGES: continue
        if s in deco or s in river_skip or s in sea_skip: continue
        ok, P, t_ray, t_seg = seg_intersection(a, b, s.a, s.b)
        if not ok or P is None or t_ray is None: continue
        if t_ray < 1e-4 or t_ray > 1.0: continue
        if t_seg is None or t_seg < 0.0 or t_seg > 1.0: continue
        if t_ray < best_t:
            best_t, best_pt = t_ray, P
    if best_pt is None:
        return None
    # allow approach to cross land only; reject path that crosses sea (river ok since we start inside it)
    if getattr(water, "sea_poly", None) and segment_intersects_polygon(a, best_pt, water.sea_poly):
        return None
    return best_pt


def generate_bridges_uniform(roads, water, desired: int = 6, target_water_span: float = 120.0, max_approach_len: float = 1000.0, level: int = 1) -> int:
    """Evenly space `desired` bridges along the river centerline.
    Each bridge draws a UNIFORM water-only deck (length ~= target_water_span, clipped by the channel),
    and adds straight 'approach' lines in this overlay that extend until they hit the road network.
    No changes are made to the actual road graph."""
    global _BRIDGE_GEOMS, _APPROACH_GEOMS
    if not getattr(water, 'river_poly', None) or len(water.river_poly) < 3:
        return 0
    rp = water.river_poly
    left, right = _split_left_right(rp)
    samples = max(200, int((_poly_length(rp) if '_poly_length' in globals() else len(rp)*4) / 4))
    center = _centerline_from_banks(left, right, samples=samples)
    if len(center) < 8:
        return 0
    cum = _arclen(center)
    total = cum[-1] if cum else 0.0
    if total <= 1e-6:
        return 0
    step = total / (desired + 1)
    targets = [step * k for k in range(1, desired + 1)]
    # cumulative lengths along center
    cum = cum
    def idx_at_len(t):
        lo, hi = 0, len(cum)-1
        while lo < hi:
            mid = (lo+hi)//2
            if cum[mid] < t: lo = mid+1
            else: hi = mid
        return max(1, min(len(center)-2, lo))
    placed = 0
    for t in targets:
        i = idx_at_len(t)
        # local normal at center[i]
        vx, vy = (center[i+1][0]-center[i-1][0], center[i+1][1]-center[i-1][1])
        L = (vx*vx+vy*vy)**0.5 or 1.0
        tx, ty = (vx/L, vy/L)
        nx, ny = (ty, -tx)
        endpoints = _bridge_endpoints_through_polygon(center[i], (nx, ny), rp, land_offset=0.0, min_channel_width=1.0)
        if not endpoints:
            # fallback: shoot a long normal through the polygon and take the middle pair
            R = 99999.0
            a_far = (center[i][0] - nx*R, center[i][1] - ny*R)
            b_far = (center[i][0] + nx*R, center[i][1] + ny*R)
            hits = []
            n = len(rp)
            for k in range(n):
                c = rp[k]; d = rp[(k+1)%n]
                ok, P, ta, _ = seg_intersection(a_far, b_far, c, d)
                if ok and P is not None and 0.0 <= ta <= 1.0:
                    hits.append((ta, P))
            hits.sort(key=lambda x: x[0])
            if len(hits) < 2:
                continue
            mid = len(hits)//2 - 1
            p0, p1 = hits[mid][1], hits[mid+1][1]
        else:
            p0, p1 = endpoints
        # center point and uniform span centered within channel
        mx, my = ((p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5)
        lenw = ((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2)**0.5 or 1.0
        uxw, uyw = ((p1[0]-p0[0])/lenw, (p1[1]-p0[1])/lenw)
        # make sure direction agrees with +normal
        if uxw*nx + uyw*ny < 0.0:
            uxw, uyw = -uxw, -uyw
        half = min(target_water_span*0.5, lenw*0.5)
        A = (mx - uxw*half, my - uyw*half)
        B = (mx + uxw*half, my + uyw*half)
        MW, MH = getattr(roads,'MAP_WIDTH', None), getattr(roads,'HEIGHT', None)
        if MW is None or MH is None or (_inside_map_rect(A, MW, MH) and _inside_map_rect(B, MW, MH)):
            _BRIDGE_GEOMS.append((A, B, level))
        # approaches: extend along deck dir until we hit road network (overlay-only)
        # approaches: extend along deck dir until we hit road network (overlay-only)
        hit_L = _ray_kth_hit_on_road(roads, water, A, (-uxw, -uyw), max_len=max_approach_len)
        hit_R = _ray_kth_hit_on_road(roads, water, B, ( uxw,  uyw), max_len=max_approach_len)
        if hit_L is not None:
            _add_curve_from_hit(A, (-uxw, -uyw), hit_L, roads, level)
        if hit_R is not None:
            _add_curve_from_hit(B, ( uxw,  uyw), hit_R, roads, level)

    return placed



def _ray_kth_hit_on_road(roads, water, start: Point, dir_vec: Point, avoid_seg=None, max_len: float = 2000.0, skip: int = 1, tube_tol: float = 10.0):
    """Return the kth (skip-th) intersection point along a ray with *visible* road segments.
    skip=0 -> first hit, skip=1 -> second hit, etc. Skips decorative, water-suppressed, and bridge segs."""
    from geometry import seg_intersection, segment_intersects_polygon
    import math
    ux, uy = dir_vec
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux / L, uy / L
    a = start
    b = (start[0] + ux * max_len, start[1] + uy * max_len)
    hits = []
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())
    for s in roads.segments:
        if not getattr(s, "active", True):
            continue
        if s is avoid_seg:
            continue
        if s in _BRIDGES:
            continue
        if s in deco or s in river_skip or s in sea_skip:
            continue
        ok, P, t_ray, t_seg = seg_intersection(a, b, s.a, s.b)
        if not ok or P is None or t_ray is None:
            continue
        if t_ray < 1e-4 or t_ray > 1.0:
            continue
        if t_seg is None or t_seg < 0.0 or t_seg > 1.0:
            pass
        else:
            hits.append((t_ray, P))
        # --- collinear / near-collinear fallback: if the segment runs almost along the ray,
        # and sits within a small lateral tube, treat the entrance point as a hit.
        # Compute lateral distance of both endpoints to the ray and choose smallest positive projection.
        import math
        # ray unit direction
        ux, uy = (b[0]-a[0], b[1]-a[1]); denom = math.hypot(ux, uy) or 1.0; ux/=denom; uy/=denom
        # function to compute signed lateral distance and forward t (in [0, max_len])
        def _lat_t(Pt):
            vx, vy = (Pt[0]-a[0], Pt[1]-a[1])
            t = vx*ux + vy*uy
            lat = abs(vx*(-uy) + vy*(ux))
            return lat, t
        lat1, t1 = _lat_t(s.a)
        lat2, t2 = _lat_t(s.b)
        # if either endpoint is inside the tube and ahead on the ray, add as candidate
        cands = []
        if lat1 <= tube_tol and 1e-3 <= t1 <= max_len:
            cands.append(t1/max_len)
        if lat2 <= tube_tol and 1e-3 <= t2 <= max_len:
            cands.append(t2/max_len)
        if cands:
            tmin = min(cands)
            # reconstruct point on ray
            Px = a[0] + ux * (tmin*max_len)
            Py = a[1] + uy * (tmin*max_len)
            # SEA crossing check for the path a->P
            from geometry import segment_intersects_polygon
            if getattr(water, 'sea_poly', None) and segment_intersects_polygon(a, (Px, Py), water.sea_poly):
                pass
            else:
                hits.append((tmin, (Px, Py)))
    if not hits:
        return None
    hits.sort(key=lambda x: x[0])
    # pick kth hit if available
    k = skip
    if k < len(hits):
        Pt = hits[k][1]
        # reject if path crosses SEA (land allowed; river is fine from start inside)
        if getattr(water, "sea_poly", None):
            if segment_intersects_polygon(a, Pt, water.sea_poly):
                return None
        return Pt
    return None


def _ray_hits_sorted(roads, water, start: Point, dir_vec: Point, max_len: float = 4000.0, avoid_seg=None):
    """Return all intersections of a ray with visible road segments as [(t_ray, P), ...] sorted by t_ray.
    t_ray is 0..1 along the segment [start, start+dir*max_len]."""
    from geometry import seg_intersection
    import math
    ux, uy = dir_vec
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux/L, uy/L
    a = start
    b = (start[0] + ux*max_len, start[1] + uy*max_len)
    hits = []
    deco = getattr(roads, "decorative_segments", set())
    river_skip = getattr(roads, "river_skip_segments", set())
    sea_skip = getattr(roads, "sea_skip_segments", set())
    for s in roads.segments:
        if not getattr(s, "active", True): continue
        if s is avoid_seg: continue
        if s in _BRIDGES: continue
        if s in deco or s in river_skip or s in sea_skip: continue
        ok, P, t_ray, t_seg = seg_intersection(a, b, s.a, s.b)
        if not ok or P is None or t_ray is None: continue
        if t_seg is None or t_seg < 0.0 or t_seg > 1.0: continue
        if t_ray < 1e-4 or t_ray > 1.0: continue
        hits.append((t_ray, P))
    hits.sort(key=lambda x: x[0])
    return hits



def _sample_along_ray_find_road(roads, water, start: Point, dir_vec: Point, *, step: float = 80.0, steps: int = 80, tube: float = 36.0):
    """Walk forward along a straight ray from `start` and return the first projection point onto a
    *visible* road segment within `tube` distance. Returns the snapped point or None. Overlay-only.
    The path from start->hit must not cross SEA (river crossing is allowed)."""
    from geometry import point_seg_dist, segment_intersects_polygon
    import math
    ux, uy = dir_vec
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux/L, uy/L
    deco = getattr(roads, 'decorative_segments', set())
    river_skip = getattr(roads, 'river_skip_segments', set())
    sea_skip = getattr(roads, 'sea_skip_segments', set())
    for k in range(1, max(1, int(steps))+1):
        px = start[0] + ux * (k * step)
        py = start[1] + uy * (k * step)
        best = None; best_d = 1e9; best_proj = None
        for s in roads.segments:
            if not getattr(s, 'active', True):
                continue
            if s in _BRIDGES:
                continue
            if s in deco or s in river_skip or s in sea_skip:
                continue
            d, proj, t = point_seg_dist((px, py), s.a, s.b)
            if t is None:
                continue
            if d < best_d:
                best_d = d; best = s; best_proj = proj
        if best is not None and best_proj is not None and best_d <= tube:
            if getattr(water, 'sea_poly', None) and segment_intersects_polygon(start, best_proj, water.sea_poly):
                return None
            return best_proj
    return None

def _ray_hit_with_seg(roads, water, start: Point, dir_vec: Point, max_len: float = 2000.0, tube: float = 10.0):
    """
    Return (point, segment, t_on_seg) where the ray from `start` along `dir_vec` first meets a visible
    road segment's tube. Avoid decorative, skipped, and bridge segments.
    """
    import math
    ux, uy = dir_vec
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux / L, uy / L
    a = start
    b = (start[0] + ux * max_len, start[1] + uy * max_len)
    deco = getattr(roads, 'decorative_segments', set())
    river_skip = getattr(roads, 'river_skip_segments', set())
    sea_skip = getattr(roads, 'sea_skip_segments', set())
    hits = []
    for s in roads.segments:
        if not getattr(s, 'active', True): 
            continue
        if s in deco or s in river_skip or s in sea_skip:
            continue
        if hasattr(roads, 'river_bridge_segments') and s in roads.river_bridge_segments:
            continue
        # treat proximity within a tube as a hit
        from geometry import point_seg_dist, segment_intersects_polygon
        d, proj, t = point_seg_dist(a, s.a, s.b)
        # project ray parameter to the projection point
        vx, vy = proj[0]-a[0], proj[1]-a[1]
        ray_t = (vx*ux + vy*uy)
        lat  = abs(vx*(-uy) + vy*(ux))
        if ray_t >= 1e-3 and lat <= tube:
            # ensure travelling path doesn't cross sea
            P = (a[0] + ux*ray_t, a[1] + uy*ray_t)
            if getattr(water, 'sea_poly', None) and segment_intersects_polygon(a, P, water.sea_poly):
                continue
            hits.append((ray_t, P, s, t))
    if not hits:
        return None
    hits.sort(key=lambda x: x[0])
    return hits[0]

def _add_curve_from_hit(start_pt: Point, deck_dir: Point, hit_pt: Point, roads, level: int):
    # Find the segment and its direction at hit_pt to compute a smooth blend curve
    from geometry import point_seg_dist
    best=None; bestd=1e9; bestseg=None; bestt=None
    for s in roads.segments:
        if not getattr(s, 'active', True):
            continue
        d, proj, t = point_seg_dist(hit_pt, s.a, s.b)
        if t is None: 
            continue
        dd = (proj[0]-hit_pt[0])**2 + (proj[1]-hit_pt[1])**2
        if dd < bestd:
            bestd=dd; bestseg=s; best=proj; bestt=t
    if bestseg is None:
        return
    # directions
    import math
    ux, uy = deck_dir
    L = math.hypot(ux, uy) or 1.0
    ux, uy = ux/L, uy/L
    vx, vy = (bestseg.b[0]-bestseg.a[0], bestseg.b[1]-bestseg.a[1])
    Lv = math.hypot(vx, vy) or 1.0
    vx, vy = vx/Lv, vy/Lv
    # Estimate curve length and handle lengths
    dx, dy = hit_pt[0]-start_pt[0], hit_pt[1]-start_pt[1]
    dlen = math.hypot(dx, dy)
        # Handle lengths tuned for smoother merge
    try:
        road_w = float(roads.params['ROAD_WIDTH'][level])
    except Exception:
        road_w = 4.0
    hmin = road_w * 2.0
    hmax = 240.0
    h1 = max(hmin, min(hmax, dlen * 0.5))
    h2 = max(hmin, min(hmax, dlen * 0.5))
    # Construct cubic Bezier control points ensuring tangency at both ends
    C0 = start_pt
    C1 = (start_pt[0] + ux*h1, start_pt[1] + uy*h1)
    C3 = hit_pt
    C2 = (hit_pt[0] - vx*h2,  hit_pt[1] - vy*h2)
    # Avoid curves that cross water; shrink handles if needed
    def _curve_points(C0,C1,C2,C3,S=20):
        for i in range(S+1):
            t = i/float(S); u = 1.0 - t
            x = (u*u*u*C0[0] + 3*u*u*t*C1[0] + 3*u*t*t*C2[0] + t*t*t*C3[0])
            y = (u*u*u*C0[1] + 3*u*u*t*C1[1] + 3*u*t*t*C2[1] + t*t*t*C3[1])
            yield (x,y)
    def _curve_crosses_water(C0,C1,C2,C3):
        rp = getattr(roads, 'water', None) or getattr(globals(), 'water', None)
        river = getattr(rp, 'river_poly', None) if rp else None
        sea   = getattr(rp, 'sea_poly', None) if rp else None
        try:
            from geometry import point_in_poly
        except Exception:
            point_in_poly = None
        for pt in _curve_points(C0,C1,C2,C3, S=22):
            if river and point_in_poly and point_in_poly(pt, river): return True
            if sea and point_in_poly and point_in_poly(pt,   sea): return True
        return False
    # shrink up to 3 times if it crosses water
    tries = 3
    _C1, _C2 = C1, C2
    while tries>0 and _curve_crosses_water(C0, _C1, _C2, C3):
        _C1 = (C0[0] + ( _C1[0]-C0[0]) * 0.7, C0[1] + ( _C1[1]-C0[1]) * 0.7)
        _C2 = (C3[0] + ( _C2[0]-C3[0]) * 0.7, C3[1] + ( _C2[1]-C3[1]) * 0.7)
        tries -= 1
    _APPROACH_CURVES.append(((C0, _C1, _C2, C3), level))

def _inside_map_rect(pt, w, h):
    x,y = pt
    return 0.0 <= x <= float(w) and 0.0 <= y <= float(h)

def _clamp_point_to_rect(pt, w, h):
    x,y = pt
    return (max(0.0, min(float(w), x)), max(0.0, min(float(h), y)))

def _clip_segment_to_rect(a, b, w, h):
    # Liang-Barsky style clipping for axis-aligned rectangle [0,w]x[0,h]
    x0,y0=a; x1,y1=b
    dx=x1-x0; dy=y1-y0
    p=[-dx, dx, -dy, dy]
    q=[x0-0, w-x0, y0-0, h-y0]
    u0=0.0; u1=1.0
    for pi,qi in zip(p,q):
        if abs(pi) < 1e-12:
            if qi < 0: 
                return None
        else:
            t = -qi/pi
            if pi < 0:
                if t > u1: return None
                if t > u0: u0 = t
            else:
                if t < u0: return None
                if t < u1: u1 = t
    return (x0+u0*dx, y0+u0*dy), (x0+u1*dx, y0+u1*dy)
# ----------------------- public API: overlay (ABOVE water) -----------------------
def _segment_inside_water_portion(a: Point, b: Point, river_poly: List[Point]) -> Optional[Tuple[Point, Point]]:
    """Return the portion of segment AB that lies over water (intersecting the polygon).
    If it doesn't intersect, return None."""
    hits: List[Tuple[float, Point]] = []
    n = len(river_poly)
    for i in range(n):
        c = river_poly[i]; d = river_poly[(i+1)%n]
        ok, P, t, _ = seg_intersection(a,b,c,d)
        if ok and P is not None and 0.0<=t<=1.0:
            hits.append((t,P))
    # Also consider if endpoints are inside
    inside_a = point_in_poly(a, river_poly)
    inside_b = point_in_poly(b, river_poly)
    if inside_a: hits.append((0.0, a))
    if inside_b: hits.append((1.0, b))
    if len(hits) < 2: return None
    hits.sort(key=lambda x:x[0])
    p = hits[0][1]; q = hits[-1][1]
    return p,q


def draw_overlay(screen, world_to_screen, cam_zoom, roads, water,
                 deck_color=(225,225,228),
                 edge_color=(80,80,80),
                 rail_thickness: float = 2.0,
                 pier_spacing: float = 28.0,
                 pier_thickness: float = 4.0):
    """Draw bridges ABOVE the river with decks extended across the green/bank area.
    - Deck starts at the river edge and extends to the first visible road on each bank.
    - Piers are only drawn over water.
    - Approaches (road-looking) start exactly at the extended deck ends and are drawn in the bridge overlay.
    """
    import pygame
    if not getattr(water, 'river_poly', None):
        return
    river_poly = water.river_poly

    def to_screen(poly): return [world_to_screen(p) for p in poly]

    # collect geoms to draw
    geoms = list(_BRIDGE_GEOMS)
    # include preview geom (if any) as a special marker.  We draw
    # preview bridges in a lighter color.  We store preview separately
    # so that committed bridges remain unaffected.
    preview_geom = None
    from inspect import isclass
    global _PREVIEW_GEOM
    if _PREVIEW_GEOM is not None:
        preview_geom = _PREVIEW_GEOM
    # To prevent lasso-like connectors from being drawn, we clear the
    # approach geometry lists before any drawing occurs.  The bridge
    # overlay originally stored straight and curved "approach" segments
    # in `_APPROACH_GEOMS` and `_APPROACH_CURVES` which attempted to
    # visually reconnect bridges back to the road network.  With the
    # regeneration logic in city_weaver.py that rebuilds the
    # road network to align with bridge axes, these connectors are
    # unnecessary and visually distracting.  Clearing the lists here
    # ensures that no approach segments or curves are drawn.
    global _APPROACH_GEOMS, _APPROACH_CURVES
    _APPROACH_GEOMS = []
    _APPROACH_CURVES = []
    if not geoms:
        # If there are no bridge decks to draw, exit early.  We ignore
        # cleared approach lists because those are intentionally empty.
        return

    # width parameters
    try:
        road_w = float(roads.params['ROAD_WIDTH'][1])
    except Exception:
        road_w = 4.0
    deck_w = max(road_w*1.25, 6.0)

    # endpoint mapping for approaches
    end_shift = {}

    # --- draw each bridge: piers then extended deck ---
    def _draw_bridge(a, b, level, is_preview=False):
        # compute water-only portion along the stored segment (a->b)
        water_span = _segment_inside_water_portion(a, b, river_poly)
        if water_span is None:
            return
        p0, p1 = water_span  # these lie on water boundary

        # --- piers over water only ---
        ux, uy = (p1[0]-p0[0], p1[1]-p0[1]); Lw = (ux*ux+uy*uy)**0.5 or 1.0
        ux, uy = ux/Lw, uy/Lw
        nx, ny = (uy, -ux)
        # skip piers for preview to avoid visual clutter
        if not is_preview:
            if Lw > 24.0:
                npiers = int(Lw // pier_spacing)
                if npiers >= 1:
                    for i in range(1, npiers+1):
                        t = i / (npiers+1)
                        cx = p0[0] + ux * (t * Lw); cy = p0[1] + uy * (t * Lw)
                        w = pier_thickness * 0.5
                        hh = max(6.0, road_w*0.35)
                        poly = [
                            (cx - nx*w, cy - ny*w),
                            (cx + nx*w, cy + ny*w),
                            (cx + nx*w + ux*hh, cy + ny*w + uy*hh),
                            (cx - nx*w + ux*hh, cy - ny*w + uy*hh),
                        ]
                        pygame.draw.polygon(screen, (160,160,165), to_screen(poly))

        # --- extend deck from water edges out to the first visible road on each side ---
        # directions outward relative to the long axis
        Lseg = ((b[0]-a[0])**2 + (b[1]-a[1])**2) ** 0.5 or 1.0
        tx, ty = ((b[0]-a[0]) / Lseg, (b[1]-a[1]) / Lseg)
        # determine which end of (a,b) corresponds to p0 along (a->b)
        def proj_t(P): return ((P[0]-a[0]) * tx + (P[1]-a[1]) * ty) / Lseg
        if proj_t(p0) > proj_t(p1):
            p0, p1 = p1, p0
        # cast rays from p0 back toward a, and from p1 forward toward b
        max_len = 4000.0
        hits0 = _ray_hits_sorted(roads, water, p0, (-tx, -ty), max_len=max_len)
        hits1 = _ray_hits_sorted(roads, water, p1, ( tx,  ty), max_len=max_len)
        margin = max(10.0, road_w*0.6)
        def pick_ext(hits):
            if hits:
                return max(48.0, min(max_len*hits[0][0] - margin, 420.0))
            return 96.0
        ext0 = pick_ext(hits0); ext1 = pick_ext(hits1)
        EXT_MULT = 1.0  # neutralized; final length controlled by BRIDGE_LENGTH_SCALE  # slightly shorter (~-6%)  # subtle extension (~12%)  # make bridges a bit longer on each side
        ext0 *= EXT_MULT; ext1 *= EXT_MULT
        p0e = (p0[0] - tx*ext0, p0[1] - ty*ext0)
        p1e = (p1[0] + tx*ext1, p1[1] + ty*ext1)
        MW = getattr(roads,'MAP_WIDTH', None); MH = getattr(roads,'HEIGHT', None)
        if MW is not None and MH is not None:
            clipped = _clip_segment_to_rect(p0e, p1e, MW, MH)
            if clipped:
                p0e, p1e = clipped
        # Apply symmetric visual scaling regardless of clipping
        scale = BRIDGE_LENGTH_SCALE_DEFAULT
        if scale != 1.0:
            cx = 0.5 * (p0e[0] + p1e[0]); cy = 0.5 * (p0e[1] + p1e[1])
            p0e = (cx + (p0e[0]-cx)*scale, cy + (p0e[1]-cy)*scale)
            p1e = (cx + (p1e[0]-cx)*scale, cy + (p1e[1]-cy)*scale)



        # record mapping so approach starts use extended ends
        da0 = ((a[0]-p0[0])**2 + (a[1]-p0[1])**2)**0.5
        db0 = ((b[0]-p0[0])**2 + (b[1]-p0[1])**2)**0.5
        if da0 <= db0:
            end_shift[a] = p0e; end_shift[b] = p1e
        else:
            end_shift[a] = p1e; end_shift[b] = p0e

        # draw the extended deck body and rails
        
        deck_poly = road_tube_poly(p0e, p1e, deck_w)
        # choose color: normal or preview
        poly_color = deck_color
        line_color = edge_color
        if is_preview:
            # lighten deck and edges for preview
            poly_color = tuple(min(255, int(c*1.05)) for c in deck_color)
            line_color = tuple(min(255, int(c*1.05)) for c in edge_color)

        # --- stylish rendering ---
        # 1) soft "shadow" (simple offset polygon) for depth
        try:
            pts_screen = to_screen(deck_poly)
            shadow_pts = [(x+2, y+2) for (x, y) in pts_screen]
            pygame.draw.polygon(screen, (50,50,60), shadow_pts)
        except Exception:
            pts_screen = to_screen(deck_poly)

        # 2) fill deck
        pygame.draw.polygon(screen, poly_color, pts_screen)

        # 3) rounded end caps for a pill-shaped deck
        #    (draw circles at both ends with radius = deck_w/2)
        r = max(2, int(0.5 * deck_w))
        pygame.draw.circle(screen, poly_color, world_to_screen(p0e), r)
        pygame.draw.circle(screen, poly_color, world_to_screen(p1e), r)

        # 4) outer outline (edge) and subtle inner highlight
        pygame.draw.polygon(screen, line_color, pts_screen, max(1, int(1*cam_zoom)))
        # outline the caps
        pygame.draw.circle(screen, line_color, world_to_screen(p0e), r, max(1, int(1*cam_zoom)))
        pygame.draw.circle(screen, line_color, world_to_screen(p1e), r, max(1, int(1*cam_zoom)))

        # subtle inner highlight line along the deck center
        cx0, cy0 = world_to_screen(p0e); cx1, cy1 = world_to_screen(p1e)
        try:
            pygame.draw.line(screen, (245,245,248), (cx0, cy0), (cx1, cy1), max(1, int(1*cam_zoom)))
        except Exception:
            pass

        # 5) minimal posts/rail accents along edges (skip for preview to reduce clutter)
        if not is_preview:
            # compute edge vectors in world space
            ux2, uy2 = (p1e[0]-p0e[0], p1e[1]-p0e[1]); L2 = (ux2*ux2+uy2*uy2)**0.5 or 1.0
            ux2, uy2 = ux2/L2, uy2/L2
            nx2, ny2 = (uy2, -ux2)
            # posts every ~36 world units
            step = 36.0
            nposts = int(L2 // step)
            post_h = max(2, int(0.25*deck_w))  # short ticks
            for i in range(1, nposts):
                t = i * step / L2
                px = p0e[0] + ux2 * t * L2
                py = p0e[1] + uy2 * t * L2
                # two small ticks near edges
                A0 = world_to_screen((px + nx2*(0.5*deck_w - 1.0), py + ny2*(0.5*deck_w - 1.0)))
                B0 = world_to_screen((px + nx2*(0.5*deck_w - 1.0 - post_h), py + ny2*(0.5*deck_w - 1.0 - post_h)))
                A1 = world_to_screen((px - nx2*(0.5*deck_w - 1.0), py - ny2*(0.5*deck_w - 1.0)))
                B1 = world_to_screen((px - nx2*(0.5*deck_w - 1.0 - post_h), py - ny2*(0.5*deck_w - 1.0 - post_h)))
                pygame.draw.line(screen, line_color, A0, B0, max(1, int(1*cam_zoom)))
                pygame.draw.line(screen, line_color, A1, B1, max(1, int(1*cam_zoom)))
        ux2, uy2 = (p1e[0]-p0e[0], p1e[1]-p0e[1]); L2 = (ux2*ux2+uy2*uy2)**0.5 or 1.0
        ux2, uy2 = ux2/L2, uy2/L2
        nx2, ny2 = (uy2, -ux2)
        inset = max(1.0, road_w*0.4)
        halfw = deck_w*0.5 - inset
        r1a = (p0e[0]+nx2*halfw, p0e[1]+ny2*halfw); r1b = (p1e[0]+nx2*halfw, p1e[1]+ny2*halfw)
        r2a = (p0e[0]-nx2*halfw, p0e[1]-ny2*halfw); r2b = (p1e[0]-nx2*halfw, p1e[1]-ny2*halfw)
        pygame.draw.line(screen, line_color, world_to_screen(r1a), world_to_screen(r1b), max(1, int(rail_thickness*cam_zoom)))
        pygame.draw.line(screen, line_color, world_to_screen(r2a), world_to_screen(r2b), max(1, int(rail_thickness*cam_zoom)))
    # end def _draw_bridge

    # draw committed bridges
    for (a, b, level) in geoms:
        _draw_bridge(a, b, level, is_preview=False)
    # draw preview bridge if present
    if preview_geom is not None:
        pa, pb, lvl = preview_geom
        _draw_bridge(pa, pb, lvl, is_preview=True)
        # Extra visibility markers for preview (does not affect final geometry)
        try:
            import pygame as _pg
            mid = ((pa[0]+pb[0])/2.0, (pa[1]+pb[1])/2.0)
            _pg.draw.circle(screen, (220, 50, 50), world_to_screen(mid), max(2, int(3*cam_zoom)), 0)
            for P in (pa, pb):
                _pg.draw.circle(screen, (30, 30, 30), world_to_screen(P), max(2, int(2*cam_zoom)), 1)
        except Exception:
            pass


    # --- after all decks, draw overlay approaches using extended starts ---
    if _APPROACH_GEOMS:
        for (p, q, lvl) in _APPROACH_GEOMS:
            try:
                road_w = float(roads.params['ROAD_WIDTH'][lvl])
            except Exception:
                road_w = 4.0
            start_p = end_shift.get(p, p)
            poly = road_tube_poly(start_p, q, road_w)
            pygame.draw.polygon(screen, (164,168,176), to_screen(poly))




    
    
# --- draw thick smoothed approach curves ---
    if _APPROACH_CURVES:
        import pygame as _pg
        for (C0, C1, C2, C3), lvl in _APPROACH_CURVES:
            try:
                w_world = float(roads.params['ROAD_WIDTH'][lvl])
            except Exception:
                w_world = 4.0
            # sample curve in WORLD space
            pts_world = []
            S = max(12, int((((C3[0]-C0[0])**2 + (C3[1]-C0[1])**2)**0.5) / 12))
            for i in range(S+1):
                t = i/float(S); u = 1.0 - t
                x = (u*u*u*C0[0] + 3*u*u*t*C1[0] + 3*u*t*t*C2[0] + t*t*t*C3[0])
                y = (u*u*u*C0[1] + 3*u*u*t*C1[1] + 3*u*t*t*C2[1] + t*t*t*C3[1])
                pts_world.append((x,y))
            # draw as short overlapping tubes to avoid self-intersection artifacts
            overlap = 0.05  # 5% overlap to hide seams
            for i in range(len(pts_world)-1):
                a = pts_world[i]
                b = pts_world[i+1]
                vx, vy = (b[0]-a[0], b[1]-a[1])
                L = (vx*vx+vy*vy)**0.5 or 1.0
                ux, uy = vx/L, vy/L
                a2 = (a[0] + ux * (overlap*L), a[1] + uy * (overlap*L))
                b2 = (b[0] - ux * (overlap*L), b[1] - uy * (overlap*L))
                MW = getattr(roads,'MAP_WIDTH', None); MH = getattr(roads,'HEIGHT', None)
                if MW is not None and MH is not None:
                    clipped = _clip_segment_to_rect(a2, b2, MW, MH)
                    if not clipped:
                        continue
                    a2, b2 = clipped
                poly = road_tube_poly(a2, b2, w_world)
                _pg.draw.polygon(screen, (164,168,176), to_screen(poly))
            # subtle line as an edge
def reset_bridges():
    """Public reset hook used by UI; clears overlay-only bridge data."""
    _clear_bridge_overlay_cache()

def cancel_preview_bridge():
    """Cancel any active preview bridge.  This does not modify the
    finalized bridge overlay state."""
    global _PREVIEW_GEOM
    _PREVIEW_GEOM = None

def commit_preview_bridge():
    """Commit the current preview bridge to the overlay, if it exists.
    After committing, the preview is cleared and its endpoints are
    appended to the list of overlay bridge geometries.  Returns True
    if a bridge was committed, False otherwise."""
    global _PREVIEW_GEOM, _BRIDGE_GEOMS
    if _PREVIEW_GEOM is not None:
        # Append endpoints to overlay geometry list
        a, b, lvl = _PREVIEW_GEOM
        _BRIDGE_GEOMS.append((a, b, lvl))
        _PREVIEW_GEOM = None
        return True
    return False

# ---------------------------------------------------------------------------
# Extended preview bridge helpers with road-heading blending and span extension

def _find_nearest_road_heading(pos: Point, roads, max_dist: float = 200.0):
    """
    Find the direction (unit vector) of the road segment nearest to a given point.
    The search considers all segments in the road network (including truncated
    river-crossing segments) and returns the heading of the closest segment
    whose perpendicular distance to ``pos`` is less than ``max_dist``.  If
    no suitable segment is found, returns None.

    Parameters
    ----------
    pos : Point
        World-space coordinate used to query nearby segments.
    roads : RoadSystem
        The road network containing a ``segments`` list.
    max_dist : float, optional
        Maximum allowed perpendicular distance (world units) to consider a
        segment as a candidate.

    Returns
    -------
    tuple or None
        (ux, uy) unit vector along the heading of the nearest segment, or
        None if no segment is within ``max_dist``.
    """
    try:
        segs = getattr(roads, 'segments', [])
    except Exception:
        return None
    best_dist2 = (max_dist * max_dist)
    best_dir = None
    for s in segs:
        try:
            a = s.a; b = s.b
        except Exception:
            continue
        vx = b[0] - a[0]
        vy = b[1] - a[1]
        L2 = vx * vx + vy * vy
        if L2 <= 1e-9:
            continue
        # project pos onto the segment
        wx = pos[0] - a[0]
        wy = pos[1] - a[1]
        t = (wx * vx + wy * vy) / L2
        if t < 0.0:
            proj = a
        elif t > 1.0:
            proj = b
        else:
            proj = (a[0] + vx * t, a[1] + vy * t)
        dx = pos[0] - proj[0]
        dy = pos[1] - proj[1]
        dist2 = dx * dx + dy * dy
        if dist2 < best_dist2:
            best_dist2 = dist2
            # compute unit direction along the segment
            L = (L2) ** 0.5
            if L > 1e-6:
                best_dir = (vx / L, vy / L)
    return best_dir

def start_bridge_preview_blend(pos: Point, water, roads=None, level: int = 1, angle_blend: float = 0.0) -> bool:
    """
    Like :func:`start_bridge_preview`, but allows blending the bridge direction
    between the river normal and the heading of the nearest road segment.  The
    ``angle_blend`` factor controls the interpolation; 0.0 uses only the
    river normal, 1.0 uses only the road heading.  The resulting bridge
    segment is also extended on both ends by half its span length.

    Parameters
    ----------
    pos : Point
        World-space cursor location.
    water : Water
        Water object containing ``river_poly``.
    roads : RoadSystem, optional
        Road network for deriving a nearby road heading.
    level : int, optional
        Level assigned to the preview bridge.
    angle_blend : float, optional
        Blend factor between river normal (0) and road heading (1).

    Returns
    -------
    bool
        True if a preview bridge was created, False otherwise.
    """
    global _PREVIEW_GEOM
    river = getattr(water, 'river_poly', None)
    if not river or len(river) < 6:
        _PREVIEW_GEOM = None
        return False
    nearest = _nearest_river_segment(pos, river)
    if not nearest:
        _PREVIEW_GEOM = None
        return False
    center, seg_idx, river_normal = nearest
    # attempt to find nearby road heading
    road_dir = None
    if roads is not None and angle_blend > 0.0:
        road_dir = _find_nearest_road_heading(center, roads)
    normal = river_normal
    if road_dir is not None and angle_blend > 0.0:
        nx, ny = river_normal
        rx, ry = road_dir
        bx = nx * (1.0 - angle_blend) + rx * angle_blend
        by = ny * (1.0 - angle_blend) + ry * angle_blend
        L = (bx * bx + by * by) ** 0.5
        if L > 1e-6:
            normal = (bx / L, by / L)
    # find endpoints across the river
    endpoints = _bridge_endpoints_through_polygon(center, normal, river, land_offset=0.0, min_channel_width=1.0)
    if endpoints is None:
        endpoints = _preview_endpoints_from_point(pos, water, land_offset=0.0)
        if endpoints is None:
            _PREVIEW_GEOM = None
            return False
        a, b = endpoints
    else:
        a, b = endpoints
    # extend by half span
    ux = b[0] - a[0]
    uy = b[1] - a[1]
    L = (ux * ux + uy * uy) ** 0.5
    if L > 1e-6:
        ux_norm = ux / L
        uy_norm = uy / L
        ext = L * 0.5  # slightly extended preview span
        a_ext = (a[0] - ux_norm * ext, a[1] - uy_norm * ext)
        b_ext = (b[0] + ux_norm * ext, b[1] + uy_norm * ext)
    else:
        a_ext, b_ext = a, b
    _PREVIEW_GEOM = (a_ext, b_ext, level)
    return True

def update_bridge_preview_blend(pos: Point, water, roads=None, level: int = 1, angle_blend: float = 0.0) -> bool:
    """Update the blended preview bridge to a new cursor position.  See
    ``start_bridge_preview_blend`` for details.  Returns True if a
    preview exists, False otherwise."""
    return start_bridge_preview_blend(pos, water, roads=roads, level=level, angle_blend=angle_blend)

def _nearest_river_segment(point: Point, river_poly: List[Point]):
    """Return the nearest river segment and projection point for a given
    world-space point.  The result is a tuple (proj_point, seg_index,
    normal) where proj_point is the closest point on the segment,
    seg_index is the starting index of the segment in river_poly, and
    normal is a unit vector perpendicular to the segment pointing
    outward from the river center.  The normal orientation is chosen
    arbitrarily and may need to be flipped downstream.  If river_poly
    has fewer than 2 edges, returns None."""
    import math
    best_dist = float('inf')
    best_proj = None
    best_idx = None
    best_normal = None
    n = len(river_poly)
    if n < 2:
        return None
    for i in range(n):
        p0 = river_poly[i]
        p1 = river_poly[(i+1) % n]
        # vector along segment
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
        L2 = vx*vx + vy*vy
        if L2 <= 1e-9:
            continue
        # projection of point onto segment
        wx, wy = point[0] - p0[0], point[1] - p0[1]
        t = max(0.0, min(1.0, (wx*vx + wy*vy) / L2))
        proj = (p0[0] + vx * t, p0[1] + vy * t)
        dx, dy = (point[0] - proj[0], point[1] - proj[1])
        dist2 = dx*dx + dy*dy
        if dist2 < best_dist:
            best_dist = dist2
            best_proj = proj
            best_idx = i
            # normal perpendicular to segment
            seg_len = math.sqrt(L2)
            # (vy, -vx) gives a perpendicular; orientation chosen arbitrarily
            nx, ny = (vy / seg_len, -vx / seg_len)
            best_normal = (nx, ny)
    if best_proj is None or best_idx is None or best_normal is None:
        return None
    return best_proj, best_idx, best_normal

def _preview_endpoints_from_point(pos: Point, water, land_offset: float = 0.0) -> Optional[Tuple[Point, Point]]:
    """Compute bridge endpoints across the river for an arbitrary world-space
    point by projecting a perpendicular through the river polygon.  Uses
    the nearest river edge to determine a local normal direction, then
    shoots a long ray along +/-normal through the river polygon.  If a
    valid span is found, returns (a,b) endpoints on land (with optional
    land_offset).  Otherwise returns None."""
    river = getattr(water, 'river_poly', None)
    if not river or len(river) < 6:
        return None
    nearest = _nearest_river_segment(pos, river)
    if not nearest:
        return None
    center, seg_idx, normal = nearest
    # Use the local normal to cast through the polygon
    endpoints = _bridge_endpoints_through_polygon(center, normal, river, land_offset=land_offset, min_channel_width=1.0)
    if endpoints:
        return endpoints
    # Fallback: similar to _bridge_endpoints_through_polygon, but choose the
    # two nearest intersections around the center if the direct approach
    # fails (e.g. concave or narrow sections).  Cast a long ray.
    import math
    R = 100000.0
    a = (center[0] - normal[0] * R, center[1] - normal[1] * R)
    b = (center[0] + normal[0] * R, center[1] + normal[1] * R)
    hits = []
    n = len(river)
    for i in range(n):
        p0 = river[i]; p1 = river[(i+1) % n]
        ok, P, t, _ = seg_intersection(a, b, p0, p1)
        if ok and P is not None and t is not None and 0.0 <= t <= 1.0:
            hits.append((t, P))
    if len(hits) < 2:
        return None
    hits.sort(key=lambda x: x[0])
    # pick the pair closest to center (around t=0.5)
    best = None; best_gap = float('inf')
    for i in range(len(hits) - 1):
        t0, _ = hits[i]; t1, _ = hits[i+1]
        # mid t of this interval
        mid_t = 0.5 * (t0 + t1)
        # distance from mid to 0.5 (center of the ray)
        gap = abs(mid_t - 0.5)
        if gap < best_gap:
            best_gap = gap
            best = (hits[i][1], hits[i+1][1])
    if best is None:
        return None
    p_in, q_in = best
    # optionally push onto land by land_offset
    if land_offset > 0.0:
        ux, uy = normal
        # ensure normal points outward from the polygon at p_in (flip if necessary)
        # test a point slightly along normal; if it's inside river, flip normal
        from geometry import point_in_poly
        test = (p_in[0] + ux * 0.5, p_in[1] + uy * 0.5)
        if point_in_poly(test, river):
            ux, uy = -ux, -uy
        p = (p_in[0] - ux * land_offset, p_in[1] - uy * land_offset)
        q = (q_in[0] + ux * land_offset, q_in[1] + uy * land_offset)
        return p, q
    return p_in, q_in

def start_bridge_preview(pos: Point, water, roads=None, level: int = 1, angle_blend: float = 0.5) -> bool:
    """Begin a new preview bridge at the given world-space position.  The
    preview endpoints are computed by blending the river normal with
    the heading of the nearest road.  ``angle_blend`` controls the
    weight of the road orientation (0.0 = strictly perpendicular,
    1.0 = strictly follow road heading).  If ``roads`` is None, the
    river normal is used exclusively.

    Any existing preview is replaced.  Returns True if a preview was
    successfully created.
    """
    global _PREVIEW_GEOM
    # obtain river polygon
    river = getattr(water, 'river_poly', None)
    if not river or len(river) < 6:
        _PREVIEW_GEOM = None
        return False
    # find nearest point on river and its normal
    nearest = _nearest_river_segment(pos, river)
    if not nearest:
        _PREVIEW_GEOM = None
        return False
    center_pt, seg_idx, river_normal = nearest
    import math
    # compute road heading if roads provided
    if roads is not None and angle_blend > 0.0:
        try:
            # try to snap to nearest road segment
            proj, seg, t = roads.try_snap_segment(pos)
            if seg is not None:
                dx, dy = (seg.b[0] - seg.a[0], seg.b[1] - seg.a[1])
                L = math.hypot(dx, dy) or 1.0
                road_dir = (dx / L, dy / L)
                # blend river normal and road direction
                w = max(0.0, min(1.0, angle_blend))
                ux = (1.0 - w) * river_normal[0] + w * road_dir[0]
                uy = (1.0 - w) * river_normal[1] + w * road_dir[1]
                L2 = math.hypot(ux, uy) or 1.0
                normal = (ux / L2, uy / L2)
            else:
                normal = river_normal
        except Exception:
            normal = river_normal
    else:
        normal = river_normal
    # Compute endpoints across river along blended normal
    endpoints = _preview_endpoints_from_point(pos, water, land_offset=0.0)
    # If we had a road blend, refine endpoints by using the blended normal
    if endpoints is None or (roads is not None and angle_blend > 0.0):
        try:
            endpoints2 = _bridge_endpoints_through_polygon(center_pt, normal, river, land_offset=0.0, min_channel_width=1.0)
            if endpoints2:
                endpoints = endpoints2
            else:
                # fallback: shoot long ray along normal and pick two intersections
                R = 100000.0
                a = (center_pt[0] - normal[0] * R, center_pt[1] - normal[1] * R)
                b = (center_pt[0] + normal[0] * R, center_pt[1] + normal[1] * R)
                hits = []
                for i in range(len(river)):
                    p0 = river[i]; p1 = river[(i+1) % len(river)]
                    ok, P, t_line, _ = seg_intersection(a, b, p0, p1)
                    if ok and P is not None and t_line is not None and 0.0 <= t_line <= 1.0:
                        hits.append((t_line, P))
                if len(hits) >= 2:
                    hits.sort(key=lambda x: x[0])
                    best = None; best_gap = float('inf')
                    for i in range(len(hits)-1):
                        t0 = hits[i][0]; t1 = hits[i+1][0]
                        gap = abs(0.5 * (t0 + t1) - 0.5)
                        if gap < best_gap:
                            best_gap = gap; best = (hits[i][1], hits[i+1][1])
                    if best is not None:
                        endpoints = best
        except Exception:
            pass
    if endpoints is None:
        _PREVIEW_GEOM = None
        return False
    a_pt, b_pt = endpoints
    # Expand bridge length by half its span on both sides
    try:
        ux, uy = (b_pt[0] - a_pt[0], b_pt[1] - a_pt[1])
        span = math.hypot(ux, uy) or 1.0
        # half span extension
        ext = 0.5 * span
        ux_norm, uy_norm = (ux / span, uy / span)
        a_exp = (a_pt[0] - ux_norm * ext, a_pt[1] - uy_norm * ext)
        b_exp = (b_pt[0] + ux_norm * ext, b_pt[1] + uy_norm * ext)
    except Exception:
        a_exp, b_exp = a_pt, b_pt
    _PREVIEW_GEOM = (a_exp, b_exp, level)
    return True

def update_bridge_preview(pos: Point, water, roads=None, level: int = 1, angle_blend: float = 0.5) -> bool:
    """Update the current preview bridge to a new cursor position.  If
    there is no active preview, this behaves like ``start_bridge_preview``.
    Returns True if an updated preview exists, False otherwise.  The
    road network and blend factor are passed through to ``start_bridge_preview``.
    """
    global _PREVIEW_GEOM
    return start_bridge_preview(pos, water, roads=roads, level=level, angle_blend=angle_blend)

def generate_bridges_from_roads(roads, water, desired: int = None, min_spacing: float = 200.0, max_span: float = 400.0, level: int = 1):
    """
    Generate bridge spans from the existing road network by projecting truncated road segments
    across the river along their headings.  Each segment in ``roads.river_skip_segments`` has been
    truncated just before crossing the river.  For each such segment, this function casts a ray
    from the truncated endpoint across the river and uses the first two intersections with the
    river polygon to define a bridge deck.  Bridges are overlay-only and do not modify the road
    network.

    Parameters
    ----------
    roads : RoadSystem
        The road network containing a ``river_skip_segments`` set of truncated segments.
    water : Water
        Object holding the ``river_poly`` polygon.
    desired : int or None
        Maximum number of bridges to generate.  If None, generate for all candidate segments.
    min_spacing : float
        Minimum allowed distance between the midpoints of adjacent bridges.  Helps avoid clustering.
    max_span : float
        Maximum allowed length (in world units) of the water-only bridge span.  Longer candidates are skipped.
    level : int
        Road level assigned to generated bridge decks.

    Returns
    -------
    int
        The number of bridges added to the overlay.
    """
    # Clear any existing bridge overlay state
    _clear_bridge_overlay_cache()
    # Ensure a river polygon is present
    river = getattr(water, 'river_poly', None)
    if not river:
        return 0
    import math
    candidates = []
    # Use segments that were truncated by the river as seeds for bridge projections
    for seg in getattr(roads, 'river_skip_segments', set()):
        # Compute the direction vector from seg.a to seg.b (towards the river)
        dx = seg.b[0] - seg.a[0]
        dy = seg.b[1] - seg.a[1]
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        # Starting point for projection is the truncated end near the river
        start = seg.b
        # Cast a ray far across the river; choose a large distance relative to map size
        max_len = 4000.0
        far = (start[0] + ux * max_len, start[1] + uy * max_len)
        # Find intersections of the ray with the river polygon edges
        intersections = []
        for i in range(len(river)):
            p0, p1 = river[i], river[(i + 1) % len(river)]
            ok, P, t_line, t_seg = seg_intersection(start, far, p0, p1)
            if ok and P is not None and t_line is not None and 0.0 <= t_line <= 1.0:
                intersections.append((t_line, P))
        # We need at least two intersections to span the river
        if len(intersections) >= 2:
            intersections.sort(key=lambda x: x[0])
            a_pt = intersections[0][1]
            b_pt = intersections[1][1]
            # Compute span length and ensure it is within allowed range
            span = math.hypot(b_pt[0] - a_pt[0], b_pt[1] - a_pt[1])
            if span >= 5.0 and span <= max_span:
                # Compute midpoint for spacing checks
                mid = ((a_pt[0] + b_pt[0]) * 0.5, (a_pt[1] + b_pt[1]) * 0.5)
                candidates.append((mid, a_pt, b_pt))
    # Sort candidates by the x-coordinate of their midpoints to approximate ordering along the river
    candidates.sort(key=lambda c: c[0][0])
    # Filter candidates by minimum spacing and limit count by desired
    kept = []
    for mid, a_pt, b_pt in candidates:
        skip = False
        for existing_mid, _, _ in kept:
            # squared distance between midpoints
            dx = mid[0] - existing_mid[0]
            dy = mid[1] - existing_mid[1]
            if dx * dx + dy * dy < min_spacing * min_spacing:
                skip = True
                break
        if skip:
            continue
        kept.append((mid, a_pt, b_pt))
        if desired is not None and len(kept) >= desired:
            break
    # Append bridge geometries to the overlay
    global _BRIDGE_GEOMS
    count = 0
    for mid, a_pt, b_pt in kept:
        _BRIDGE_GEOMS.append((a_pt, b_pt, level))
        count += 1
    # If no bridges were generated via road projections, fall back to a uniform
    # spacing strategy.  This prevents the G key from appearing to do nothing
    # when there are no candidate road segments near the river.  Using the
    # uniform bridge generator ensures that some bridges always appear over
    # the river, even if they are not directly aligned to roads.  We respect
    # the caller-provided `desired` count if available; otherwise a default
    # of 4 bridges is used.
    if count == 0:
        # choose a reasonable default number of bridges when not provided
        fallback_n = desired if desired is not None else 4
        try:
            # generate_bridges_uniform internally clears the overlay and
            # populates _BRIDGE_GEOMS.  We rely on that behavior here to
            # populate the overlay when there are no road-based bridges.
            return generate_bridges_uniform(roads, water, desired=fallback_n, level=level)
        except Exception:
            # If the uniform generator fails for any reason, simply return 0
            return 0
    return count