"""
facilities.py
================

This module defines a system for placing special-purpose structures
("facilities") on top of the road and building network.  Facilities
behave much like regular buildings: they occupy space on the ground,
have a polygonal footprint oriented along the road network, and must
respect the same collision rules as houses (no overlaps with roads,
water, walls or other buildings).  Squares (public plazas) are a
special case: they occupy an entire block around a road junction and
are drawn in green.

The module also defines a simple UI consisting of checkboxes to
select which facility types should be generated.  The UI is intended
to live on the left side of the main map and works independently of
the existing slider panel on the right.

To integrate the facility system into the main city weaver:

    from facilities import FacilitiesSystem, FacilitiesUI

    facilities = FacilitiesSystem(map_size=(MAP_WIDTH, HEIGHT), params=PARAMS)
    facilities_ui = FacilitiesUI((0, 0, Si(220), HEIGHT), facility_names=list(facilities.specs))
    facilities_ui.set_font(font)

    # When regenerating buildings (e.g. pressing B) or when toggling
    # checkboxes, call facilities.generate(...).  Draw the facilities
    # after buildings but before UI overlays:
    facilities.draw(screen, world_to_screen, cam_zoom)
    facilities_ui.draw_bg(screen); facilities_ui.draw(screen)

The UI returns True from handle_event() when it consumes an event and
toggles one or more checkboxes; call facilities.generate(...) in that
case to refresh the placement.

"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional

from geometry import rect_from_axes, poly_aabb, polys_intersect, road_tube_poly, point_in_poly
from quadtree import Quadtree



# ============================ Block detection helpers ============================
def _convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 2: return points
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _find_enclosed_blocks(road_system, map_rect=None, grid_step: float = 16.0, inflate: float = 1.15):
    """Detect enclosed 'blocks' fully surrounded by roads. Returns polygons (largest first)."""
    from geometry import point_seg_dist, poly_aabb
    try:
        segs = [s for s in getattr(road_system, 'segments', []) if getattr(s, 'active', True)]
        if not segs: return []
    except Exception:
        return []
    try:
        if map_rect is None:
            gb = getattr(road_system, 'gen_rect', None) or getattr(road_system, 'generation_rect', None)
            if gb and len(gb) == 4:
                x0,y0,x1,y1 = float(gb[0]), float(gb[1]), float(gb[2]), float(gb[3])
                W, H = x1-x0, y1-y0
            else:
                xs, ys = [], []
                for s in segs:
                    ax,ay=s.a; bx,by=s.b
                    xs += [ax,bx]; ys += [ay,by]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                pad = max(32.0, 0.05*max(1.0, x1-x0, y1-y0))
                x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
                W, H = x1-x0, y1-y0
        else:
            (x0,y0,x1,y1) = map_rect; W, H = x1-x0, y1-y0
    except Exception:
        return []
    if W <= 0 or H <= 0: return []
    nx = max(8, int(W // grid_step)); ny = max(8, int(H // grid_step))
    blocked = [[False]*ny for _ in range(nx)]
    def road_width_of(seg):
        """
        Return the effective blocking width of a road segment.  The width is
        derived from the road's level and the configuration dictionary on
        the road system.  A small padding multiplier is applied via the
        ``inflate`` parameter to ensure neighbouring grid cells adjacent
        to the road are also marked as blocked.  The previous code used a
        generous padding factor of 1.15 which resulted in detected
        blocks being smaller than the surrounding open space.  To allow
        plazas to fill their blocks more completely, the padding
        multiplier is now tuned down via the caller-supplied ``inflate``
        argument.
        """
        try:
            lvl = int(getattr(seg, 'level', 1))
            eff = int(road_system.map_lvl(lvl)) if hasattr(road_system, 'map_lvl') else lvl
            w = float(road_system.params.get('ROAD_WIDTH', {}).get(eff, 4.0))
            # Apply the inflate factor directly; callers control inflate
            return max(2.0, w) * inflate
        except Exception:
            return 6.0 * inflate
    for s in segs:
        w = road_width_of(s)
        (ax,ay),(bx,by) = s.a, s.b
        minx = int((min(ax,bx)-x0) // grid_step) - 2
        maxx = int((max(ax,bx)-x0) // grid_step) + 2
        miny = int((min(ay,by)-y0) // grid_step) - 2
        maxy = int((max(ay,by)-y0) // grid_step) + 2
        minx = max(0, minx); miny = max(0, miny)
        maxx = min(nx-1, maxx); maxy = min(ny-1, maxy)
        for ix in range(minx, maxx+1):
            cx = x0 + ix*grid_step + grid_step*0.5
            for iy in range(miny, maxy+1):
                cy = y0 + iy*grid_step + grid_step*0.5
                d,_,_ = point_seg_dist((cx,cy), (ax,ay), (bx,by))
                if d <= w*0.55:
                    blocked[ix][iy] = True
    from collections import deque
    comps = []
    seen = [[False]*ny for _ in range(nx)]
    for ix in range(nx):
        for iy in range(ny):
            if blocked[ix][iy] or seen[ix][iy]: continue
            q = deque([(ix,iy)]); seen[ix][iy] = True; cells = []; touches_border=False
            while q:
                x,y = q.popleft()
                cells.append((x,y))
                if x==0 or y==0 or x==nx-1 or y==ny-1: touches_border=True
                for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    xx,yy = x+dx, y+dy
                    if 0<=xx<nx and 0<=yy<ny and not blocked[xx][yy] and not seen[xx][yy]:
                        seen[xx][yy]=True; q.append((xx,yy))
            if not touches_border and len(cells) >= 6:
                comps.append(cells)
    if not comps: return []
    polys = []
    def center(ix,iy): return (x0 + ix*grid_step + grid_step*0.5, y0 + iy*grid_step + grid_step*0.5)
    for cells in comps:
        pts = [center(x,y) for (x,y) in cells]
        hull = _convex_hull(pts)
        if len(hull) >= 3:
            polys.append(hull)
    def poly_area(poly):
        a = 0.0
        for i in range(len(poly)):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
            a += x1*y2 - x2*y1
        return abs(a)*0.5
    polys.sort(key=poly_area, reverse=True)
    return polys

def _shrink_poly_centroid(poly, factor=0.88):
    """Inset a polygon by moving vertices toward its centroid.
    This preserves shape and helps keep the plaza inside the block.
    """
    if not poly or len(poly) < 3: return poly
    cx = sum(x for x, _ in poly) / len(poly)
    cy = sum(y for _, y in poly) / len(poly)
    out = []
    for (x,y) in poly:
        nx = cx + (x - cx) * factor
        ny = cy + (y - cy) * factor
        out.append((nx, ny))
    return out
# ========================== end helpers ==========================
class FacilitySpec:
    """Specification for a facility type.

    Parameters
    ----------
    name : str
        Human-readable name of the facility.
    scale : float
        Relative scale multiplier applied to the base house dimensions
        (frontage and depth).  Values >1 produce larger footprints.
    count : int
        Default number of instances to attempt placing when this
        facility type is enabled.
    color : Tuple[int, int, int]
        Fill colour used when drawing the facility footprint.  The
        colour should be a light pastel hue to contrast with regular
        houses.
    kind : str
        Either "building" for rectilinear structures aligned with
        roads or "square" for public plazas that occupy entire
        road junctions.  Squares are drawn as large green blocks.
    """

    def __init__(self, name: str, scale: float = 1.5, count: int = 2,
                 color: Tuple[int, int, int] = (200, 200, 200), kind: str = "building") -> None:
        self.name = name
        self.scale = float(scale)
        self.count = int(count)
        self.color = color
        self.kind = kind  # either 'building' or 'square'
        self.count = int(count) if count is not None else 1


class FacilitiesSystem:
    def square_polys(self):
        """Return polygons for placed 'Square' facilities."""
        out = []
        for fac in getattr(self, "facilities", []):
            spec = fac.get("spec", None)
            nm = (getattr(spec, "name", "") or "").lower()
            kd = getattr(spec, "kind", "")
            if kd == "square" or nm == "square":
                p = fac.get("poly")
                if p:
                    out.append(p)
        return out

    """System responsible for generating and drawing facilities.

    Facilities are placed after the road and house network has been
    generated.  They sit alongside houses but are larger in footprint
    and use a distinctive colour per type.  Squares occupy an entire
    block around a multi-way junction.  All facilities must remain
    within the map bounds, inside any active city walls and may not
    overlap roads, water, houses or other facilities.
    """

    def __init__(self, map_size: Tuple[int, int], params: Dict) -> None:
        self.MAP_WIDTH, self.HEIGHT = map_size
        self.params = params
        # List of placed facility dictionaries; each has keys
        # 'poly', 'aabb', 'spec'.  The spec references a FacilitySpec.
        self.facilities: List[Dict] = []
        # Quadtree for fast spatial lookup of placed facilities
        self.fac_qt: Optional[Quadtree] = None
        # Define default specifications.  Users can toggle these via
        # the FacilitiesUI.  Additional specs can be added here.
        self.specs: Dict[str, FacilitySpec] = {}
        self._init_default_specs()
        # Font used for drawing labels above facility bullets.  This is
        # assigned from the main program via set_font().  A separate
        # font is used instead of piggy‑backing on the UI so that
        # facility labels remain legible regardless of where the UI is
        # located.
        self.font = None
        # Track which facility currently has its name label visible.  When
        # the user clicks on a facility's marker, this reference is set
        # to that facility's dictionary.  Clicking a different marker
        # will switch the active label; clicking elsewhere hides any
        # active label.
        self.active_label_for = None

    def _init_default_specs(self) -> None:
        """Populate the default facility specification dictionary.

        Colours are chosen to be distinct yet pastel, so they stand
        apart from regular houses but are not distracting.  All counts
        default to 2 as requested by the user.
        """
        # Helper to register a spec
        def reg(name: str, scale: float, color: Tuple[int, int, int], kind: str = "building", count: int = 1):
            self.specs[name] = FacilitySpec(name=name, scale=scale, color=color, kind=kind, count=count)

        # Public plazas (occupy entire blocks)
        reg("Square", 4.0, (100, 180, 100), kind="square", count=2)
        # Core civic amenities
        reg("School", 2.5, (255, 230, 150))
        reg("Bank", 2.5, (220, 190, 150))
        reg("Restaurant", 2.5, (255, 190, 140))
        reg("Hospital", 2.8, (255, 150, 150))
        reg("Post Office", 2.3, (240, 210, 170))
        reg("Pansion", 2.2, (220, 180, 220))
        reg("Hotel", 2.4, (200, 170, 230))
        reg("Jail", 2.4, (180, 180, 180))
        # Cultural and civic structures
        reg("Church", 2.6, (190, 210, 255))
        reg("Bath", 2.3, (180, 210, 255))
        reg("Music Hall", 2.6, (255, 190, 230))
        reg("Club", 2.4, (240, 120, 180))
        reg("Station", 2.7, (200, 200, 220))
        reg("Library", 2.4, (255, 220, 160))
        reg("Museum", 2.5, (200, 220, 170))
        reg("Pub", 2.3, (230, 160, 110))
        reg("Gun Shop", 2.2, (190, 140, 100))
        reg("Ministries", 2.5, (180, 200, 220))
        reg("Pharmacist", 2.3, (180, 220, 180))
        reg("Police Station", 2.6, (160, 160, 220))
        reg("Market", 2.5, (240, 200, 140))
        reg("Alchemist", 2.4, (200, 180, 250))
        reg("Vendor", 2.0, (230, 190, 140))
        reg("Blacksmith", 2.2, (150, 120, 100))

    # ------------------------------------------------------------------
    # Font and interaction
    # ------------------------------------------------------------------
    def set_font(self, font) -> None:
        """Assign the font used to render facility name labels.

        The city weaver passes in the global font after
        initialisation so that the facilities system can render
        tooltips.  Without a font the labels simply will not
        draw.
        """
        self.font = font

    def handle_click(self, screen_pos: Tuple[int, int], world_to_screen, cam_zoom: float) -> bool:
        """Handle a mouse click in screen coordinates.

        If the click falls within the marker for a facility,
        update the active label reference to that facility and
        return True (the event was consumed).  Otherwise clear
        any active label and return False.

        Parameters
        ----------
        screen_pos : (x, y)
            The position of the mouse in screen coordinates.
        world_to_screen : callable
            Function that converts a world coordinate to screen
            space.  Needed to derive marker positions.
        cam_zoom : float
            The current camera zoom level.  Unused currently but
            supplied for future refinements.
        """
        mx, my = screen_pos
        consumed = False
        marker_radius = 5 + 2  # match the outer radius drawn in draw()
        # Iterate in reverse order so that markers drawn later (on
        # top) are given priority.  This makes it feel natural when
        # multiple markers overlap at the same pixel.
        for fac in reversed(self.facilities):
            pos = fac.get("_bullet_screen_pos")
            if not pos:
                continue
            sx, sy = pos
            dx = mx - sx
            dy = my - sy
            if dx * dx + dy * dy <= (marker_radius) * (marker_radius):
                # Clicked inside this marker.  Toggle if already
                # active, otherwise set as active.
                if self.active_label_for is fac:
                    self.active_label_for = None
                else:
                    self.active_label_for = fac
                consumed = True
                break
        # If click was not on a marker, clear any active label.
        if not consumed:
            self.active_label_for = None
        return consumed

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------
    def _poly_centroid(self, poly: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Compute the centroid (arithmetic mean) of a polygon."""
        cx = sum(p[0] for p in poly) / max(1, len(poly))
        cy = sum(p[1] for p in poly) / max(1, len(poly))
        return (cx, cy)

    def _inside_walls(self, poly: List[Tuple[float, float]]) -> bool:
        """
        Return True if the entire polygon lies inside the active wall
        polygon.  If walls are disabled or no polygon is defined this
        returns True.  A polygon is considered inside if all vertices
        and its centroid are strictly inside the wall polygon.
        """
        try:
            import walls as _walls  # late import to avoid cycle if walls unused
            st = _walls.get_wall_state() if hasattr(_walls, "get_wall_state") else {}
            enabled = bool(st.get("enabled"))
            wall_poly = st.get("poly") or []
            if not (enabled and wall_poly):
                return True
            for (x, y) in poly:
                if not point_in_poly((x, y), wall_poly):
                    return False
            cx, cy = self._poly_centroid(poly)
            if not point_in_poly((cx, cy), wall_poly):
                return False
            return True
        except Exception:
            return True

    def _candidate_ok(self, poly: List[Tuple[float, float]], b_qt: Quadtree,
                       road_system, water, fac_qt: Quadtree) -> bool:
        """
        Determine whether a candidate facility polygon is valid.  It must
        satisfy the following:

          * All vertices lie inside the map bounds.
          * The polygon lies inside the active walls (if any).
          * It does not intersect any existing houses (via b_qt) or
            facilities (via fac_qt).
          * It does not intersect any road tubes.
          * It does not intersect the river or sea.
        """
        # Bounds check
        for (x, y) in poly:
            if not (0.0 <= x < self.MAP_WIDTH and 0.0 <= y < self.HEIGHT):
                return False
        # Respect walls
        if not self._inside_walls(poly):
            return False
        aabb = poly_aabb(poly)
        # Collide with houses
        nearby: List[Dict] = []
        b_qt.query(aabb, nearby)
        for other in nearby:
            opoly = other.get("poly", None)
            if opoly and polys_intersect(poly, opoly):
                return False
        # Collide with other facilities
        fnear: List[Dict] = []
        fac_qt.query(aabb, fnear)
        for other in fnear:
            opoly = other.get("poly", None)
            if opoly and polys_intersect(poly, opoly):
                return False
        # Road collision check: enlarge search rect by a pad equal to the
        # widest road.  Only active segments are considered.
        pad = max(self.params.get("ROAD_WIDTH", {0: 1}).values())
        rx, ry, rw, rh = aabb
        road_rect = (rx - pad, ry - pad, rw + pad * 2, rh + pad * 2)
        cand_segments: List = []
        road_system.qt.query(road_rect, cand_segments)
        for seg in cand_segments:
            if not getattr(seg, "active", False):
                continue
            width = road_system.params["ROAD_WIDTH"][getattr(seg, "level", 0)]
            tube = road_tube_poly(seg.a, seg.b, width)
            if polys_intersect(poly, tube):
                return False
        # Water collision check
        if water and getattr(water, "river_poly", None):
            if polys_intersect(poly, water.river_poly):
                # Only reject if centroid lies in water (same as houses)
                cx, cy = self._poly_centroid(poly)
                if point_in_poly((cx, cy), water.river_poly):
                    return False
        if water and getattr(water, "sea_poly", None):
            if polys_intersect(poly, water.sea_poly):
                return False
        return True

    # ------------------------------------------------------------------
    # Generation entry point
    # ------------------------------------------------------------------
    def generate(self, road_system, building_system, water, enabled_names: List[str]) -> None:
        """Generate all enabled facilities.

        Parameters
        ----------
        road_system : RoadSystem
            The fully constructed road network.
        building_system : BuildingSystem
            Holds the list of existing houses.  A quadtree will be
            built from the houses here to perform collision tests.
        water : WaterSystem
            Provides river and sea polygons for exclusion zones.
        enabled_names : List[str]
            Names of facility types to generate.  Only types present
            in self.specs are honoured; unknown names are ignored.
        """
        # Build a quadtree of existing houses for quick overlap checks.
        # Houses that have been replaced by facilities (i.e. marked
        # "removed") are skipped so they no longer impede placement
        # of subsequent facilities.  Because generate() is called
        # afresh whenever houses regenerate or UI toggles, removed
        # flags are reset by the building system when houses are
        # regenerated.
        b_qt = Quadtree((0, 0, float(self.MAP_WIDTH), float(self.HEIGHT)))
        try:
            for b in getattr(building_system, "buildings", []):
                if b.get("removed"):
                    continue
                aabb = b.get("aabb", None)
                if aabb:
                    b_qt.insert(aabb, b)
        except Exception:
            # gracefully handle missing building list
            pass

        # Reset facility state
        self.facilities = []
        self.fac_qt = Quadtree((0, 0, float(self.MAP_WIDTH), float(self.HEIGHT)))

        for name in enabled_names:
            spec = self.specs.get(name)
            if not spec:
                continue
            if spec.kind == "square":
                # _GREEN_BLOCKS_PATCH: fill two enclosed blocks and skip houses inside
                try:
                    # Use a reduced inflate factor when detecting enclosed blocks.
                    # A smaller inflate yields larger detected block polygons and
                    # results in plazas that more closely fill the available
                    # space.  The grid_step remains coarse so as not to
                    # significantly impact performance.
                    polys = _find_enclosed_blocks(road_system, grid_step=12.0, inflate=1.0)
                except Exception:
                    polys = []
                if polys:
                    take = min(int(getattr(spec, 'count', 2)), len(polys))
                    import random
                    random.shuffle(polys)
                    for poly in polys[:take]:
                        # Reduce the shrink factor so plazas expand to fill their
                        # entire block.  A factor of 0.95 retains a small
                        # margin so that squares do not overlap roads while
                        # appearing noticeably larger than before.  Setting
                        # factor to 1.0 would remove all margins; adjust
                        # further if overlapping is observed.
                        poly = _shrink_poly_centroid(poly, factor=0.95)
                        aabb = poly_aabb(poly)
                        near = []
                        try:
                            b_qt.query(aabb, near)
                        except Exception:
                            near = []
                        # Remove any houses whose footprints intersect with the
                        # proposed plaza.  Those houses will no longer
                        # participate in subsequent placement checks.
                        for h in near:
                            try:
                                hpoly = h.get("poly")
                                if hpoly and polys_intersect(poly, hpoly):
                                    h["removed"] = True
                            except Exception:
                                pass
                        fac = {"poly": poly, "aabb": aabb, "spec": spec}
                        self.facilities.append(fac)
                        try:
                            self.fac_qt.insert(aabb, fac)
                        except Exception:
                            pass
                    continue
                # (Fallback to junction-based squares if none enclosed)

                # Squares do not replace houses; they occupy open
                # junction blocks.  We build them using the same
                # quadtree of houses so that squares avoid
                # overlapping any existing house footprints.
                self._generate_squares(spec, road_system, b_qt, water)
            else:
                # For building‑type facilities, pass the building system
                # alongside the quadtree so that the generator can mark
                # houses as removed once replaced.  The quadtree is
                # reused throughout this call; removed houses are
                # dynamically skipped in subsequent placement checks.
                self._generate_spec_buildings(spec, road_system, building_system, b_qt, water)

    def _generate_spec_buildings(self, spec: FacilitySpec, road_system, building_system, b_qt: Quadtree, water) -> None:
        """Place facilities by replacing existing houses.

        Instead of attempting to randomly place facilities along the
        road network as if they were independent objects, this
        implementation chooses existing houses and replaces them
        directly.  This guarantees that each facility faces a road and
        occupies a valid building plot.  The facility footprint is
        scaled relative to the chosen house and made square (equal
        frontage and depth) to satisfy the user's request.  Houses
        marked as replaced are flagged with a ``removed`` field so
        they no longer render.  Facilities never overlap other
        houses, roads, water, walls or existing facilities.

        Parameters
        ----------
        spec : FacilitySpec
            The specification for the facility type being placed.
        road_system : RoadSystem
            Provides access to the road network for collision tests.
        building_system : BuildingSystem
            Provides the list of existing houses.  Houses are marked
            as removed when replaced by a facility.
        b_qt : Quadtree
            Spatial index of houses built from ``building_system``
            excluding those already removed.  This quadtree is
            regenerated on each call to generate() and reused here.
        water : WaterSystem
            Provides river and sea polygons for exclusion tests.
        """
        # Build a list of candidate indices referencing houses that
        # have not been replaced.  To improve placement success
        # especially for large facilities, compute a size score for each
        # house based on its frontage/depth and sort by descending
        # size.  A small random jitter ensures diversity between runs.
        candidate_info: List[Tuple[float, int]] = []  # (negative score, index)
        for idx, house in enumerate(getattr(building_system, "buildings", [])):
            if house.get("removed"):
                continue
            poly = house.get("poly")
            if not poly or len(poly) < 4:
                continue
            # Compute base size of this house: take the larger of
            # frontage and depth as the footprint size.  Use this as
            # the primary sorting key.  The negative sign ensures
            # descending sort when sorted in ascending order.
            try:
                p0, p1, p2, p3 = poly[0], poly[1], poly[2], poly[3]
                # Frontage along p0->p1, depth along p0->p3
                tx, ty = p1[0] - p0[0], p1[1] - p0[1]
                tl = (tx * tx + ty * ty) ** 0.5
                nx_v, ny_v = p3[0] - p0[0], p3[1] - p0[1]
                nl = (nx_v * nx_v + ny_v * ny_v) ** 0.5
                base_size = max(tl, nl)
            except Exception:
                # Fallback: treat degenerate polygon as zero size
                base_size = 0.0
            # Add a tiny random jitter so that houses with identical
            # base sizes don't always sort in the same order.  The
            # jitter magnitude is tiny relative to base_size so it
            # doesn't materially affect ordering.
            jitter = random.random() * 1e-3
            candidate_info.append((-base_size + jitter, idx))
        if not candidate_info or spec.count <= 0:
            return
        # Sort by the negative base size (descending by size)
        candidate_info.sort()
        # Limit the number of candidates we evaluate to improve
        # performance.  Larger specs attempt more candidates while
        # maintaining a reasonable upper bound.  Without this limit,
        # scanning every house for every spec can cause noticeable
        # slowdowns when many facilities are enabled.
        max_candidates = max(int(spec.count) * 50, 300)
        selected = [idx for (_, idx) in candidate_info[:max_candidates]]
        placed = 0
        # Iterate through candidate houses until we place the desired
        # number of facilities or exhaust the list.
        for idx in selected:
            if placed >= spec.count:
                break
            house = building_system.buildings[idx]
            if house.get("removed"):
                continue
            poly = house.get("poly")
            if not poly or len(poly) != 4:
                continue
            # Determine orientation vectors based on the house polygon.
            # Use p0->p1 as the tangential (along the road) direction
            # and p0->p3 as the normal (away from the road) direction.
            p0, p1, p2, p3 = poly
            # Compute unit tangential vector
            tx, ty = p1[0] - p0[0], p1[1] - p0[1]
            tl = (tx * tx + ty * ty) ** 0.5
            if tl < 1e-6:
                continue
            ux, uy = tx / tl, ty / tl
            # Compute unit normal vector
            nx_v, ny_v = p3[0] - p0[0], p3[1] - p0[1]
            nl = (nx_v * nx_v + ny_v * ny_v) ** 0.5
            if nl < 1e-6:
                continue
            nx, ny = nx_v / nl, ny_v / nl
            # Compute original frontage and depth
            frontage = tl
            depth = nl
            # Determine base size from the original house (max of frontage
            # and depth).  We will attempt to place a facility at
            # decreasing scales if the full scale does not fit.  This
            # adaptive approach increases the chance that larger
            # facilities find a valid placement by shrinking until
            # successful.  We never shrink below the original house
            # size (scale factor of 1.0).
            base_size = max(frontage, depth)
            # Construct a list of scale factors to try: start with the
            # requested scale and progressively reduce.  Use a few
            # discrete reductions down to unity.  Additional reductions
            # can be appended here if necessary.
            reductions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
            placed_this_candidate = False
            
            # Fit-to-block: scan maximal square, then cap by spec target
            fac_scale = float(self.params.get("FACILITY_SCALE", 1.0))
            s_max = self._max_square_size_replace(poly, ux, uy, nx, ny, depth, base_size, b_qt, road_system, water, self.fac_qt)
            if s_max <= 0.0:
                continue
            s_target = base_size * spec.scale * fac_scale
            s_use = min(s_target, s_max)
            cx_old, cy_old = self._poly_centroid(poly)
            shift_x = nx * (s_use - depth) / 2.0
            shift_y = ny * (s_use - depth) / 2.0
            cx_new = cx_old + shift_x
            cy_new = cy_old + shift_y
            fpoly = rect_from_axes((cx_new, cy_new), (ux, uy), (nx, ny), s_use, s_use)
            if self._candidate_ok_replace(fpoly, poly, b_qt, road_system, water, self.fac_qt):
                aabb = poly_aabb(fpoly)
                fac = {"poly": fpoly, "aabb": aabb, "spec": spec}
                # Remove any houses inside the plaza
                aabb = poly_aabb(poly)
                near = []
                b_qt.query(aabb, near)
                for h in near:
                    hpoly = h.get("poly")
                    if hpoly and polys_intersect(poly, hpoly):
                        h["removed"] = True
                self.facilities.append(fac)
                self.fac_qt.insert(aabb, fac)
                house["removed"] = True
                placed += 1
                continue
            s_use2 = max(0.0, 0.9 * s_use)
            if s_use2 > 0.0:
                shift_x = nx * (s_use2 - depth) / 2.0
                shift_y = ny * (s_use2 - depth) / 2.0
                cx_new = cx_old + shift_x
                cy_new = cy_old + shift_y
                fpoly = rect_from_axes((cx_new, cy_new), (ux, uy), (nx, ny), s_use2, s_use2)
                if self._candidate_ok_replace(fpoly, poly, b_qt, road_system, water, self.fac_qt):
                    aabb = poly_aabb(fpoly)
                    fac = {"poly": fpoly, "aabb": aabb, "spec": spec}
                    self.facilities.append(fac)
                    self.fac_qt.insert(aabb, fac)
                    house["removed"] = True
                    placed += 1
                    continue
            # proceed to next candidate
  # If no placement succeeded,
# End of reductions loop.  If no placement succeeded, attempt a tiny fallback near the house centroid.
            if not placed_this_candidate:
                tiny_scale = max(0.1, 0.15 * float(self.params.get("FACILITY_SCALE", 1.0)))
                new_size = base_size * spec.scale * tiny_scale
                fpoly = rect_from_axes((cx_old, cy_old), (ux, uy), (nx, ny), new_size, new_size)
                if self._candidate_ok_replace(fpoly, poly, b_qt, road_system, water, self.fac_qt):
                    aabb = poly_aabb(fpoly)
                    fac = {"poly": fpoly, "aabb": aabb, "spec": spec}
                    self.facilities.append(fac)
                    self.fac_qt.insert(aabb, fac)
                    house["removed"] = True
                    placed += 1
                    placed_this_candidate = True
                    # continue to next candidate
            # proceed to next candidate house.
            if placed_this_candidate:
                continue
        # End for

    def _candidate_ok_replace(self, poly: List[Tuple[float, float]], skip_poly: List[Tuple[float, float]],
                              b_qt: Quadtree, road_system, water, fac_qt: Quadtree) -> bool:
        """Variant of _candidate_ok used when replacing an existing house.

        This check is identical to `_candidate_ok` except that
        intersections with the house being replaced (``skip_poly``)
        are ignored.  All other houses, facilities, roads, walls and
        water bodies are respected.
        """
        # Bounds check
        for (x, y) in poly:
            if not (0.0 <= x < self.MAP_WIDTH and 0.0 <= y < self.HEIGHT):
                return False
        # Respect walls
        if not self._inside_walls(poly):
            return False
        aabb = poly_aabb(poly)
        # Collide with houses
        nearby: List[Dict] = []
        b_qt.query(aabb, nearby)
        for other in nearby:
            # Skip if this building has been removed by a previous
            # facility replacement.  Removed houses should not block
            # placement of new facilities.
            if other.get("removed"):
                continue
            opoly = other.get("poly", None)
            if not opoly:
                continue
            # Skip the house we are replacing
            if opoly is skip_poly or opoly == skip_poly:
                continue
            if polys_intersect(poly, opoly):
                return False
        # Collide with other facilities
        fnear: List[Dict] = []
        fac_qt.query(aabb, fnear)
        for other in fnear:
            opoly = other.get("poly", None)
            if opoly and polys_intersect(poly, opoly):
                return False
        # Road collision check
        pad = max(self.params.get("ROAD_WIDTH", {0: 1}).values())
        rx, ry, rw, rh = aabb
        road_rect = (rx - pad, ry - pad, rw + pad * 2, rh + pad * 2)
        cand_segments: List = []
        road_system.qt.query(road_rect, cand_segments)
        for seg in cand_segments:
            if not getattr(seg, "active", False):
                continue
            width = road_system.params["ROAD_WIDTH"][getattr(seg, "level", 0)]
            tube = road_tube_poly(seg.a, seg.b, width)
            if polys_intersect(poly, tube):
                return False
        # Water collision check
        if water and getattr(water, "river_poly", None):
            if polys_intersect(poly, water.river_poly):
                # Only reject if centroid lies in water (same as houses)
                cx, cy = self._poly_centroid(poly)
                if point_in_poly((cx, cy), water.river_poly):
                    return False
        if water and getattr(water, "sea_poly", None):
            if polys_intersect(poly, water.sea_poly):
                return False
        return True

    def _candidate_ok_plaza(self, poly, road_system, water, fac_qt: Quadtree) -> bool:
        """Collision check for plazas (squares) that ignores houses.
        Squares are allowed to overlap houses because those houses will be
        removed once the plaza is placed. We still forbid intersections
        with roads, water, walls, and existing facilities.
        """
        # Bounds
        for (x, y) in poly:
            if not (0.0 <= x < self.MAP_WIDTH and 0.0 <= y < self.HEIGHT):
                return False
        # Respect walls
        if not self._inside_walls(poly):
            return False
        # Avoid other facilities
        aabb = poly_aabb(poly)
        fnear = []
        fac_qt.query(aabb, fnear)
        for other in fnear:
            opoly = other.get("poly", None)
            if opoly and polys_intersect(poly, opoly):
                return False
        # Avoid roads
        pad = max(self.params.get("ROAD_WIDTH", {0: 1}).values())
        rx, ry, rw, rh = aabb
        road_rect = (rx - pad, ry - pad, rw + pad * 2, rh + pad * 2)
        cand_segments = []
        road_system.qt.query(road_rect, cand_segments)
        for seg in cand_segments:
            if not getattr(seg, "active", False):
                continue
            width = road_system.params["ROAD_WIDTH"][getattr(seg, "level", 0)]
            tube = road_tube_poly(seg.a, seg.b, width)
            if polys_intersect(poly, tube):
                return False
        # Avoid water
        if water and getattr(water, "river_poly", None):
            if polys_intersect(poly, water.river_poly):
                cx, cy = self._poly_centroid(poly)
                if point_in_poly((cx, cy), water.river_poly):
                    return False
        if water and getattr(water, "sea_poly", None):
            if polys_intersect(poly, water.sea_poly):
                return False
        return True

    def _max_square_size_replace(self, house_poly, ux, uy, nx, ny, depth, base_size,
                                 b_qt: Quadtree, road_system, water, fac_qt: Quadtree) -> float:
        """Compute the maximum square size that can fit when replacing
        a given house. Keeps the road-facing edge anchored by shifting the
        centre along the normal by (s - depth)/2 as it grows."""
        # Start from a conservative size (slightly smaller than current house depth)
        s_ok = max(4.0, min(base_size, depth * 0.9))
        s_hi = s_ok
        # Exponentially grow until we fail, to bracket the max size
        for _ in range(12):
            shift_x = nx * (s_hi - depth) / 2.0
            shift_y = ny * (s_hi - depth) / 2.0
            cx_old, cy_old = self._poly_centroid(house_poly)
            cx_new, cy_new = cx_old + shift_x, cy_old + shift_y
            fpoly = rect_from_axes((cx_new, cy_new), (ux, uy), (nx, ny), s_hi, s_hi)
            if self._candidate_ok_replace(fpoly, house_poly, b_qt, road_system, water, fac_qt):
                s_ok = s_hi
                s_hi *= 1.5
            else:
                break
        # Binary search between s_ok and s_hi for precision
        s_lo = s_ok
        s_hi = max(s_hi, s_ok)
        for _ in range(18):
            s_mid = 0.5 * (s_lo + s_hi)
            shift_x = nx * (s_mid - depth) / 2.0
            shift_y = ny * (s_mid - depth) / 2.0
            cx_old, cy_old = self._poly_centroid(house_poly)
            cx_new, cy_new = cx_old + shift_x, cy_old + shift_y
            fpoly = rect_from_axes((cx_new, cy_new), (ux, uy), (nx, ny), s_mid, s_mid)
            if self._candidate_ok_replace(fpoly, house_poly, b_qt, road_system, water, fac_qt):
                s_lo = s_mid
            else:
                s_hi = s_mid
            if abs(s_hi - s_lo) < 0.25:
                break
        return max(0.0, s_lo)

    def _max_square_size_generic(self, center, ux, uy, nx, ny,
                                 b_qt: Quadtree, road_system, water, fac_qt: Quadtree, plaza: bool = False) -> float:
        """Compute the maximum square size centred at `center` that avoids collisions.
        Starts from a small size; if even that fails, it shrinks further, then grows
        exponentially to bracket, then binary searches for precision.
        """
        px, py = center
        # start at a small size
        s_min = 0.5
        s_hi = 1.0
        def ok_at(s):
            poly = rect_from_axes((px, py), (ux, uy), (nx, ny), s, s)
            return (self._candidate_ok_plaza(poly, road_system, water, fac_qt) if plaza
                    else self._candidate_ok(poly, b_qt, road_system, water, fac_qt))
        # If 1.0 fails, shrink until s_min
        if not ok_at(s_hi):
            s_hi = 0.75
            while s_hi >= s_min and not ok_at(s_hi):
                s_hi *= 0.5
            if s_hi < s_min:
                return 0.0
        # Grow until failure
        s_ok = s_hi
        for _ in range(16):
            s_try = s_ok * 1.6
            if ok_at(s_try):
                s_ok = s_try
            else:
                break
        # Binary search between s_ok and the first failing above it
        s_lo = s_ok
        s_up = s_ok * 1.6
        for _ in range(20):
            s_mid = 0.5 * (s_lo + s_up)
            if ok_at(s_mid):
                s_lo = s_mid
            else:
                s_up = s_mid
            if abs(s_up - s_lo) < 0.25:
                break
        return max(0.0, s_lo)



    def _generate_squares(self, spec: FacilitySpec, road_system, b_qt: Quadtree, water) -> None:
        """Place large plaza squares at multi-way road junctions.

        A square occupies a roughly square footprint oriented according
        to the principal directions of the junction.  Candidate
        junctions are nodes with degree >= 3 in the road graph.  The
        size of the square is based on the maximum road width.  The
        algorithm iterates through candidate junctions in random
        order, trying to place up to spec.count squares.
        """
        # Determine candidate junctions from the road graph
        road_graph = getattr(road_system, "road_graph", {})
        nodes = getattr(road_system, "nodes", [])
        candidates: List[Tuple[int, Tuple[float, float]]] = []
        for idx, neighbours in road_graph.items():
            try:
                if len(neighbours) >= 3:
                    pos = nodes[idx]
                    candidates.append((idx, pos))
            except Exception:
                continue
        if not candidates:
            return
        random.shuffle(candidates)
        # Precompute a base size for squares: use the maximum road width
        try:
            max_road_width = max(self.params.get("ROAD_WIDTH", {0: 1}).values())
        except Exception:
            max_road_width = 6.0
        base_size = max_road_width * 8.0  # wide enough to cover a junction
        placed = 0
        for idx, pos in candidates:
            if placed >= spec.count:
                break
            px, py = pos
            # Determine orientation: use the averaged direction of all
            # connected segments.  If the result is near zero length,
            # fall back to axis-aligned orientation.
            neighbours = road_graph.get(idx, [])
            sum_vx, sum_vy = 0.0, 0.0
            for nbr in neighbours:
                try:
                    nx_p, ny_p = nodes[nbr]
                    vx, vy = nx_p - px, ny_p - py
                    l = (vx * vx + vy * vy) ** 0.5
                    if l > 1e-6:
                        sum_vx += vx / l
                        sum_vy += vy / l
                except Exception:
                    continue
            # Normalise averaged vector
            lu = (sum_vx * sum_vx + sum_vy * sum_vy) ** 0.5
            if lu > 1e-6:
                ux, uy = (sum_vx / lu, sum_vy / lu)
            else:
                ux, uy = (1.0, 0.0)
            
            # -- NEW: choose a centre inside the biggest angular gap around the junction --
            fac_scale = float(self.params.get("FACILITY_SCALE", 1.0))
            # Build unit direction vectors to neighbours
            rays = []
            for nbr in neighbours:
                try:
                    nx_p, ny_p = nodes[nbr]
                    vx, vy = nx_p - px, ny_p - py
                    L = (vx*vx + vy*vy) ** 0.5
                    if L > 1e-6:
                        ux_r, uy_r = vx / L, vy / L
                        ang = math.atan2(uy_r, ux_r)
                        rays.append((ang, (ux_r, uy_r)))
                except Exception:
                    continue
            if len(rays) < 3:
                continue
            rays.sort(key=lambda t: t[0])
            rays_cycle = rays + [(rays[0][0] + 2*math.pi, rays[0][1])]
            best = None
            for (a0, u0), (a1, u1) in zip(rays_cycle[:-1], rays_cycle[1:]):
                gap = (a1 - a0)
                if gap < math.radians(45):
                    continue
                bx, by = (u0[0] + u1[0], u0[1] + u1[1])
                bl = (bx*bx + by*by) ** 0.5
                if bl < 1e-5:
                    continue
                bx, by = bx / bl, by / bl
                if (best is None) or (gap > best[0]):
                    best = (gap, (bx, by))
            if best is None:
                continue
            bx, by = best[1]
            # move centre into block along gap bisector
            try:
                max_road_width = max(self.params.get("ROAD_WIDTH", {0: 1}).values())
            except Exception:
                max_road_width = 6.0
            step_into_block = max_road_width * 2.0
            cx, cy = px + bx * step_into_block, py + by * step_into_block
            # Use bisector as ux; perpendicular as nx
            ux, uy = bx, by
            nx_vec, ny_vec = by, -bx
            size = min(self._max_square_size_generic((cx, cy), ux, uy, nx_vec, ny_vec, b_qt, road_system, water, self.fac_qt, plaza=True),
                       max_road_width * 8.0 * spec.scale * fac_scale)
            poly = rect_from_axes((cx, cy), (ux, uy), (nx_vec, ny_vec), size, size)
            ok = self._candidate_ok_plaza(poly, road_system, water, self.fac_qt)
            if not ok:
                # Try axis-aligned orientation as a fallback
                ux2, uy2 = 1.0, 0.0
                nx2, ny2 = 0.0, 1.0
                poly2 = rect_from_axes((px, py), (ux2, uy2), (nx2, ny2), size, size)
                if self._candidate_ok_plaza(poly2, road_system, water, self.fac_qt):
                    poly = poly2
                    ok = True
            if ok:
                # Remove any houses inside the plaza
                aabb = poly_aabb(poly)
                near = []
                b_qt.query(aabb, near)
                for h in near:
                    hpoly = h.get("poly")
                    if hpoly and polys_intersect(poly, hpoly):
                        h["removed"] = True
                fac = {"poly": poly, "aabb": aabb, "spec": spec}
                self.facilities.append(fac)
                self.fac_qt.insert(aabb, fac)
                placed += 1
    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, screen, world_to_screen, cam_zoom: float) -> None:
        """Draw all facilities on the provided surface.

        Facilities are drawn after houses but before bridges and UI.  A
        simple drop-shadow is drawn first to lift the facility off the
        ground slightly.  The facility's colour comes from its
        specification.  A dark outline improves legibility at small
        zoom levels.
        """
        import pygame  # Imported here to avoid a hard dependency when used headless
        # Draw each facility
        shadow_offset = (2, 2)
        shadow_color = (100, 100, 100)
        # For bullet positioning we need to record the screen position
        # of each facility’s centroid.  Store these in the facility
        # dict so that handle_click() can use them without recomputing.
        for fac in self.facilities:
            poly = fac["poly"]
            spec: FacilitySpec = fac.get("spec")
            colour = spec.color if spec else (180, 180, 180)
            # Transform to screen coordinates
            pts = [world_to_screen(p) for p in poly]
            sh  = [(x + shadow_offset[0], y + shadow_offset[1]) for (x, y) in pts]
            # Draw shadow
            pygame.draw.polygon(screen, shadow_color, sh)
            # Fill polygon
            pygame.draw.polygon(screen, colour, pts)
            # Outline
            pygame.draw.polygon(screen, (80, 80, 80), pts, max(1, int(2 * cam_zoom)))
            # Compute and store centroid in screen space for bullet drawing
            cx, cy = self._poly_centroid(poly)
            sx, sy = world_to_screen((cx, cy))
            fac["_bullet_screen_pos"] = (sx, sy)
        # Draw facility markers (bullets) on top of all polygons.  A marker
        # appears for each building-type facility (not squares).  The
        # marker colour is derived from the facility specification so
        # that the bullet matches the footprint colour.  An outer
        # border and inner centre dot use darker shades of the base
        # colour for contrast.  This makes it clear which facility the
        # bullet relates to.
        for fac in self.facilities:
            spec: FacilitySpec = fac.get("spec")
            # Skip markers for squares: they cover an entire block and
            # do not need bullets or labels.
            if spec and spec.kind == "square":
                continue
            sx, sy = fac.get("_bullet_screen_pos", (0, 0))
            radius = 5
            if spec:
                r, g, b = spec.color
                # Darken colours for border and centre dot
                border_col = (max(r - 60, 0), max(g - 60, 0), max(b - 60, 0))
                centre_col = (max(r - 80, 0), max(g - 80, 0), max(b - 80, 0))
                fill_col = spec.color
            else:
                border_col = (60, 60, 60)
                fill_col = (200, 200, 200)
                centre_col = (30, 30, 30)
            pygame.draw.circle(screen, border_col, (int(sx), int(sy)), radius + 2)
            pygame.draw.circle(screen, fill_col, (int(sx), int(sy)), radius)
            pygame.draw.circle(screen, centre_col, (int(sx), int(sy)), max(1, radius // 2))
        # Draw the label for the currently active facility if any.  The
        # label is rendered above the marker so as not to occlude
        # underlying geometry.  It will only be drawn if a font has
        # been set via set_font() and a facility is active.
        if self.font and self.active_label_for in self.facilities:
            fac = self.active_label_for
            spec: FacilitySpec = fac.get("spec")
            name = spec.name if spec else "?"
            text_surf = self.font.render(name, True, (30, 30, 30))
            padding_x, padding_y = 6, 4
            tw, th = text_surf.get_width(), text_surf.get_height()
            sx, sy = fac.get("_bullet_screen_pos", (0, 0))
            # Position the label above the marker with a small offset
            lx = int(sx - tw / 2 - padding_x)
            ly = int(sy - th - 2 * padding_y - 8)
            label_rect = pygame.Rect(lx, ly, tw + 2 * padding_x, th + 2 * padding_y)
            # Draw label background with a subtle border and drop
            # shadow so it is legible on any background
            shadow_offset = (2, 2)
            shadow_rect = label_rect.move(shadow_offset)
            pygame.draw.rect(screen, (0, 0, 0), shadow_rect, border_radius=4)
            pygame.draw.rect(screen, (250, 250, 250), label_rect, border_radius=4)
            pygame.draw.rect(screen, (100, 100, 100), label_rect, 1, border_radius=4)
            screen.blit(text_surf, (lx + padding_x, ly + padding_y))

    # ------------------------------------------------------------------
    # State persistence helpers
    # ------------------------------------------------------------------
    def serialize_state(self) -> Dict:
        """Return a serializable snapshot of the facilities."""
        return {
            "facilities": [
                {
                    "poly": [tuple(p) for p in f["poly"]],
                    "aabb": tuple(f["aabb"]),
                    "name": f.get("spec").name if f.get("spec") else None,
                }
                for f in self.facilities
            ]
        }

    def restore_state(self, state: Dict) -> None:
        """Restore facilities from a previously serialized state."""
        self.facilities = []
        self.fac_qt = Quadtree((0, 0, float(self.MAP_WIDTH), float(self.HEIGHT)))
        for f in state.get("facilities", []):
            poly = [tuple(p) for p in f.get("poly", [])]
            aabb = tuple(f.get("aabb", (0, 0, 0, 0)))
            name = f.get("name")
            spec = self.specs.get(name)
            fac = {"poly": poly, "aabb": aabb, "spec": spec}
            self.facilities.append(fac)
            self.fac_qt.insert(aabb, fac)


class _Checkbox:
    """Internal checkbox widget used by the FacilitiesUI."""
    def __init__(self, label: str, is_on: bool = True) -> None:
        self.label = label
        self.is_on = bool(is_on)
        self.rect = None  # type: Optional[Tuple[int, int, int, int]]

    def draw(self, screen, font, x: int, y: int, width: int) -> int:
        import pygame
        # Dimensions
        box_size = 14
        row_height = max(box_size, font.get_height()) + 6
        # Draw checkbox outline
        box_rect = pygame.Rect(x, y + (row_height - box_size) // 2, box_size, box_size)
        pygame.draw.rect(screen, (60, 60, 60), box_rect, 1)
        # Fill box if selected
        if self.is_on:
            inner = box_rect.inflate(-4, -4)
            pygame.draw.rect(screen, (100, 100, 100), inner)
        # Draw label next to checkbox
        label_surface = font.render(self.label, True, (30, 30, 30))
        screen.blit(label_surface, (x + box_size + 8, y + (row_height - label_surface.get_height()) // 2))
        # Update rect for event handling: full row clickable
        self.rect = pygame.Rect(x, y, width, row_height)
        return y + row_height

    def handle_event(self, event) -> bool:
        import pygame
        if not self.rect:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_on = not self.is_on
                return True
        return False


class _CountSlider:
    """Simple horizontal slider for adjusting integer counts.

    This slider draws a track with a movable knob and a small
    numeric label showing the current value.  It keeps a direct
    reference to the corresponding FacilitySpec so that adjusting
    the slider automatically updates the spec.count value.  The
    slider supports click and drag interactions.
    """

    def __init__(self, spec: FacilitySpec, min_val: int = 0, max_val: int = 10) -> None:
        self.spec = spec
        self.min = int(min_val)
        self.max = int(max_val)
        self.value = int(spec.count)
        self.dragging = False
        # Bounding box used for interaction; updated in draw()
        self.rect: Optional[pygame.Rect] = None

    def get_min_height(self, font) -> int:
        """Return a reasonable minimum height for the slider row.

        The slider's height should at least accommodate the font
        height plus some padding and the knob diameter.  Returning
        24 ensures the control remains legible at small sizes.
        """
        return max(font.get_height() + 6, 24)

    def draw(self, screen, font, x: int, y: int, width: int, height: int) -> None:
        import pygame
        # Determine geometry for the track and knob
        track_h = 6
        track_y = int(y + height / 2 - track_h / 2)
        track_rect = pygame.Rect(x, track_y, width, track_h)
        # Draw track
        pygame.draw.rect(screen, (220, 220, 220), track_rect, border_radius=3)
        # Determine knob position (centre of knob)
        t = 0.0 if self.max == self.min else (self.value - self.min) / (self.max - self.min)
        knob_x = x + t * width
        knob_r = min(7, height // 4)
        knob_y = track_y + track_h // 2
        pygame.draw.circle(screen, (80, 80, 80), (int(knob_x), int(knob_y)), knob_r)
        # Draw value text to the right of slider
        val_surf = font.render(str(int(self.value)), True, (30, 30, 30))
        screen.blit(val_surf, (x + width + 4, y + (height - val_surf.get_height()) // 2))
        # Update bounding rect (includes slider and value text)
        total_w = width + 4 + val_surf.get_width()
        self.rect = pygame.Rect(x, y, total_w, height)

    def _update_from_mouse(self, mx: int, x: int, w: int) -> None:
        # Compute normalised position along the track [0,1]
        t = (mx - x) / max(1, w)
        t = max(0.0, min(1.0, t))
        raw = self.min + t * (self.max - self.min)
        # Round to nearest integer value
        val = int(round(raw))
        val = max(self.min, min(self.max, val))
        if val != self.value:
            self.value = val
            # Update the linked FacilitySpec count
            self.spec.count = val

    def handle_event(self, event, x: int, width: int) -> bool:
        import pygame
        if not self.rect:
            return False
        consumed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                # Immediately update value on click
                self._update_from_mouse(event.pos[0], x, width)
                consumed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                consumed = True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Update value while dragging
            self._update_from_mouse(event.pos[0], x, width)
            consumed = True
        return consumed


class FacilitiesUI:
    """Left-hand UI panel containing checkboxes for facility types."""
    def __init__(self, rect: Tuple[int, int, int, int], facility_names: List[str], specs: Optional[Dict[str, FacilitySpec]] = None) -> None:
        """Create a UI panel for controlling facilities.

        Parameters
        ----------
        rect : Tuple[int,int,int,int]
            The screen rectangle defining the panel position and size.
        facility_names : List[str]
            A list of facility names to display in the panel.
        specs : Optional[Dict[str, FacilitySpec]]
            A mapping from facility names to their corresponding
            FacilitySpec objects.  When provided, this allows the
            sliders to directly update the correct spec.  If omitted,
            dummy specs will be created for each facility.  Passing
            the spec mapping is recommended to avoid accidental
            mismatches.
        """
        import pygame
        self.rect = pygame.Rect(rect)
        self.font = None
        self.scroll = 0
        self.inner_h = 0
        # Build controls: each facility has a checkbox and a count slider
        self.controls: List[Tuple[_Checkbox, _CountSlider]] = []
        # Provide a fallback for missing specs
        specs = specs or {}
        # Keep a reference to the specs mapping so that new facilities
        # registered here are reflected in the FacilitiesSystem.  The
        # mapping is passed in from the facilities system on
        # construction and therefore modifications propagate to the
        # generator.
        self.specs = specs
        for name in facility_names:
            cb = _Checkbox(name, is_on=True)
            spec_obj = specs.get(name)
            if spec_obj is None:
                # Create a dummy spec with a default count and scale
                spec_obj = FacilitySpec(name=name, scale=1.0)
            slider = _CountSlider(spec_obj, min_val=0, max_val=10)
            self.controls.append((cb, slider))
        # Maintains the on/off state of each facility by name
        self.enabled: Dict[str, bool] = {name: True for name in facility_names}
        # Track whether UI changes require facilities regeneration.  Sliders
        # and checkboxes set this flag when modified; the main event
        # loop triggers generation on mouse release.  Without this
        # guard the city may freeze while dragging sliders because
        # regeneration is triggered on every mouse move.
        self.await_regen: bool = False

        # --- custom facility creation ---
        # A small "add facility" button appears at the bottom of the
        # panel.  When clicked it enters a text entry mode allowing
        # the user to type a new facility name.  Pressing Enter will
        # commit the new facility; Escape cancels the entry.  The
        # resulting facility will have a random pastel colour and a
        # default scale/count similar to other civic structures.
        self._add_button_rect = None  # type: Optional[pygame.Rect]
        self._adding_custom = False
        self._custom_text: str = ""

    def set_font(self, font) -> None:
        self.font = font

    def draw_bg(self, screen) -> None:
        import pygame
        # Draw background and a subtle border on the right
        pygame.draw.rect(screen, (250, 250, 250), self.rect)
        pygame.draw.line(screen, (200, 200, 200), (self.rect.right, self.rect.top), (self.rect.right, self.rect.bottom), 2)

    def draw(self, screen) -> None:
        import pygame
        if not self.font:
            return
        x = self.rect.left + 16
        y = self.rect.top + 16 - self.scroll
        w = self.rect.width - 32
        # Clip drawing to within the panel
        prev_clip = screen.get_clip()
        screen.set_clip(self.rect)
        self.inner_h = 0
        for cb, slider in self.controls:
            # The row is split into a label region for the checkbox and a
            # slider region.  Allocate a third of the width for the
            # slider and the remainder for the label and checkbox.
            slider_w = max(80, w // 3)
            label_w = w - slider_w - 8  # leave a small gap
            # Draw the checkbox and label in the left portion of the row
            row_end_y = cb.draw(screen, self.font, x, y, label_w)
            # Determine row height based on the checkbox; ensure
            # sufficient height for the slider
            row_h = row_end_y - y
            row_h = max(row_h, slider.get_min_height(self.font))
            # Draw the slider on the right side of the row
            slider.draw(screen, self.font, x + label_w + 8, y, slider_w, row_h)
            # Compute updated y for next row
            y = y + row_h
            self.inner_h = max(self.inner_h, y - (self.rect.top - self.scroll))
        # After drawing all facility rows, append an extra row for
        # creating custom facilities.  This row either displays a button
        # inviting the user to add a facility or, when in text entry
        # mode, shows an input field capturing the new name.
        import pygame
        # Determine row dimensions
        add_row_h = max(20, self.font.get_height() + 8)
        add_y = y
        # Determine x and available width (same as above)
        add_x = x
        add_w = w
        if self._adding_custom:
            # Draw input box background and border
            box_rect = pygame.Rect(add_x, add_y, add_w, add_row_h)
            pygame.draw.rect(screen, (230, 230, 230), box_rect, border_radius=4)
            pygame.draw.rect(screen, (180, 180, 180), box_rect, 1, border_radius=4)
            # Compose prompt and current text
            prompt = "Enter name: "
            text = prompt + self._custom_text
            text_surf = self.font.render(text, True, (30, 30, 30))
            # If too long, trim the left part so the end remains visible
            max_w = add_w - 8
            if text_surf.get_width() > max_w:
                trim = 0
                while trim < len(text) and self.font.size(text[trim:])[0] > max_w:
                    trim += 1
                trimmed = text[trim:]
                text_surf = self.font.render(trimmed, True, (30, 30, 30))
            screen.blit(text_surf, (add_x + 4, add_y + (add_row_h - text_surf.get_height()) // 2))
            # When in input mode, treat the entire box as the interactive area
            self._add_button_rect = box_rect
        else:
            # Draw add button with plus icon and label
            btn_rect = pygame.Rect(add_x, add_y, add_w, add_row_h)
            # Determine hover to adjust background colour
            mx, my = pygame.mouse.get_pos()
            hover = btn_rect.collidepoint(mx, my)
            bg_col = (240, 240, 240) if hover else (250, 250, 250)
            pygame.draw.rect(screen, bg_col, btn_rect)
            pygame.draw.rect(screen, (200, 200, 200), btn_rect, 1)
            # Render plus sign and label
            plus_surf = self.font.render("+", True, (60, 60, 60))
            label_surf = self.font.render("Add facility", True, (60, 60, 60))
            # Position plus and label with some padding
            pad = 4
            px = add_x + pad
            py = add_y + (add_row_h - plus_surf.get_height()) // 2
            screen.blit(plus_surf, (px, py))
            lx = px + plus_surf.get_width() + 6
            ly = add_y + (add_row_h - label_surf.get_height()) // 2
            screen.blit(label_surf, (lx, ly))
            # Store the button rect for click handling
            self._add_button_rect = btn_rect
        # Update y and inner height to include the add row
        y = add_y + add_row_h
        self.inner_h = max(self.inner_h, y - (self.rect.top - self.scroll))
        screen.set_clip(prev_clip)

    def handle_event(self, event) -> bool:
        import pygame
        # Custom facility entry mode takes precedence over all other
        # interactions.  When typing a name the UI captures keyboard
        # events and only exits when the user presses Enter, Escape or
        # clicks outside the input box.  No scrolling or other
        # interactions occur while in this mode.
        if self._adding_custom:
            # Handle keyboard input
            if event.type == pygame.KEYDOWN:
                # Commit the new facility on Enter
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    name = self._custom_text.strip()
                    if name:
                        self._add_custom_facility(name)
                        # Mark that regeneration should occur and
                        # synthesise a mouse‑up event to trigger it
                        self.await_regen = True
                        try:
                            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, {"button": 1, "pos": (self.rect.left, self.rect.top)}))
                        except Exception:
                            pass
                    # Reset entry state
                    self._custom_text = ""
                    self._adding_custom = False
                    return True
                # Cancel entry on Escape
                elif event.key == pygame.K_ESCAPE:
                    self._custom_text = ""
                    self._adding_custom = False
                    return True
                # Handle backspace
                elif event.key == pygame.K_BACKSPACE:
                    if len(self._custom_text) > 0:
                        self._custom_text = self._custom_text[:-1]
                    return True
                else:
                    # Append printable unicode characters
                    if event.unicode and event.unicode.isprintable():
                        self._custom_text += event.unicode
                    return True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Click outside the input box cancels the entry
                if self._add_button_rect and not self._add_button_rect.collidepoint(event.pos):
                    self._custom_text = ""
                    self._adding_custom = False
                    return True
                else:
                    # Click inside the input area is consumed but does not
                    # perform any action
                    return True
            else:
                # Other events are ignored while entering text
                return False

        consumed = False
        # Scroll handling (when not adding a custom facility)
        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(*pygame.mouse.get_pos()):
                max_scroll = max(0, self.inner_h - self.rect.height)
                self.scroll = max(0, min(max_scroll, self.scroll - event.y * 20))
                return True
        # Only handle clicks within the panel for other events
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            # If click is outside panel, ignore unless we are adding custom
            if not self.rect.collidepoint(*event.pos):
                return False
        # Check add button click before updating control rectangles
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._add_button_rect and self._add_button_rect.collidepoint(event.pos):
                # Enter custom facility entry mode
                self._adding_custom = True
                self._custom_text = ""
                return True
        # Compute positions and update control rectangles on the fly
        x = self.rect.left + 16
        y = self.rect.top + 16 - self.scroll
        w = self.rect.width - 32
        for cb, slider in self.controls:
            # Determine widths consistent with draw()
            slider_w = max(80, w // 3)
            label_w = w - slider_w - 8
            # Estimate row height using font height and slider minimum height
            row_h = max(max(14, self.font.get_height()) + 6, slider.get_min_height(self.font))
            # Update checkbox and slider rectangles for hit testing
            cb.rect = pygame.Rect(x, y, label_w, row_h)
            slider.rect = pygame.Rect(x + label_w + 8, y, slider_w + 40, row_h)
            # Prioritise slider dragging
            if slider.handle_event(event, x + label_w + 8, slider_w):
                consumed = True
                self.await_regen = True
            # Check checkbox toggle
            if cb.handle_event(event):
                self.enabled[cb.label] = cb.is_on
                consumed = True
                self.await_regen = True
            y += row_h
        # Handle events outside of controls (e.g. releasing slider) do nothing special
        return consumed

    # Internal helper: register a new custom facility with the given
    # human-readable name.  A new FacilitySpec is created and added
    # to the shared specs mapping as well as to the UI control list.
    # Colours are chosen randomly from a pastel palette so that each
    # custom facility is distinct yet not overly saturated.  If a
    # facility with the same name already exists the request is
    # ignored.
    def _add_custom_facility(self, name: str) -> None:
        name = name.strip()
        if not name:
            return
        # If the facility already exists, skip creation
        if name in self.enabled:
            return
        import random
        # Generate a pastel colour by mixing random values with a light base
        base = 200
        r = base + random.randint(-30, 55)
        g = base + random.randint(-30, 55)
        b = base + random.randint(-30, 55)
        # Clamp to [0,255]
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        colour = (r, g, b)
        # Create a new FacilitySpec for this custom facility.  Use a
        # slightly larger scale than a regular house to make the
        # facility noticeable but not enormous.  Default count to 2 so
        # that new facilities appear by default.
        spec = FacilitySpec(name=name, scale=2.5, color=colour, kind="building", count=2)
        # Register the spec with the underlying system via the shared
        # specs mapping
        self.specs[name] = spec
        # Create UI controls (checkbox + slider)
        cb = _Checkbox(name, is_on=True)
        slider = _CountSlider(spec, min_val=0, max_val=10)
        self.controls.append((cb, slider))
        # Mark it as enabled by default
        self.enabled[name] = True