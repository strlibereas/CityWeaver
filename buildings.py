
import random, math
from quadtree import Quadtree
from geometry import rect_from_axes, poly_aabb, polys_intersect, road_tube_poly

HOUSE_FILL   = (214, 219, 224)
HOUSE_STROKE = (120, 130, 140)

class BuildingSystem:
    def remove_inside_polys(self, clip_polys):
        """Remove buildings whose centroid is inside (or that intersect) any polygon in clip_polys."""
        if not clip_polys:
            return 0
        from geometry import polys_intersect, point_in_poly, poly_aabb
        removed, kept = 0, []
        clips = [(P, poly_aabb(P)) for P in clip_polys if P]
        for b in self.buildings:
            hp = b.get("poly")
            if not hp:
                kept.append(b); continue
            ha = b.get("aabb") or poly_aabb(hp)
            # building centroid (quad assumed)
            cx = (hp[0][0]+hp[1][0]+hp[2][0]+hp[3][0]) * 0.25
            cy = (hp[0][1]+hp[1][1]+hp[2][1]+hp[3][1]) * 0.25
            hit = False
            for P, Pa in clips:
                if not P or not Pa: 
                    continue
                # AABB reject
                if ha[2] < Pa[0] or Pa[2] < ha[0] or ha[3] < Pa[1] or Pa[3] < ha[1]:
                    continue
                if point_in_poly((cx, cy), P) or polys_intersect(hp, P):
                    hit = True
                    break
            if hit:
                removed += 1
            else:
                kept.append(b)
        self.buildings = kept
        return removed

    def __init__(self, map_size, params):
        self.MAP_WIDTH, self.HEIGHT = map_size
        self.params = params
        self.buildings=[]

    def reset(self):
        self.buildings.clear()

    def _candidate_ok(self, poly, building_qt, road_system, water):
        # Ensure all vertices lie within the map bounds
        for x, y in poly:
            if not (0 <= x < self.MAP_WIDTH and 0 <= y < self.HEIGHT):
                return False
        # Enforce that buildings stay within city walls, if they are enabled.
        try:
            import walls as _walls
            state = _walls.get_wall_state() if hasattr(_walls, 'get_wall_state') else {}
            enabled = bool(state.get('enabled'))
            wall_poly = state.get('poly') or []
            if enabled and wall_poly:
                from geometry import point_in_poly as _pip
                for (vx, vy) in poly:
                    if not _pip((vx, vy), wall_poly):
                        return False
                # Check centroid to avoid sliver placements
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
                if not _pip((cx, cy), wall_poly):
                    return False
        except Exception:
            # If wall checking fails, allow the building
            pass
        aabb = poly_aabb(poly)
        nearby=[]; building_qt.query(aabb, nearby)
        for other in nearby:
            other_poly = other["poly"]
            if polys_intersect(poly, other_poly):
                return False
        pad = max(2, int(sum(road_system.params["ROAD_WIDTH"].values())/4))
        road_rect=(aabb[0]-pad, aabb[1]-pad, aabb[2]+pad*2, aabb[3]+pad*2)
        cand=[]; road_system.qt.query(road_rect, cand)
        for s in cand:
            if not getattr(s,"active",False):
                continue
            tube = road_tube_poly(s.a, s.b, road_system.params["ROAD_WIDTH"][s.level])
            if polys_intersect(poly, tube):
                return False
        if water and water.river_poly:
            from geometry import polys_intersect as _pi, point_in_poly as _pip
            if _pi(poly, water.river_poly):
                # Relaxed rule near river: reject only if the centroid is in water
                cx = sum(p[0] for p in poly) / len(poly)
                cy = sum(p[1] for p in poly) / len(poly)
                if _pip((cx,cy), water.river_poly):
                    return False
        if water and water.sea_poly:
            from geometry import polys_intersect as _pi
            if _pi(poly, water.sea_poly):
                return False
        return True

    def generate(self, road_system, water):
        self.buildings.clear()
        if not road_system.segments: return
        bqt = Quadtree((0,0,self.MAP_WIDTH,self.HEIGHT))
        scale = max(0.05, float(self.params['HOUSE_SCALE']))
        frontage0 = self.params['HOUSE_FRONTAGE_BASE'] * scale
        depth0    = self.params['HOUSE_DEPTH_BASE'] * scale
        gap_min   = max(0.0, float(self.params['HOUSE_STEP_MIN']))
        setback_add = float(self.params['HOUSE_OFFSET_BASE'])
        for s in road_system.segments:
            if not s.active: continue
            a, b = s.a, s.b
            dx, dy = b[0]-a[0], b[1]-a[1]
            L = (dx*dx+dy*dy)**0.5
            if L < 4: continue
            ux, uy = dx/(L or 1.0), dy/(L or 1.0)
            px, py = uy, -ux
            setback = road_system.params['ROAD_WIDTH'][s.level] + setback_add
            for side in (+1, -1):
                a_off = (a[0] + px*setback*side, a[1] + py*setback*side)
                t = 0.0
                while t < L:
                    frontage = max(4.0, frontage0 * random.uniform(0.8, 1.2))
                    depth    = max(6.0, depth0    * random.uniform(0.8, 1.2))
                    if t + frontage > L: break
                    mid_t = t + frontage * 0.5
                    cx = a_off[0] + ux * mid_t + px * side * (depth * 0.5)
                    cy = a_off[1] + uy * mid_t + py * side * (depth * 0.5)
                    tvec = (ux, uy)
                    nvec = (px*side, py*side)
                    poly = rect_from_axes((cx, cy), tvec, nvec, frontage, depth)
                    ok = self._candidate_ok(poly, bqt, road_system, water)
                    if (not ok) and water and water.river_poly:
                        tries = 0
                        d = depth
                        while tries < 4 and not ok and d > 6.0:
                            d *= 0.7
                            poly = rect_from_axes((cx, cy), tvec, nvec, frontage, d)
                            ok = self._candidate_ok(poly, bqt, road_system, water)
                            tries += 1
                    if ok:
                        bld={'poly':poly, 'aabb':poly_aabb(poly)}
                        self.buildings.append(bld); bqt.insert(bld['aabb'], bld)
                    t += frontage + gap_min * random.uniform(0.8, 1.2)
    def draw(self, screen, world_to_screen, cam_zoom):
        import pygame
        shadow_offset = (2, 2); shadow_color=(150,150,150)
        for b in self.buildings:
            # Skip houses that have been marked as removed by the facilities
            # system.  These houses have been replaced by special
            # buildings and should no longer render.
            if b.get("removed"):
                continue
            poly = b["poly"]
            pts = [world_to_screen(p) for p in poly]
            sh  = [(x+shadow_offset[0], y+shadow_offset[1]) for (x,y) in pts]
            pygame.draw.polygon(screen, shadow_color, sh)
            pygame.draw.polygon(screen, HOUSE_FILL, pts)
            pygame.draw.polygon(screen, HOUSE_STROKE, pts, max(1,int(2*cam_zoom)))

    def serialize_state(self):
        return {"buildings":[{"poly":[tuple(p) for p in b["poly"]], "aabb":tuple(b["aabb"])} for b in self.buildings]}
    def restore_state(self, state):
        self.buildings = []
        for b in state.get("buildings", []):
            self.buildings.append({"poly":[tuple(p) for p in b["poly"]], "aabb":tuple(b["aabb"])})