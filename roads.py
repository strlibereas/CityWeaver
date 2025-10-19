
import math, random
from collections import defaultdict
from quadtree import Quadtree
from geometry import seg_intersection, seg_aabb, point_seg_dist, road_tube_poly, segment_intersects_polygon, point_in_poly

ROAD_NAME_COLOR = (60,60,60)

class RoadSeg:
    __slots__=("a","b","level","active","batch")
    def __init__(self, a, b, level, batch):
        self.a=a; self.b=b; self.level=level; self.active=True; self.batch=batch

class RoadSystem:
    def _end_at_first_intersection(self, a, b):
        from geometry import seg_intersection
        query = seg_aabb(a, b)
        # Slightly expand query to catch near-misses
        x0,y0,x1,y1 = query; pad=2
        query = (x0-pad,y0-pad,x1+pad,y1+pad)
        for seg in self.qt.query(query):
            if not getattr(seg,'active',True):
                continue
            # ignore sharing the same batch immediate predecessor
            ok, P, t1, t2 = seg_intersection(a, b, seg.a, seg.b)
            if ok and P is not None and 0.0 < t1 < 1.0 and 0.0 < t2 < 1.0:
                return P
        return b

    def _nearest_segment_projection(self, p):
        """Return (seg, proj_point, dist, t) for nearest active segment to point p.
        Uses quadtree to search a small neighborhood.
        """
        (px,py) = p
        pad = 120.0
        query = (px-pad, py-pad, px+pad, py+pad)
        best = (None, None, 1e9, None)
        from geometry import point_seg_dist
        for seg in self.qt.query(query):
            if not getattr(seg, 'active', True):
                continue
            if seg in self.decorative_segments:
                continue
            d, pr, t = point_seg_dist(p, seg.a, seg.b, ret_proj=True)
            if d < best[2]:
                best = (seg, pr, d, t)
        return best

    def _angle_between(self, u, v):
        import math
        ux,uy=u; vx,vy=v
        nu = math.hypot(ux,uy) or 1.0
        nv = math.hypot(vx,vy) or 1.0
        dot = max(-1.0, min(1.0, (ux*vx+uy*vy)/(nu*nv)))
        return math.degrees(math.acos(dot))

    def _try_snap(self, a, b, level):
        """Snap b to nearest segment projection if close and aligned.
        Returns (new_b, snapped_bool).
        """
        seg, pr, d, t = self._nearest_segment_projection(b)
        if seg is None or pr is None:
            return b, False
        SNAP_DIST = max(18.0, self.params['ROAD_WIDTH'][self.map_lvl(level)] * 2.2)
        if d > SNAP_DIST:
            return b, False
        # alignment gate: avoid T-junctions that are too sharp
        vx,vy = (b[0]-a[0], b[1]-a[1])
        sx,sy = (seg.b[0]-seg.a[0], seg.b[1]-seg.a[1])
        ang = self._angle_between((vx,vy), (sx,sy))
        if ang > 70.0:  # too sharp; don't snap
            return b, False
        return pr, True

    def _steer_toward_network(self, a, b, level):
        """When near the network but not snap-worthy, bend b slightly toward it to avoid spirals."""
        seg, pr, d, t = self._nearest_segment_projection(b)
        if seg is None or pr is None:
            return b
        ATT_DIST = 140.0
        if d > ATT_DIST:
            return b
        blend = max(0.0, min(1.0, (ATT_DIST - d) / ATT_DIST)) * 0.45
        nb = (b[0]*(1.0-blend) + pr[0]*blend, b[1]*(1.0-blend) + pr[1]*blend)
        return nb

    def _reject_backtrack(self, a, b, prev_dir):
        """Cull near U-turns and tiny-angle jitter that cause swirling."""
        if prev_dir is None:
            return False
        vx,vy=(b[0]-a[0], b[1]-a[1])
        ang = self._angle_between(prev_dir, (vx,vy))
        return ang > 150.0 or ang < 6.0

    def _seg_len(self, seg):
        (ax,ay),(bx,by) = seg
        return math.hypot(bx-ax, by-ay)

    def _clip_to_gen_rect(self, a, b, margin: float = 2.0, min_len: float = 3.0):
        x0,y0,x1,y1 = self.gen_bounds
        # inward safety margin to avoid AA bleeding
        x0+=margin; y0+=margin; x1-=margin; y1-=margin
        edges = [((x0,y0),(x1,y0)), ((x1,y0),(x1,y1)), ((x1,y1),(x0,y1)), ((x0,y1),(x0,y0))]
        def _inside(p): return x0 <= p[0] <= x1 and y0 <= p[1] <= y1
        A,B=a,b
        if _inside(A) and _inside(B):
            if ( (B[0]-A[0])**2 + (B[1]-A[1])**2 )**0.5 < min_len: return A, None
            return A,B
        hits=[]
        from geometry import seg_intersection
        for e1,e2 in edges:
            ok,P,t,_ = seg_intersection(A,B,e1,e2)
            if ok and P is not None and 0.0<=t<=1.0: hits.append((t,P))
        if _inside(A) and not _inside(B):
            if hits: hits.sort(key=lambda x:x[0]); return A, hits[0][1]
            # drop slivers
            A_,B_ = A, hits[0][1]
            if ( (B_[0]-A_[0])**2 + (B_[1]-A_[1])**2 )**0.5 < min_len: return A, None
            return A_, B_
            return A, None
        if not _inside(A) and _inside(B):
            if hits: hits.sort(key=lambda x:x[0]); return hits[-1][1], B
            # drop slivers
            A_,B_ = hits[-1][1], B
            if ( (B_[0]-A_[0])**2 + (B_[1]-A_[1])**2 )**0.5 < min_len: return A, None
            return A_, B_
            return A, None
        if len(hits)>=2:
            hits.sort(key=lambda x:x[0]); return hits[0][1], hits[-1][1]
            # drop slivers
            A_,B_ = hits[0][1], hits[-1][1]
            if ( (B_[0]-A_[0])**2 + (B_[1]-A_[1])**2 )**0.5 < min_len: return A, None
            return A_, B_
        return A, None

    # --- global constraint kill-switch (river-safe) ---
    constraints_disabled: bool = False  # keep constraints off unless explicitly enabled

    def disable_constraints(self):
        self.constraints_disabled = True

    def enable_constraints(self):
        self.constraints_disabled = False

    # --- mutation lock to prevent rebranching during river/bridge generation ---
    _locked_mutations: bool = False

    def lock_mutations(self):
        self._locked_mutations = True

    def unlock_mutations(self):
        self._locked_mutations = False

    def _p(self, key, default=None): return self.params.get(key, default)

    def map_lvl(self, level:int) -> int:
        """Map generator level to effective category level.
        Default remap promotes each tier up one:
          0->0 (highways stay highways)
          1->0 (roads become highways)
          2->1 (tertiaries become roads)
          3->2 (alleys become tertiaries)
        Override via PARAMS['LEVEL_REMAP'] if present.
        """
        remap = self.params.get('LEVEL_REMAP', {0:0, 1:0, 2:1, 3:2})
        return int(remap.get(int(level), int(level)))

    def __init__(self, map_size, world_bounds, params, address_names):
        self.MAP_WIDTH, self.HEIGHT = map_size
        self.WORLD_X, self.WORLD_Y, self.WORLD_WIDTH, self.WORLD_HEIGHT = world_bounds
        self.params = params
        self.qt = Quadtree((self.WORLD_X, self.WORLD_Y, self.WORLD_WIDTH, self.HEIGHT*2))
        self.gen_bounds = (0.0, 0.0, float(self.MAP_WIDTH), float(self.HEIGHT))
        self.nodes=[]; self.segments=[]; self.road_graph=defaultdict(list)
        self.active=[]; self.batch_id=0; self.batch_order=[]
        self.decorative_segments=set()
        self.address_names = address_names or []
        self.segment_names = {}
        self.river_bridge_segments=set(); self.river_skip_segments=set(); self.sea_skip_segments=set()
        self.river_cross_count=0

    # -------- generation bounds --------
    def set_generation_bounds(self, bounds):
        if getattr(self, 'constraints_disabled', False):
            return None
        self.gen_bounds = tuple(bounds)
    def _inside_gen(self, pt):
        x0,y0,x1,y1 = self.gen_bounds
        return (x0 <= pt[0] <= x1) and (y0 <= pt[1] <= y1)

    # -------- graph helpers --------
    def add_node(self, p):
        self.nodes.append(p); return len(self.nodes)-1
    def connect_nodes(self, i, j):
        if j not in self.road_graph[i]: self.road_graph[i].append(j)
        if i not in self.road_graph[j]: self.road_graph[j].append(i)

    # -------- proximity / snapping --------
    def near_parallel_too_close(self, a, b, min_spacing_override=None):
        # IMPORTANT: this only affects manual segments when override is passed.
        ms = self.params["MIN_PARALLEL_SPACING"] if min_spacing_override is None else float(min_spacing_override)
        pad = ms + max(self.params["ROAD_WIDTH"].values())
        rect=(min(a[0],b[0])-pad, min(a[1],b[1])-pad, abs(a[0]-b[0])+pad*2, abs(a[1]-b[1])+pad*2)
        cand=[]; self.qt.query(rect, cand)
        av=(b[0]-a[0], b[1]-a[1]); la=math.hypot(*av) or 1.0
        cos_limit = math.cos(math.radians(self.params["MIN_PARALLEL_ANGLE_DEG"]))
        for seg in cand:
            if not isinstance(seg, RoadSeg) or not seg.active: continue
            eps=1.0
            # allow touching at endpoints
            if (math.hypot(a[0]-seg.a[0],a[1]-seg.a[1])<eps or
                math.hypot(a[0]-seg.b[0],a[1]-seg.b[1])<eps or
                math.hypot(b[0]-seg.a[0],b[1]-seg.a[1])<eps or
                math.hypot(b[0]-seg.b[0],b[1]-seg.b[1])<eps):
                continue
            bv=(seg.b[0]-seg.a[0], seg.b[1]-seg.a[1]); lb=math.hypot(*bv) or 1.0
            cosang=abs((av[0]*bv[0]+av[1]*bv[1])/(la*lb))
            if cosang>cos_limit:
                d_a,_,_ = point_seg_dist(a, seg.a, seg.b)
                d_b,_,_ = point_seg_dist(b, seg.a, seg.b)
                if d_a<ms or d_b<ms: return True
        return False

    def try_snap_node(self, point):
        if getattr(self, 'constraints_disabled', False):
            return False
        if getattr(self, '_locked_mutations', False):
            return None
        best_idx=None; best_d=1e9
        for i,p in enumerate(self.nodes):
            d=math.hypot(p[0]-point[0], p[1]-point[1])
            if d<best_d and d<=self.params["SNAP_RADIUS_NODE"]:
                best_d=d; best_idx=i
        if best_idx is not None: return self.nodes[best_idx], best_idx
        return point, None

    def try_snap_segment(self, point, extra_snap_segments=None):
        rect=(point[0]-self.params["SNAP_RADIUS_SEG"], point[1]-self.params["SNAP_RADIUS_SEG"],
              self.params["SNAP_RADIUS_SEG"]*2, self.params["SNAP_RADIUS_SEG"]*2)
        cand=[]; self.qt.query(rect,cand)
        best_point=None; best_seg=None; best_t=None; best_d=1e9
        for seg in cand:
            if not isinstance(seg, RoadSeg) or not seg.active: continue
            d,proj,t = point_seg_dist(point, seg.a, seg.b)
            if d<best_d and d<=self.params["SNAP_RADIUS_SEG"]:
                best_d=d; best_point=proj; best_seg=seg; best_t=t
        if extra_snap_segments:
            for (ra,rb) in extra_snap_segments:
                d,proj,t = point_seg_dist(point, ra, rb)
                if d<best_d and d<=self.params["SNAP_RADIUS_SEG"]:
                    best_d=d; best_point=proj; best_seg=None; best_t=None
        if best_point is None: return point, None, None
        return best_point, best_seg, best_t

    def split_segment_at_point(self, seg, P, tol=1e-3):
        if getattr(self, '_locked_mutations', False):
            return None
        _, proj, t = point_seg_dist(P, seg.a, seg.b)
        if t is None or t <= tol or t >= 1.0 - tol: return False
        seg.active = False
        s1 = RoadSeg(seg.a, proj, seg.level, seg.batch)
        s2 = RoadSeg(proj, seg.b, seg.level, seg.batch)
        s1a, s1b = self._clip_to_gen_rect(s1.a, s1.b)
        s2a, s2b = self._clip_to_gen_rect(s2.a, s2.b)
        if s1b is not None:
            s1.a, s1.b = s1a, s1b; self.segments.append(s1); self.qt.insert(seg_aabb(s1.a, s1.b), s1)
        if s2b is not None:
            s2.a, s2.b = s2a, s2b; self.segments.append(s2); self.qt.insert(seg_aabb(s2.a, s2.b), s2)

        if seg in self.segment_names:
            name = self.segment_names.pop(seg)
            self.segment_names[s1] = name; self.segment_names[s2] = name
        return True

    # -------- core add with bounds + water handling --------
    def add_segment(self, a, b, level, batch_id, water=None, min_spacing_override=None):
        if getattr(self, '_locked_mutations', False):
            return False
        # Clip to generation rectangle strictly
        a, b = self._clip_to_gen_rect(a, b)
        if b is None:
            return a, None
        if b is None:
            return a, None
         # for growth, or override=7 only for manual calls
        if self.near_parallel_too_close(a,b,min_spacing_override): return a,None

        # water logic
        # Always clip segments that would cross the river.  Previously
        # the road generator allowed every third crossing to span the
        # river by using ``river_cross_count``.  This behaviour
        # introduced unintended bridges between the two banks that
        # competed with manually generated bridge decks.  To ensure
        # that only explicit bridges traverse the river, we always
        # truncate segments at the first intersection with the river
        # polygon.  See the patched rebuild logic in
        # ``city_weaver.py`` for manual highway placement.
        skip_cross=False
        if water and water.river_poly and segment_intersects_polygon(a,b,water.river_poly):
            # increment for serialization (not used to gate crossings anymore)
            self.river_cross_count += 1
            # find the earliest intersection along the segment with the
            # river polygon and shorten the segment slightly before
            # that point.  Offsetting by a small amount prevents
            # immediate snapping of subsequent growth to the river.
            pts=[]; rp=water.river_poly
            for i in range(len(rp)):
                p1,p2 = rp[i], rp[(i+1)%len(rp)]
                ok, inter, t, _ = seg_intersection(a,b,p1,p2)
                if ok and inter is not None and 0.0<=t<=1.0:
                    pts.append((t, inter))
            if pts:
                pts.sort(key=lambda x:x[0])
                # earliest intersection point
                b_int = pts[0][1]
                # compute direction vector from a to intersection
                vx,vy = (b_int[0]-a[0], b_int[1]-a[1])
                L = math.hypot(vx,vy) or 1.0
                # back off slightly from the intersection to keep the
                # segment on land
                back_off = 2.0
                b = (b_int[0] - vx/L * back_off, b_int[1] - vy/L * back_off)
                skip_cross = True

        rect=seg_aabb(a,b); pad=4
        rect=(rect[0]-pad, rect[1]-pad, rect[2]+pad*2, rect[3]+pad*2)
        cand=[]; self.qt.query(rect,cand)
        hit_pt=None
        for seg in cand:
            if not isinstance(seg, RoadSeg) or not seg.active: continue
            hit,P,ta,tb = seg_intersection(a,b,seg.a,seg.b)
            if hit and 1e-4<ta<1-1e-4 and 1e-4<tb<1-1e-4: hit_pt=P; break

        end_pt = hit_pt if hit_pt else b
        end_pt, snapped_idx = self.try_snap_node(end_pt)
        if snapped_idx is None and not hit_pt:
            end_pt,_,_ = self.try_snap_segment(end_pt)

        start_pt, start_idx = self.try_snap_node(a)
        if start_idx is None: start_idx=self.add_node(start_pt)
        end_idx = snapped_idx if snapped_idx is not None else self.add_node(end_pt)
        self.connect_nodes(start_idx, end_idx)

        s = RoadSeg(start_pt, end_pt, level, batch_id)
        self.segments.append(s); self.qt.insert(seg_aabb(s.a,s.b), s)
        if self.address_names:
            self.segment_names[s] = random.choice(self.address_names)

        if water and water.river_poly and segment_intersects_polygon(s.a,s.b,water.river_poly):
            if skip_cross: self.river_skip_segments.add(s)
            else: self.river_bridge_segments.add(s)
        if water and water.sea_poly and segment_intersects_polygon(s.a,s.b,water.sea_poly):
            self.sea_skip_segments.add(s)

        return end_pt, end_idx

    def add_manual_segment(self, a, b, level=2, water=None, min_spacing_override=None):
        if getattr(self, '_locked_mutations', False):
            return False
        # Manual: we pass override=7 from caller; if None, it behaves like growth.
        self.batch_id += 1
        end_pt, end_idx = self.add_segment(a, b, level, self.batch_id, water=water, min_spacing_override=min_spacing_override)
        return (end_pt is not None and end_idx is not None)

    # -------- growth --------
    def next_heading(self, level, heading):
        drift = random.uniform(-max(0.35*self.params["CURVE"][self.map_lvl(level)], self.params["CURVE"][self.map_lvl(level)]*0.6), max(0.35*self.params["CURVE"][self.map_lvl(level)], self.params["CURVE"][self.map_lvl(level)]*0.6))
        return (heading + drift) % 360

    def spawn_seed(self, x, y, start_level=1):
        if getattr(self, 'constraints_disabled', False):
            return False
        if getattr(self, '_locked_mutations', False):
            return False
        self.batch_id += 1; self.batch_order.append(self.batch_id)
        seed=(x,y); seed_idx=self.add_node(seed); base_angle=random.uniform(0,360)
        for dir_deg in (base_angle, (base_angle+180)%360):
            remaining=random.randint(*self.params["LENGTH_RANGES"][start_level])
            self.active.append({"level":start_level,"heading":dir_deg,"pos":self.nodes[seed_idx],"remaining":remaining,"last_idx":seed_idx,"steps":0,"batch":self.batch_id})

    def add_active_seed(self, pos, heading_deg, level=1, length_override=None):
        """
        Insert a new growth branch starting at an arbitrary world-space
        position `pos` with a fixed heading.  This mirrors the behavior
        of `spawn_seed` but allows the caller to specify the initial
        orientation of the branch and optionally override the number of
        segments (`remaining`) to grow.  If `length_override` is None
        the length range is derived from the configured `LENGTH_RANGES`.

        The starting point is snapped to an existing node if within
        `SNAP_RADIUS_NODE`; otherwise a new node is created.  The new
        branch is appended to the active branch list and will be grown
        during subsequent calls to `step_branch`.  This function
        respects `constraints_disabled` and `_locked_mutations` to avoid
        inadvertent graph mutations.
        """
        if getattr(self, 'constraints_disabled', False):
            return False
        if getattr(self, '_locked_mutations', False):
            return False

        # Snap the seed point to the nearest existing node or create one
        start_pt, start_idx = self.try_snap_node(pos)
        if start_idx is None:
            start_idx = self.add_node(start_pt)

        # Begin a new batch for this seed
        self.batch_id += 1
        self.batch_order.append(self.batch_id)

        # Determine how many segments this branch should grow
        if length_override is None:
            lo, hi = self.params["LENGTH_RANGES"].get(level, (1, 1))
            remaining = random.randint(lo, hi)
        else:
            remaining = int(length_override)

        # Append the new branch to the active list
        self.active.append({
            "level": level,
            "heading": float(heading_deg) % 360.0,
            "pos": start_pt,
            "remaining": remaining,
            "last_idx": start_idx,
            "steps": 0,
            "batch": self.batch_id
        })
        return True

    def step_branch(self, br, water=None):
        if br["remaining"]<=0: return []
        br["heading"]=self.next_heading(br["level"], br["heading"])
        rad=math.radians(br["heading"])
        seg_len=[self.params["PRIMARY_SEGMENT_LENGTH"], self.params["SECONDARY_SEGMENT_LENGTH"], self.params["TERTIARY_SEGMENT_LENGTH"], self.params["ALLEY_SEGMENT_LENGTH"]][min(br["level"],3)]
        dirv=(math.cos(rad)*seg_len, -math.sin(rad)*seg_len)
        a=br["pos"]; b=(a[0]+dirv[0], a[1]+dirv[1])
        if not self._inside_gen(b): br["remaining"]=0; return []
        new_end, end_idx = self.add_segment(a, b, br["level"], br["batch"], water=water)  # <-- no override
        if end_idx is None: br["remaining"]=0; return []
        br["pos"]=new_end; br["last_idx"]=end_idx; br["remaining"]-=1; br["steps"]=br.get("steps",0)+1
        children=[]; next_level=br["level"]+1
        if br["remaining"]>0 and next_level<=3:
            delays=self._p("BRANCH_DELAY",{0:0,1:1,2:2,3:3}); probs=self._p("BRANCH_PROB",{0:1.0,1:0.9,2:0.7})
            base_angle=float(self._p("BRANCH_BASE_ANGLE",90)); jitter_mag=float(self._p("BRANCH_JITTER",2)); bidir=float(self._p("BIDIR_BRANCH_PROB",1.0))
            allowed = br["steps"] >= delays.get(br["level"], 0); prob = float(probs.get(br["level"],0.0))
            if allowed and (prob>=1.0 or random.random()<prob):
                jitter = random.uniform(-jitter_mag, jitter_mag)
                h1 = (br["heading"] + base_angle + jitter) % 360.0
                rng = self.params["LENGTH_RANGES"][next_level]
                children.append({"level":next_level,"heading":h1,"pos":new_end,"remaining":random.randint(*rng),"last_idx":end_idx,"steps":0,"batch":br["batch"]})
                if random.random() < bidir:
                    h2 = (br["heading"] - base_angle + jitter) % 360.0
                    children.append({"level":next_level,"heading":h2,"pos":new_end,"remaining":random.randint(*rng),"last_idx":end_idx,"steps":0,"batch":br["batch"]})
        return children

    
    # -------- cutting / deletion --------
    def remove_segments_crossing_line(self, a, b, level_filter=(1,), tube=None):
        from geometry import seg_intersection, point_seg_dist, seg_aabb
        if tube is None:
            try:
                if level_filter is None:
                    widths = list(self.params["ROAD_WIDTH"].values())
                else:
                    widths = [self.params["ROAD_WIDTH"][lv] for lv in level_filter if lv in self.params["ROAD_WIDTH"]]
                base = max(widths) if widths else 6.0
            except Exception:
                base = 6.0
            tube = max(6.0, float(base)*0.8)
        rect = seg_aabb(a,b); pad = tube + 12.0
        rect = (rect[0]-pad, rect[1]-pad, rect[2]+pad*2, rect[3]+pad*2)
        cand=[]; 
        try:
            self.qt.query(rect,cand)
        except Exception:
            cand=[]
        if not cand:
            cand = list(self.segments)
        removed = 0
        for seg in cand:
            if not isinstance(seg, RoadSeg): continue
            if not seg.active: continue
            if level_filter is not None:
                try:
                    if getattr(seg, "level", None) not in set(level_filter): 
                        continue
                except TypeError:
                    if getattr(seg, "level", None) != level_filter: 
                        continue
            hit, P, ta, tb = seg_intersection(a, b, seg.a, seg.b)
            if hit and P is not None and 0.0 <= ta <= 1.0 and 0.0 <= tb <= 1.0:
                seg.active=False
            else:
                d, _, _ = point_seg_dist(a, seg.a, seg.b)
                d2, _, _ = point_seg_dist(b, seg.a, seg.b)
                mx = (a[0]+b[0])*0.5; my=(a[1]+b[1])*0.5
                dm, _, _ = point_seg_dist((mx,my), seg.a, seg.b)
                if min(d,d2,dm) <= tube:
                    seg.active=False
                else:
                    S = max(6, int((((b[0]-a[0])**2 + (b[1]-a[1])**2 )**0.5) / max(8.0, tube)))
                    close=False
                    for i in range(S+1):
                        t=i/float(S)
                        px=a[0]+(b[0]-a[0])*t; py=a[1]+(b[1]-a[1])*t
                        dd,_,_ = point_seg_dist((px,py), seg.a, seg.b)
                        if dd <= tube:
                            close=True; break
                    if close:
                        seg.active=False
            if not seg.active:
                if seg in self.segment_names:
                    self.segment_names.pop(seg, None)
                if seg in self.decorative_segments:
                    try: self.decorative_segments.remove(seg)
                    except KeyError: pass
                removed += 1
        return removed
    
    # -------- drawing --------
    

    def get_classed(self):
        """Return a dict of category -> list of polylines, using the current (remapped) levels.
        Categories: 'highways', 'roads', 'tertiaries', 'alleys'.
        Each polyline is a list of (x,y) tuples from each segment.
        """
        cat_map = {0:'highways', 1:'roads', 2:'tertiaries', 3:'alleys'}
        classed = {'highways':[], 'roads':[], 'tertiaries':[], 'alleys':[]}
        try:
            for s in self.segments:
                if not getattr(s, 'active', True):
                    continue
                lvl = getattr(s, 'level', 1)
                try:
                    eff = int(self.map_lvl(int(lvl)))
                except Exception:
                    eff = int(lvl)
                cat = cat_map.get(max(0, min(3, eff)), 'roads')
                classed[cat].append([tuple(s.a), tuple(s.b)])
        except Exception:
            pass
        # also expose on the instance so external collectors can read it
        try:
            self.classed = classed
        except Exception:
            pass
        return classed
def draw_roads(self, screen, world_to_screen, cam_zoom, colors, water=None, aa=True, show_names=False):
        import pygame
        col = colors["ROAD"]
        for level in (0,1,2,3):
            w = self.params["ROAD_WIDTH"][self.map_lvl(level)]
            for s in self.segments:
                if not s.active or s.level!=level: continue
                if s in self.decorative_segments: continue
                if water and water.sea_poly and s in self.sea_skip_segments: continue
                a,b = s.a, s.b
                segs=[(a,b)]
                if water and water.river_poly and s in self.river_skip_segments:
                    segs = clip_segment_outside(a,b, water.river_poly)
                # Clip to map rectangle so nothing draws outside
                rect = (0.0, 0.0, float(self.MAP_WIDTH), float(self.HEIGHT))
                clipped_segs = []
                for (pa,pb) in segs:
                    c = _clip_segment_to_rect(pa, pb, rect)
                    if c: clipped_segs.append(c)
                segs = [seg for seg in clipped_segs if seg and self._seg_len(seg) >= max(2.0, self.params['ROAD_WIDTH'][self.map_lvl(level)]*0.6)]

                is_bridge = (water and water.river_poly and s in self.river_bridge_segments)
                for A,B in segs:
                    L = math.hypot(B[0]-A[0], B[1]-A[1])
                    if L < 8: continue
                    p1 = world_to_screen(A); p2 = world_to_screen(B)
                    width_px = max(1, int(w*cam_zoom))
                    if is_bridge and water and water.river_poly:
                        rp = water.river_poly; inters=[]
                        for i in range(len(rp)):
                            c=rp[i]; d=rp[(i+1)%len(rp)]
                            hit,pt,t,_ = seg_intersection(A,B,c,d)
                            if hit and pt is not None and 0.0<=t<=1.0: inters.append((t,pt))
                        if len(inters)>=2:
                            inters.sort(key=lambda x:x[0])
                            sp1 = world_to_screen(inters[0][1]); sp2 = world_to_screen(inters[-1][1])
                            pygame.draw.line(screen, (80,80,80), sp1, sp2, max(1,int((w+8)*cam_zoom)))
                            pygame.draw.line(screen, col, sp1, sp2, max(1,int((w+4)*cam_zoom)))
                    if aa and width_px<=2: pygame.draw.aaline(screen, col, p1, p2, True)
                    else: pygame.draw.line(screen, col, p1, p2, width_px)

        if show_names and self.segment_names:
            drawn=set(); min_len=30
            for s in self.segments:
                if not s.active: continue
                b_id = getattr(s, "batch", None)
                if b_id is None or b_id in drawn: continue
                if water and water.sea_poly and s in self.sea_skip_segments: continue
                a,b = s.a, s.b
                if math.hypot(b[0]-a[0], b[1]-a[1]) < min_len: continue
                mid = ((a[0]+b[0])*0.5, (a[1]+b[1])*0.5)
                if water and water.river_poly and point_in_poly(mid, water.river_poly): continue
                if water and water.sea_poly and point_in_poly(mid, water.sea_poly): continue
                name = self.segment_names.get(s)
                if not name: continue
                sx,sy = world_to_screen(mid)
                ang = math.degrees(math.atan2(-(b[1]-a[1]), (b[0]-a[0])))
                size = max(6, int(24 * cam_zoom * 0.4))
                font = pygame.font.Font(None, size)
                surf = font.render(name, True, ROAD_NAME_COLOR)
                rot = pygame.transform.rotate(surf, ang)
                rect = rot.get_rect(center=(sx,sy))
                screen.blit(rot, rect); drawn.add(b_id)

def clip_segment_outside(a,b,poly):
    from geometry import seg_intersection, point_in_poly
    pts=[]; n=len(poly)
    for i in range(n):
        p1=poly[i]; p2=poly[(i+1)%n]
        ok, inter, t, _ = seg_intersection(a,b,p1,p2)
        if ok and inter is not None and 0.0<=t<=1.0: pts.append((t, inter))
    pts.sort(key=lambda x:x[0])
    inside = point_in_poly(a, poly)
    out=[]; t_prev=0.0
    for t_curr, P in pts:
        if not inside:
            start = (a[0] + (b[0]-a[0])*t_prev, a[1] + (b[1]-a[1])*t_prev)
            out.append((start, P))
        inside = not inside; t_prev = t_curr
    if not inside:
        end=b; start=(a[0]+(b[0]-a[0])*t_prev, a[1]+(b[1]-a[1])*t_prev)
        out.append((start, end))
    return out

# ---- state (for undo) ----
def _road_serialize(self):
    return {
        "nodes": [(p[0], p[1]) for p in self.nodes],
        "segments": [((s.a[0], s.a[1]), (s.b[0], s.b[1]), s.level, s.active, s.batch) for s in self.segments],
        "active": [dict(level=br["level"], heading=br["heading"], pos=(br["pos"][0],br["pos"][1]),
                        remaining=br["remaining"], last_idx=br["last_idx"], steps=br.get("steps",0), batch=br["batch"]) for br in self.active],
        "road_graph": {int(k): list(v) for k,v in self.road_graph.items()},
        "batch_id": self.batch_id,
        "batch_order": list(self.batch_order),
        "segment_names": {i: name for i, name in enumerate([self.segment_names.get(s) for s in self.segments])},
        "river_bridge": [i for i,s in enumerate(self.segments) if s in self.river_bridge_segments],
        "river_skip":   [i for i,s in enumerate(self.segments) if s in self.river_skip_segments],
        "sea_skip":     [i for i,s in enumerate(self.segments) if s in self.sea_skip_segments],
        "river_cross_count": self.river_cross_count,
        "gen_bounds": tuple(self.gen_bounds),
    }

def _road_restore(self, state):
    self.nodes = [tuple(p) for p in state.get("nodes", [])]
    self.segments = []
    self.qt = Quadtree((self.WORLD_X, self.WORLD_Y, self.WORLD_WIDTH, self.HEIGHT * 2))
    self.road_graph = defaultdict(list)
    for k, v in state.get("road_graph", {}).items():
        self.road_graph[int(k)] = list(v)
    self.batch_id = int(state.get("batch_id", 0))
    self.batch_order = list(state.get("batch_order", []))
    self.active = [dict(level=br["level"], heading=br["heading"], pos=tuple(br["pos"]),
                        remaining=br["remaining"], last_idx=br["last_idx"],
                        steps=br.get("steps",0), batch=br["batch"]) for br in state.get("active",[])]
    segs = state.get("segments", [])
    idx_to_seg = {}
    for i,(a,b,level,active,batch) in enumerate(segs):
        s = RoadSeg(tuple(a), tuple(b), int(level), int(batch))
        s.active = bool(active)
        self.segments.append(s)
        idx_to_seg[i] = s
        self.qt.insert(seg_aabb(s.a, s.b), s)
    self.segment_names = {}
    name_map = state.get("segment_names", {})
    for i, name in name_map.items():
        s = idx_to_seg.get(i)
        if s and name: self.segment_names[s] = name
    self.river_bridge_segments = set(idx_to_seg[i] for i in state.get("river_bridge", []) if i in idx_to_seg)
    self.river_skip_segments   = set(idx_to_seg[i] for i in state.get("river_skip", []) if i in idx_to_seg)
    self.sea_skip_segments     = set(idx_to_seg[i] for i in state.get("sea_skip", []) if i in idx_to_seg)
    self.river_cross_count = int(state.get("river_cross_count", 0))
    self.gen_bounds = tuple(state.get("gen_bounds", self.gen_bounds))

RoadSystem.serialize_state = _road_serialize
RoadSystem.restore_state  = _road_restore

from water import rebuild_roads_around_river as _rebuild_river
RoadSystem.rebuild_around_river = lambda self, water, bank_offset: _rebuild_river(self, water, bank_offset)


def _clip_segment_to_rect(a,b,rect):
    from geometry import seg_intersection
    x0,y0,x1,y1 = rect
    edges = [((x0,y0),(x1,y0)), ((x1,y0),(x1,y1)), ((x1,y1),(x0,y1)), ((x0,y1),(x0,y0))]
    def _inside(p): return x0 <= p[0] <= x1 and y0 <= p[1] <= y1
    A,B=a,b
    if _inside(A) and _inside(B): return (A,B)
    hits=[]
    for e1,e2 in edges:
        ok,P,t,_ = seg_intersection(A,B,e1,e2)
        if ok and P is not None and 0.0<=t<=1.0: hits.append((t,P))
    if _inside(A) and not _inside(B):
        if hits: hits.sort(key=lambda x:x[0]); return (A, hits[0][1])
        return None
    if not _inside(A) and _inside(B):
        if hits: hits.sort(key=lambda x:x[0]); return (hits[-1][1], B)
        return None
    if len(hits)>=2:
        hits.sort(key=lambda x:x[0]); return (hits[0][1], hits[-1][1])
    return None
