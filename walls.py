"""
walls.py — v7.0
----------------
- OFF by default. No auto-generation. Walls change ONLY when you press **W** (or call generate_walls).
- **Stray line killer++**:
    * Drops fragments within CLIP_MARGIN_WORLD of the wall boundary.
    * Drops fragments shorter than MIN_SEG_LEN_WORLD.
    * **Gate alignment**: a segment is allowed through a gate channel only if it is
      directionally aligned with the gate (angle ≤ MAX_GATE_ANGLE_DEG).
- Still single‑lane channels for gates.

Hotkeys expected in your app:
    W        → enable & (re)generate
    Ctrl+W   → clear & disable
"""

from typing import List, Tuple, Optional, Dict, Any
import math, random

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

# ---------------------- Tunables ----------------------
CLIP_MARGIN_WORLD: float = 8.0      # tighten to nuke edge-glued stubs
MIN_SEG_LEN_WORLD: float = 12.0     # drop tiny clipped pieces
MAX_GATE_ANGLE_DEG: float = 22.5    # channel alignment tolerance

# ---------------------- Module state ----------------------
_ENABLED: bool = False

_POLY: List[Point] = []             # wall polygon
_GATE_POS: Dict[int, float] = {}    # edge index -> gate param u in [0,1]
_GATE_CHANNELS: List[Dict] = []     # [{'center':(x,y), 'dir':(dx,dy), 'half_len':L, 'half_width':W}]

_GATE_WIDTH_WORLD: float = 1.0
_CORRIDOR_LEN_WORLD: float = 1.0

_CLASS_MAP = {
    "highways": ["highways", "primary", "motorway", "motorways", "express", "expressways", "trunk", "trunks", "arterial", "arterials", "major", "majors"],
    "roads": ["roads", "secondary", "mains", "main", "primaries"],
    "tertiaries": ["tertiaries", "tertiary", "residential", "collector", "collectors"],
    "alleys": ["alleys", "alley", "service", "minor", "lane", "lanes"],
}

def enable(): 
    global _ENABLED; _ENABLED = True
def disable():
    global _ENABLED; _ENABLED = False
def toggle():
    global _ENABLED; _ENABLED = not _ENABLED
def is_enabled() -> bool: return _ENABLED

def configure_class_mapping(mapping: Dict[str, List[str]]):
    for k,v in mapping.items():
        if k in _CLASS_MAP and isinstance(v, (list,tuple)):
            s = set(n.lower() for n in _CLASS_MAP[k])
            for name in v:
                try: s.add(str(name).lower())
                except Exception: pass
            _CLASS_MAP[k] = sorted(s)

# ---------------------- Utils ----------------------
def _clamp(v,a,b): return a if v<a else b if v>b else v
def _seg_length(a,b): dx=a[0]-b[0]; dy=a[1]-b[1]; return (dx*dx+dy*dy)**0.5
def _normalize(vx,vy):
    L=(vx*vx+vy*vy)**0.5 or 1.0
    return (vx/L, vy/L, L)

def _seg_intersect(a,b,c,d):
    ax,ay=a; bx,by=b; cx,cy=c; dx,dy=d
    r=(bx-ax, by-ay); s=(dx-cx, dy-cy)
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < 1e-9: return None
    t = ((cx-ax)*s[1] - (cy-ay)*s[0]) / rxs
    u = ((cx-ax)*r[1] - (cy-ay)*r[0]) / rxs
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        P = (ax + r[0]*t, ay + r[1]*t)
        return (t,u,P)
    return None

def _point_segment_distance(px,py, ax,ay, bx,by):
    vx,vy = bx-ax, by-ay
    L2 = vx*vx + vy*vy
    if L2 <= 1e-12: dx,dy = px-ax, py-ay; return (dx*dx+dy*dy)**0.5
    t = ((px-ax)*vx + (py-ay)*vy)/L2; t = 0 if t<0 else 1 if t>1 else t
    cx,cy = ax + vx*t, ay + vy*t; dx,dy = px-cx, py-cy
    return (dx*dx+dy*dy)**0.5

def _dist_to_polygon_edges(p, poly):
    x,y = p; best = 1e18; n=len(poly)
    for i in range(n):
        ax,ay = poly[i]; bx,by = poly[(i+1)%n]
        d = _point_segment_distance(x,y, ax,ay, bx,by)
        if d < best: best = d
    return best

def _point_in_poly(pt, poly):
    x,y = pt; inside=False; n=len(poly)
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        if ((y1>y) != (y2>y)):
            denom = (y2 - y1) if (y2!=y1) else 1e-12
            t = (y - y1) / denom
            xint = x1 + (x2 - x1) * t
            if x < xint: inside = not inside
    return inside

# ---------------------- Data extraction ----------------------
def _as_point(obj):
    try:
        if isinstance(obj, (list,tuple)) and len(obj)==2:
            return (float(obj[0]), float(obj[1]))
        if hasattr(obj, "__iter__"):
            it=list(obj)
            if len(it)==2: return (float(it[0]), float(it[1]))
    except Exception: return None
    return None

def _polyline_from_any(item):
    try:
        if isinstance(item, (list,tuple)) and item:
            p0 = _as_point(item[0])
            if p0 is not None:
                pts=[_as_point(p) for p in item]
                pts=[p for p in pts if p is not None]
                if len(pts)>=2: return pts
            else:
                pts=[]
                for seg in item:
                    try:
                        a = (float(seg.a[0]), float(seg.a[1]))
                        b = (float(seg.b[0]), float(seg.b[1]))
                        pts.append(a); pts.append(b)
                    except Exception:
                        try:
                            a = (float(seg['a'][0]), float(seg['a'][1]))
                            b = (float(seg['b'][0]), float(seg['b'][1]))
                            pts.append(a); pts.append(b)
                        except Exception: pass
                if len(pts)>=2: return pts
    except Exception: pass
    for attr in ("points","pts"):
        if hasattr(item, attr):
            try:
                raw=getattr(item, attr)
                pts=[_as_point(p) for p in raw]
                pts=[p for p in pts if p is not None]
                if len(pts)>=2: return pts
            except Exception: pass
    try:
        a = (float(item.a[0]), float(item.a[1])); b = (float(item.b[0]), float(item.b[1]))
        return [a,b]
    except Exception: pass
    try:
        a = (float(item['a'][0]), float(item['a'][1])); b = (float(item['b'][0]), float(item['b'][1]))
        return [a,b]
    except Exception: pass
    return None

def _polyline_length(pts):
    L=0.0
    for i in range(len(pts)-1): L += _seg_length(pts[i], pts[i+1])
    return L

def _lower_keys(obj):
    if isinstance(obj, dict): return { (str(k).lower()): v for k,v in obj.items() }
    try: d = obj.__dict__; return { (str(k).lower()): v for k,v in d.items() }
    except Exception: return {}

def _try_collect_from_container(container, synonyms):
    out=[]; keys = _lower_keys(container)
    for name in synonyms:
        key = str(name).lower()
        if key in keys:
            try: items = list(keys[key])
            except Exception: items = [keys[key]]
            for it in items:
                pl = _polyline_from_any(it)
                if pl and len(pl)>=2: out.append(pl)
    return out

def _collect_polylines_by_class(roads, debug=False):
    sources = [roads]
    for attr in ("network","layers","data","graph","graphs","classes","classed","groups"):
        if hasattr(roads, attr):
            try: sources.append(getattr(roads, attr))
            except Exception: pass

    result = {"highways":[], "roads":[], "tertiaries":[], "alleys":[]}
    found_any=False
    for cat, synonyms in _CLASS_MAP.items():
        polylines=[]
        polylines += _try_collect_from_container(roads, synonyms)
        for src in sources[1:]:
            polylines += _try_collect_from_container(src, synonyms)
        if polylines:
            result[cat] = polylines; found_any = True

    
    if not found_any:
        # Derive classes directly from a RoadSystem-like object by grouping
        # segments using its (possibly remapped) levels.
        segs = []
        try:
            s = getattr(roads, "segments", None)
            if s is not None:
                segs.extend(list(s))
        except Exception:
            pass
        cat_map = {0:"highways", 1:"roads", 2:"tertiaries", 3:"alleys"}
        for s in segs:
            try:
                lvl = int(getattr(s, "level", 1))
            except Exception:
                lvl = 1
            # If the road system provides a level mapping, use it.
            try:
                eff = int(roads.map_lvl(lvl))
            except Exception:
                eff = lvl
            cat = cat_map.get(max(0, min(3, eff)), "roads")
            pl = _polyline_from_any(s)
            if pl and len(pl) >= 2:
                result[cat].append(pl)
                found_any = True


    if debug:
        try:
            print("[walls v7.0] Detected road classes:")
            for k,v in result.items():
                cnt = len(v); total_len = sum(_polyline_length(pl) for pl in v) if cnt else 0.0
                print(f"  - {k}: {cnt} polylines, total length ~ {total_len:.1f}")
            if not found_any: print("  (No classes/segments detected)")
        except Exception: pass
    return result

# ---------------------- Generation ----------------------
def reset_walls():
    global _POLY, _GATE_POS, _GATE_CHANNELS, _GATE_WIDTH_WORLD, _CORRIDOR_LEN_WORLD
    _POLY = []; _GATE_POS = {}; _GATE_CHANNELS = []
    _GATE_WIDTH_WORLD = 1.0; _CORRIDOR_LEN_WORLD = 1.0

def generate_walls(roads, water=None, n_gates:int=3, n_verts:Optional[int]=None, margin:Optional[float]=None,
                   noise:float=0.18, angle_jitter:float=0.25, random_seed:Optional[int]=None,
                   gate_width_frac:float=0.05, corridor_len_frac:float=0.60,
                   use_classes:bool=True, debug=False):
    global _POLY, _GATE_POS, _GATE_CHANNELS, _GATE_WIDTH_WORLD, _CORRIDOR_LEN_WORLD

    try:
        W = float(getattr(roads, "MAP_WIDTH")); H = float(getattr(roads, "HEIGHT"))
    except Exception:
        W, H = 2048.0, 2048.0

    if random_seed is not None:
        try: random.seed(int(random_seed))
        except Exception: pass

    if margin is None: margin = 0.08 * min(W, H)
    K = 12 if n_verts is None else int(max(10, min(15, n_verts)))

    # endpoints influence center
    endpoints=[]
    classes = _collect_polylines_by_class(roads, debug=debug) if use_classes else {"roads":[]}
    for arr in classes.values():
        for pl in arr:
            endpoints.append(pl[0]); endpoints.append(pl[-1])
    if not endpoints: cx, cy = 0.5*W, 0.5*H
    else:
        cx = _clamp(sum(p[0] for p in endpoints)/len(endpoints), 0.2*W, 0.8*W)
        cy = _clamp(sum(p[1] for p in endpoints)/len(endpoints), 0.2*H, 0.8*H)

    # sectors + jitter for organic hull
    weights=[ -math.log(max(1e-9, 1.0-random.random())) for _ in range(K) ]
    S=sum(weights) or 1.0
    parts=[ (w/S)*2.0*math.pi for w in weights ]
    a0=random.uniform(0, 2.0*math.pi)
    base=[]; sizes=[]
    a=a0
    for p in parts: base.append(a); sizes.append(p); a+=p
    angles=[]
    for ang,siz in zip(base,sizes):
        j=(random.random()*2-1)*angle_jitter*min(siz, 2.0*math.pi/K)
        angles.append(ang+j)

    # bounds per ray (stay within map)
    def _ray_rect_limit(cx, cy, dx, dy, W, H):
        t_max=float("inf")
        if abs(dx)>1e-9: t_max=min(t_max, (W-cx)/dx if dx>0 else (0.0-cx)/dx)
        else: 
            if cx<0.0 or cx>W: return 0.0
        if abs(dy)>1e-9: t_max=min(t_max, (H-cy)/dy if dy>0 else (0.0-cy)/dy)
        else:
            if cy<0.0 or cy>H: return 0.0
        return 0.0 if t_max<0 else t_max

    bound_r=[]
    for th in angles:
        dx,dy=math.cos(th), math.sin(th)
        tmax=_ray_rect_limit(cx,cy,dx,dy,W,H)
        bound_r.append(max(0.0, tmax-2.0))

    # adaptive radius per sector
    base_r=[]
    for i,th in enumerate(angles):
        window = sizes[i]*0.9
        def _in_win(ang):
            d=(ang-th+math.pi)%(2*math.pi)-math.pi
            return abs(d) <= 0.5*window
        rmax=0.0
        for (px,py) in endpoints:
            ang=math.atan2(py-cy, px-cx)
            if _in_win(ang):
                r=((px-cx)**2+(py-cy)**2)**0.5
                if r>rmax: rmax=r
        r=min(bound_r[i]-margin, rmax+0.7*margin)
        r=max(0.22*min(W,H), r)
        r=_clamp(r, 0.0, bound_r[i]-max(2.0,0.5*margin))
        base_r.append(r)

    # smooth + jitter
    def _smooth_loop(vals,k=1):
        n=len(vals); out=[0.0]*n
        for i in range(n):
            s=vals[i]; c=1.0
            for j in range(1,k+1):
                s+=vals[(i-j)%n]+vals[(i+j)%n]; c+=2.0
            out[i]=s/c
        return out
    r_sm=_smooth_loop(base_r,k=1)
    r_final=[]
    for i,r in enumerate(r_sm):
        out_room=max(0.0,(bound_r[i]-2.0)-r); in_room=max(0.0, r-0.15*min(W,H))
        amp=noise*margin
        d=(random.random()*2-1)*amp
        if d>0: d=min(d, 0.8*out_room)
        else:   d=-min(-d, 0.8*in_room)
        r_final.append(r+d)

    # polygon
    poly=[]
    for i,th in enumerate(angles):
        dx,dy=math.cos(th), math.sin(th)
        x=cx+r_final[i]*dx; y=cy+r_final[i]*dy
        x=_clamp(x,0.0,W); y=_clamp(y,0.0,H)
        poly.append((x,y))
    _POLY = poly

    # choose gates where big roads hit the hull; store channel directions
    n=len(_POLY); edges=[ (_POLY[i], _POLY[(i+1)%n]) for i in range(n) ]
    weight = {"highways":4.0, "roads":2.5, "tertiaries":1.5, "alleys":1.0}
    inters=[]
    for cat, arr in classes.items():
        wcat = weight.get(cat, 1.0)
        for pts in arr:
            Lp = _polyline_length(pts); score = Lp * wcat
            for i in range(len(pts)-1):
                a=pts[i]; b=pts[i+1]
                for ei,(p,q) in enumerate(edges):
                    hit=_seg_intersect(a,b,p,q)
                    if hit is not None:
                        t,u,P = hit
                        inters.append((score, ei, float(u), a, b))

    inters.sort(key=lambda x: x[0], reverse=True)
    chosen={}; chosen_dir={}; need=max(1,int(n_gates))
    for score,ei,u,a,b in inters:
        if len(chosen) >= need: break
        if ei in chosen: continue
        if any((abs(ei - ej) % n) <= 1 for ej in chosen.keys()): continue
        chosen[ei]=u; chosen_dir[ei]=(a,b)
    idx=0
    while len(chosen) < need and idx < len(inters):
        ei,u,a,b = inters[idx][1], inters[idx][2], inters[idx][3], inters[idx][4]
        chosen.setdefault(ei,u); chosen_dir.setdefault(ei,(a,b)); idx+=1
    if not chosen:
        step = max(1, n // need)
        for k in range(need):
            ei=(k*step) % n; chosen[ei]=0.5; chosen_dir[ei]=((0,0),(1,0))

    # build channels
    _GATE_POS = dict(chosen); _GATE_CHANNELS = []
    _GATE_WIDTH_WORLD = gate_width_frac * min(W,H)
    _CORRIDOR_LEN_WORLD = corridor_len_frac * max(W,H)
    half_len = 0.5 * _CORRIDOR_LEN_WORLD
    for ei,u in _GATE_POS.items():
        (ax,ay),(bx,by) = edges[ei]; vx,vy,L = _normalize(bx-ax, by-ay)
        mx,my = (ax + vx*u*L, ay + vy*u*L)
        ra,rb = chosen_dir[ei]; rx,ry,_ = _normalize(rb[0]-ra[0], rb[1]-ra[1])
        half_w = max(8.0, 0.25 * 0.5 * _GATE_WIDTH_WORLD)   # single-lane
        _GATE_CHANNELS.append({"center": (mx, my), "dir": (rx, ry), "half_len": half_len, "half_width": half_w})

# ---------------------- Membership tests ----------------------
def _point_in_any_channel_aligned(mid:Point, seg_dir:Point) -> bool:
    if not _GATE_CHANNELS: return False
    mx,my = mid; dx_seg, dy_seg = seg_dir
    # normalise seg dir
    dsx,dsy,_ = _normalize(dx_seg, dy_seg)
    # angle threshold
    cos_thr = math.cos(math.radians(MAX_GATE_ANGLE_DEG))
    for ch in _GATE_CHANNELS:
        cx,cy = ch["center"]; dx,dy = ch["dir"]
        # channel bbox membership (capsule)
        vx,vy = (mx-cx, my-cy)
        s = vx*dx + vy*dy
        if -ch["half_len"] <= s <= ch["half_len"]:
            px = vx - s*dx; py = vy - s*dy
            if (px*px + py*py) <= (ch["half_width"]*ch["half_width"]):
                # direction alignment
                dot = abs(dsx*dx + dsy*dy)
                if dot >= cos_thr:
                    return True
    return False

def _segment_polygon_intersections(a,b, poly):
    ts=[]; n=len(poly)
    for i in range(n):
        p=poly[i]; q=poly[(i+1)%n]
        inter=_seg_intersect(a,b,p,q)
        if inter is not None:
            t=inter[0]
            if 0.0 <= t <= 1.0: ts.append(float(t))
    ts=sorted(set(round(t,6) for t in ts)); return ts

# ---------------------- Clipping ----------------------
def clip_segment(a:Point, b:Point) -> List[Segment]:
    if not _POLY:
        return [(a,b)]
    tvals=[0.0, 1.0]; tvals += _segment_polygon_intersections(a,b,_POLY)
    tvals = sorted(set(max(0.0,min(1.0,t)) for t in tvals))
    out=[]
    for i in range(len(tvals)-1):
        t0,t1=tvals[i], tvals[i+1]
        if t1 - t0 <= 1e-6: continue
        tm = 0.5*(t0+t1)
        mx = a[0] + (b[0]-a[0])*tm; my = a[1] + (b[1]-a[1])*tm
        seg_dir = (b[0]-a[0], b[1]-a[1])

        inside_poly = _point_in_poly((mx,my), _POLY)
        near_edge = (_dist_to_polygon_edges((mx,my), _POLY) < CLIP_MARGIN_WORLD)
        inside_channel = _point_in_any_channel_aligned((mx,my), seg_dir)

        allow = inside_channel or (inside_poly and not near_edge)
        if allow:
            p0=(a[0] + (b[0]-a[0])*t0, a[1] + (b[1]-a[1])*t0)
            p1=(a[0] + (b[0]-a[0])*t1, a[1] + (b[1]-a[1])*t1)
            if _seg_length(p0,p1) >= MIN_SEG_LEN_WORLD:
                out.append((p0,p1))
    return out

def clip_polyline(points: List[Point]) -> List[List[Point]]:
    if not points or len(points)<2 or not _POLY: return [points]
    pieces=[]
    for i in range(len(points)-1):
        a=points[i]; b=points[i+1]
        for (p,q) in clip_segment(a,b): pieces.append([p,q])
    if not pieces: return []
    # stitch adjacent
    stitched=[]; cur=[pieces[0][0], pieces[0][1]]
    for i in range(1,len(pieces)):
        prev_end=cur[-1]; s,e = pieces[i]
        if abs(prev_end[0]-s[0])<1e-6 and abs(prev_end[1]-s[1])<1e-6:
            cur.append(e)
        else:
            stitched.append(cur); cur=[s,e]
    stitched.append(cur); return stitched

# ---------------------- Drawing ----------------------
_DEFAULT_STYLES = {
    "highways": {"color": (200,200,200), "width": 8},
    "roads": {"color": (190,190,190), "width": 5},
    "tertiaries": {"color": (170,170,170), "width": 3},
    "alleys": {"color": (150,150,150), "width": 2},
}

def draw_network_clipped(screen, world_to_screen, cam_zoom:float, roads, styles:Optional[Dict[str,Dict]]=None):
    import pygame
    if styles is None: styles = _DEFAULT_STYLES
    classes = _collect_polylines_by_class(roads, debug=False)
    if not any(classes.values()): return
    for cat, polylines in classes.items():
        if not polylines: continue
        style = styles.get(cat, _DEFAULT_STYLES.get(cat, {"color": (180,180,180), "width": 3}))
        color = style.get("color", (180,180,180)); px = max(1, int(style.get("width", 3) * cam_zoom))
        for pts in polylines:
            pieces = clip_polyline(pts) if (_ENABLED and _POLY) else [pts]
            for piece in pieces:
                for i in range(len(piece)-1):
                    a=piece[i]; b=piece[i+1]
                    pygame.draw.line(screen, color, world_to_screen(a), world_to_screen(b), px)

def patch_roads_draw_for_clipping(roads):
    # kept for compatibility, not used in your vector script path
    if not hasattr(roads, "draw"): return
    import pygame
    _orig = roads.draw
    def _wrap(screen, world_to_screen, cam_zoom, *args, **kwargs):
        layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        _orig(layer, world_to_screen, cam_zoom, *args, **kwargs)
        # polygon mask only; channels are geometric
        if _ENABLED and _POLY:
            W,H = screen.get_size()
            mask = pygame.Surface((W, H), pygame.SRCALPHA); mask.fill((0,0,0,0))
            if len(_POLY) >= 3:
                import pygame as _pg
                _pg.draw.polygon(mask, (255,255,255,255), [world_to_screen(p) for p in _POLY])
            layer.blit(mask, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(layer, (0,0))
    roads.draw = _wrap

def takeover_roads_render(roads):
    if not hasattr(roads, "draw"): return
    def _noop(*a, **k): return None
    roads.draw = _noop

def draw_overlay(screen, world_to_screen, cam_zoom:float, roads, water=None,
                 color=(60,60,60), thickness:int=16, **_ignored):
    import pygame
    if not _ENABLED or not _POLY or len(_POLY) < 6: return
    px_thickness = max(1, int(thickness * cam_zoom))
    n=len(_POLY)
    for e in range(n):
        A=_POLY[e]; B=_POLY[(e+1)%n]
        u_gate=_GATE_POS.get(e, None)
        if u_gate is not None:
            ax,ay=A; bx,by=B
            vx,vy, L = _normalize(bx-ax, by-ay)
            half = 0.5*_GATE_WIDTH_WORLD
            t0=max(0.0, u_gate - half / L); t1=min(1.0, u_gate + half / L)
            p0=(ax + vx*t0*L, ay + vy*t0*L); p1=(ax + vx*t1*L, ay + vy*t1*L)
            pygame.draw.line(screen, color, world_to_screen(A), world_to_screen(p0), px_thickness)
            pygame.draw.line(screen, color, world_to_screen(p1), world_to_screen(B), px_thickness)
        else:
            pygame.draw.line(screen, color, world_to_screen(A), world_to_screen(B), px_thickness)

def get_wall_state():
    return {
        "enabled": _ENABLED,
        "poly": list(_POLY),
        "gate_pos": dict(_GATE_POS),
        "gate_channels": [dict(center=c["center"], dir=c["dir"], half_len=c["half_len"], half_width=c["half_width"]) for c in _GATE_CHANNELS],
        "gate_visual_width": _GATE_WIDTH_WORLD,
        "clip_margin_world": CLIP_MARGIN_WORLD,
        "min_seg_len_world": MIN_SEG_LEN_WORLD,
        "max_gate_angle_deg": MAX_GATE_ANGLE_DEG,
    }
