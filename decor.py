
# decor.py — centralized, lightweight rendering for grass (bushes) and rocks
# These are INSTANCED sprites (no per-frame geometry) with stable world-space positions.

import math
import pygame as _pg

# ---------------- Internal caches ----------------
_CONN_KEY = None
_CONN_POLYS = None
_WATER_KEY = None
_WATER_BUSH_POS = None
_WATER_ROCK_POS = None
_WATER_SPRITES = {}   # ("bush", lod) or ("rock", lod) -> (surface, half)

_SEA_KEY = None
_SEA_ROCK_POS = None
_SEA_SPRITES = {}

# ---------------- Utility ----------------
def _hash_poly(poly, sx=997, sy=1013):
    # fast stable hash for a large polyline/polygon by sampling
    m = 2166136261
    if not poly: return m
    step = max(1, len(poly)//128)
    for x,y in poly[::step]:
        m ^= int(x * sx) & 0xffffffff; m = (m * 16777619) & 0xffffffff
        m ^= int(y * sy) & 0xffffffff; m = (m * 16777619) & 0xffffffff
    return m

def _poly_len(p):
    L = 0.0
    for i in range(1, len(p)):
        ax,ay=p[i-1]; bx,by=p[i]
        L += math.hypot(bx-ax, by-ay)
    return L

def _point_at(p, dist):
    acc=0.0
    for i in range(1,len(p)):
        ax,ay=p[i-1]; bx,by=p[i]
        seg=math.hypot(bx-ax,by-ay)
        if acc+seg >= dist:
            t=(dist-acc)/(seg or 1.0)
            x=ax+(bx-ax)*t; y=ay+(by-ay)*t
            tx,ty=(bx-ax,by-ay); L=max(1e-9, math.hypot(tx,ty))
            return (x,y),(tx/L,ty/L)
        acc+=seg
    ax,ay=p[-2]; bx,by=p[-1]
    tx,ty=(bx-ax,by-ay); L=max(1e-9, math.hypot(tx,ty))
    return p[-1],(tx/L,ty/L)

def _cell_rng(ix,iy,seed):
    h = ((ix*2246822519) ^ (iy*3266489917) ^ seed) & 0xffffffff
    r1 = (h & 0xffff)/65536.0
    r2 = ((h>>16) & 0xffff)/65536.0
    r3 = ((h>>7)  & 0xffff)/65536.0
    return r1,r2,r3

# ---------------- Bushes (water) ----------------
def _build_water_bush_positions(poly):
    n=len(poly); k=n//2
    left  = poly[:k]
    right = list(reversed(poly[k:]))
    rkey  = _hash_poly(poly)
    def scatter(edge, sign, seed):
        total=_poly_len(edge)
        BASE=64.0; JIT=22.0; OFF_BASE=6.0; OFF_VAR=10.0
        out=[]; d=0.0
        while d<total:
            P,T=_point_at(edge,d); nx,ny=(T[1],-T[0]); nx*=sign; ny*=sign
            ix=int(P[0]//BASE); iy=int(P[1]//BASE)
            r1,r2,r3=_cell_rng(ix,iy,rkey^seed)
            along=(r1-0.5)*JIT; off=OFF_BASE+OFF_VAR*r2
            wx=P[0]+T[0]*along+nx*off; wy=P[1]+T[1]*along+ny*off
            out.append((wx,wy,int(r3*3)))  # variant 0..2
            d += BASE*(0.85+0.5*r2)
        return out
    return scatter(left,-1,0x1234ABCD)+scatter(right,+1,0xBEEF00D5)

def _get_bush_sprite(cam_zoom, variants=3):
    lod_edges=[0.6,1.0,1.6]
    lod=0
    for i,e in enumerate(lod_edges):
        if cam_zoom>e: lod=i+1
    key=("bush",lod)
    if key in _WATER_SPRITES: return _WATER_SPRITES[key]
    lod_radii=[3,4,5,7]; R=lod_radii[lod]; pad=2; size=R*2+pad*2
    shapes=[
        [(0.00,-1.00),(0.85,-0.40),(0.70,0.25),(0.00,1.00),(-0.75,0.35),(-0.90,-0.15)],
        [(0.10,-1.00),(0.90,-0.20),(0.60,0.40),(0.00,0.95),(-0.65,0.45),(-0.95,0.00),(-0.30,-0.50)],
        [(-0.10,-1.00),(0.80,-0.55),(0.95,0.10),(0.30,0.95),(-0.50,0.70),(-0.95,0.05),(-0.60,-0.45)]
    ]
    sheet=[]
    for shape in shapes[:variants]:
        s=_pg.Surface((size,size)); s.fill((255,0,255)); s.set_colorkey((255,0,255))
        cx=cy=size//2
        pts=[(int(cx+ux*R),int(cy+uy*R)) for (ux,uy) in shape]
        _pg.draw.polygon(s,(84,176,96),pts)
        sheet.append(s)
    _WATER_SPRITES[key]=(sheet,(size//2,size//2))
    return _WATER_SPRITES[key]

# ---------------- Rocks (water & sea) ----------------
def _build_rock_positions(poly, base=110.0, jit=36.0, off_base=10.0, off_var=16.0):
    n=len(poly); k=n//2
    left  = poly[:k]
    right = list(reversed(poly[k:]))
    rkey  = _hash_poly(poly, sx=911, sy=977)
    def scatter(edge, sign, seed):
        total=_poly_len(edge)
        out=[]; d=0.0
        while d<total:
            P,T=_point_at(edge,d); nx,ny=(T[1],-T[0]); nx*=sign; ny*=sign
            ix=int(P[0]//base); iy=int(P[1]//base)
            r1,r2,r3=_cell_rng(ix,iy,rkey^seed)
            along=(r1-0.5)*jit; off=off_base+off_var*r2
            wx=P[0]+T[0]*along+nx*off; wy=P[1]+T[1]*along+ny*off
            out.append((wx,wy,int(r3*3)))
            d += base*(0.9+0.6*r2)
        return out
    return scatter(left,-1,0xCAFEBABE)+scatter(right,+1,0x8BADF00D)

def _get_rock_sprite(cam_zoom, variants=3):
    lod_edges=[0.6,1.0,1.6]
    lod=0
    for i,e in enumerate(lod_edges):
        if cam_zoom>e: lod=i+1
    key=("rock",lod)
    if key in _SEA_SPRITES: return _SEA_SPRITES[key]
    lod_r=[4,6,8,10]; R=lod_r[lod]; pad=2; size=R*2+pad*2
    shapes=[
        [(-0.85,-0.40),(-0.20,-0.95),(0.55,-0.60),(0.95,0.05),(0.40,0.85),(-0.35,0.70),(-0.95,0.10)],
        [(-0.70,-0.60),(-0.10,-0.95),(0.70,-0.35),(0.95,0.20),(0.30,0.95),(-0.60,0.55),(-0.95,-0.05)],
        [(-0.90,-0.20),(-0.35,-0.85),(0.40,-0.90),(0.95,-0.10),(0.80,0.55),(0.00,0.95),(-0.75,0.60)]
    ]
    sheet=[]
    rock_color=(142,142,150)
    for shape in shapes[:variants]:
        s=_pg.Surface((size,size)); s.fill((255,0,255)); s.set_colorkey((255,0,255))
        cx=cy=size//2
        pts=[(int(cx+ux*R),int(cy+uy*R)) for (ux,uy) in shape]
        _pg.draw.polygon(s, rock_color, pts)
        sheet.append(s)
    _SEA_SPRITES[key]=(sheet,(size//2,size//2))
    return _SEA_SPRITES[key]

# ---------------- Public API ----------------
def reset_water_decor():
    global _WATER_KEY, _WATER_BUSH_POS, _WATER_ROCK_POS
    _WATER_KEY=None; _WATER_BUSH_POS=None; _WATER_ROCK_POS=None; _CONN_KEY=None; _CONN_POLYS=None

def reset_sea_decor():
    global _SEA_KEY, _SEA_ROCK_POS
    _SEA_KEY=None; _SEA_ROCK_POS=None

def draw_water_decor(screen, world_to_screen, cam_zoom, water):
    """Draw bushes + rocks for RIVER layer, centralized & stable."""
    if not water or not getattr(water,'river_poly',None): return
    poly = list(water.river_poly)
    if len(poly)<8: return
    global _WATER_KEY, _WATER_BUSH_POS, _WATER_ROCK_POS
    key = _hash_poly(poly)
    if key != _WATER_KEY or _WATER_BUSH_POS is None or _WATER_ROCK_POS is None:
        _WATER_KEY = key
        _WATER_BUSH_POS = _build_water_bush_positions(poly)
        _WATER_ROCK_POS = _build_rock_positions(poly)
    # Sprites (LOD)
    bush_sheet, (bhx,bhy) = _get_bush_sprite(cam_zoom)
    rock_sheet, (rhx,rhy) = _get_rock_sprite(cam_zoom)
    W=screen.get_width(); H=screen.get_height()

    # Draw bushes
    for (wx,wy,v) in _WATER_BUSH_POS:
        sx,sy = world_to_screen((wx,wy)); sx=int(sx); sy=int(sy)
        if -bhx<=sx<=W+bhx and -bhy<=sy<=H+bhy:
            v = 0 if v<0 else (2 if v>2 else v)
            screen.blit(bush_sheet[v], (sx-bhx, sy-bhy))

    # Draw rocks
    for (wx,wy,v) in _WATER_ROCK_POS:
        sx,sy = world_to_screen((wx,wy)); sx=int(sx); sy=int(sy)
        if -rhx<=sx<=W+rhx and -rhy<=sy<=H+rhy:
            v = 0 if v<0 else (2 if v>2 else v)
            screen.blit(rock_sheet[v], (sx-rhx, sy-rhy))

def draw_sea_decor(screen, world_to_screen, cam_zoom, sea):
    """Optional: rocks along sea shoreline (sparser)."""
    if not sea or not getattr(sea,'coast_poly',None): return
    poly = list(sea.coast_poly)
    if len(poly)<4: return
    global _SEA_KEY, _SEA_ROCK_POS
    key=_hash_poly(poly, sx=733, sy=881)
    if key!=_SEA_KEY or _SEA_ROCK_POS is None:
        _SEA_KEY=key
        _SEA_ROCK_POS=_build_rock_positions(poly, base=150.0, jit=42.0, off_base=12.0, off_var=20.0)
    rock_sheet, (rhx,rhy) = _get_rock_sprite(cam_zoom)
    W=screen.get_width(); H=screen.get_height()
    for (wx,wy,v) in _SEA_ROCK_POS:
        sx,sy = world_to_screen((wx,wy)); sx=int(sx); sy=int(sy)
        if -rhx<=sx<=W+rhx and -rhy<=sy<=H+rhy:
            v = 0 if v<0 else (2 if v>2 else v)
            screen.blit(rock_sheet[v], (sx-rhx, sy-rhy))



def draw_bridge_connectors(screen, world_to_screen, cam_zoom, roads, water, bridges_mod):
    """Visual-only curved connectors from bridge ends to nearest road segment."""
    import pygame as _pg
    from math import hypot
    global _CONN_KEY, _CONN_POLYS
    # Bridges geometry (list of (A,B,level))
    bridges = getattr(bridges_mod, '_BRIDGE_GEOMS', [])
    if not bridges: 
        _CONN_POLYS = None; _CONN_KEY = None
        return
    # Build a cheap key from bridges + rough road count + water hash (so cache busts on new river)
    bkey=len(bridges)
    for (A,B,lvl) in bridges[:16]:
        bkey ^= (int(A[0])<<1) ^ (int(A[1])<<2) ^ (int(B[0])<<3) ^ (int(B[1])<<4) ^ (lvl<<5)
    rkey=len(getattr(roads,'segments',[]))
    wkey=0
    if water and getattr(water,'river_poly',None):
        step=max(1, len(water.river_poly)//64)
        for x,y in water.river_poly[::step]:
            wkey ^= (int(x)<<1) ^ (int(y)<<2)
    cache_key=(bkey,rkey,wkey)
    if _CONN_KEY != cache_key:
        _CONN_POLYS = []
        def nearest_point_on_seg(P,s):
            ax,ay=s.a; bx,by=s.b; vx,vy=bx-ax,by-ay; L2=vx*vx+vy*vy or 1.0
            t=max(0.0,min(1.0,((P[0]-ax)*vx+(P[1]-ay)*vy)/L2)); Q=(ax+vx*t, ay+vy*t)
            return Q, hypot(Q[0]-P[0],Q[1]-P[1])
        def nearest(P):
            best=(None,None,1e9)
            for s in getattr(roads,'segments',[]):
                Q,d=nearest_point_on_seg(P,s)
                if d<best[2]: best=(Q,s,d)
            return best
        for (A,B,lvl) in bridges:
            for P in (A,B):
                Q,seg,dist=nearest(P)
                if Q is None or dist>280.0: 
                    continue
                dx,dy=Q[0]-P[0], Q[1]-P[1]; L=max(1.0, hypot(dx,dy)); ux,uy=dx/L,dy/L
                c1=(P[0]+ux*min(80.0,L*0.35), P[1]+uy*min(80.0,L*0.35))
                c2=(Q[0]-ux*min(80.0,L*0.35), Q[1]-uy*min(80.0,L*0.35))
                steps=max(20,int(L/8.0)); poly=[]
                for i in range(steps+1):
                    t=i/float(steps); u=1.0-t
                    x=u*u*u*P[0]+3*u*u*t*c1[0]+3*u*t*t*c2[0]+t*t*t*Q[0]
                    y=u*u*u*P[1]+3*u*u*t*c1[1]+3*u*t*t*c2[1]+t*t*t*Q[1]
                    poly.append((x,y))
                _CONN_POLYS.append(poly)
        _CONN_KEY = cache_key
    col=(170,174,186); wpx=max(2,int(6*cam_zoom*1.6))
    if _CONN_POLYS:
        for poly in _CONN_POLYS:
            pts=[world_to_screen(p) for p in poly]
            if len(pts)>=2: _pg.draw.lines(screen,col,False,pts,wpx)
