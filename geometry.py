
import math

def dot(a,b): return a[0]*b[0] + a[1]*b[1]
def sub(a,b): return (a[0]-b[0], a[1]-b[1])
def add(a,b): return (a[0]+b[0], a[1]+b[1])
def mul(v,s): return (v[0]*s, v[1]*s)

def seg_intersection(a1, a2, b1, b2):
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = a1,a2,b1,b2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-9:
        return (False, None, None, None)
    t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4)) / den
    u = ((x1-x3)*(y1-y2)-(y1-y3)*(x1-x2)) / den
    if -1e-6 <= t <= 1+1e-6 and -1e-6 <= u <= 1+1e-6:
        Px = x1 + t*(x2-x1); Py = y1 + t*(y2-y1)
        return (True, (Px, Py), t, u)
    return (False, None, None, None)

def point_seg_dist(p, a, b):
    ap = sub(p, a); ab = sub(b, a); ab2 = dot(ab, ab)
    if ab2 == 0: return (math.hypot(*ap), a, 0.0)
    t = max(0.0, min(1.0, dot(ap,ab)/ab2))
    proj = add(a, mul(ab, t))
    return (math.hypot(p[0]-proj[0], p[1]-proj[1]), proj, t)

def seg_aabb(a,b):
    minx, miny = min(a[0],b[0]), min(a[1],b[1])
    maxx, maxy = max(a[0],b[0]), max(a[1],b[1])
    return (minx, miny, maxx-minx, maxy-miny)

def norm(v):
    l = math.hypot(v[0], v[1])
    return (0.0, 0.0) if l==0 else (v[0]/l, v[1]/l)

def rect_from_axes(center, tvec, nvec, w_along_t, h_along_n):
    cx, cy = center; tx, ty = tvec; nx, ny = nvec
    hw = w_along_t * 0.5; hh = h_along_n * 0.5
    return [
        (cx + tx*hw + nx*hh, cy + ty*hw + ny*hh),
        (cx - tx*hw + nx*hh, cy - ty*hw + ny*hh),
        (cx - tx*hw - nx*hh, cy - ty*hw - ny*hh),
        (cx + tx*hw - nx*hh, cy + ty*hw - ny*hh),
    ]

def poly_aabb(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))

def road_tube_poly(a,b,width):
    ux,uy = norm((b[0]-a[0], b[1]-a[1]))
    nx,ny = (uy, -ux)
    hh = width*0.5
    return [
        (a[0]+nx*hh, a[1]+ny*hh),
        (b[0]+nx*hh, b[1]+ny*hh),
        (b[0]-nx*hh, b[1]-ny*hh),
        (a[0]-nx*hh, a[1]-ny*hh),
    ]

def _proj_range(poly, axis):
    ax,ay = axis
    mn=mx = poly[0][0]*ax + poly[0][1]*ay
    for x,y in poly[1:]:
        v = x*ax + y*ay
        if v<mn: mn=v
        if v>mx: mx=v
    return mn,mx

def polys_intersect(poly1, poly2):
    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
            edge = (x2-x1, y2-y1)
            axis = (-edge[1], edge[0])
            l = (axis[0]**2 + axis[1]**2)**0.5
            if l == 0: continue
            axis = (axis[0]/l, axis[1]/l)
            a1,a2 = _proj_range(poly1, axis); b1,b2 = _proj_range(poly2, axis)
            if a2 < b1 or b2 < a1: return False
    return True

def point_in_poly(pt, poly):
    x,y = pt; inside = False; n=len(poly)
    for i in range(n):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
        if (y1>y)!=(y2>y):
            xinters = (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1
            if x < xinters: inside = not inside
    return inside

def segment_intersects_polygon(a, b, poly):
    if point_in_poly(a, poly) or point_in_poly(b, poly): return True
    n = len(poly)
    for i in range(n):
        c = poly[i]; d = poly[(i+1)%n]
        hit,_,_,_ = seg_intersection(a,b,c,d)
        if hit: return True
    return False
