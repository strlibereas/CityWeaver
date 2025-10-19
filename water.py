
import random, math

WATER_COLOR = (120, 180, 255)

class WaterSystem:

    
    
    def draw_overlay(self, screen, world_to_screen, cam_zoom):
        """River drawn ABOVE roads; visual banks with two offset 'roads', rocks (inside), and organic grass blobs."""
        import pygame, math, random
        if not self.river_poly and not self.sea_poly:
            return

        def to_screen(poly):
            return [world_to_screen(p) for p in poly]
        def _px(n):
            # convert screen pixels to world units for consistent on-screen sizing
            return n / max(0.001, cam_zoom)
        def _zoom_px(n):
            # pixels that increase with zoom (so shapes get larger as you zoom in)
            return (n * (0.8 + 0.6*cam_zoom)) / max(0.001, cam_zoom)

        # --- Base water fill + thick outline to mask seams
        if self.river_poly:
            river_pts = to_screen(self.river_poly)
            pygame.draw.polygon(screen, WATER_COLOR, river_pts)
            pygame.draw.polygon(screen, (80,120,180), river_pts, max(2, int(4*cam_zoom)))

            N = len(self.river_poly)
            if N >= 6 and N % 2 == 0:
                half = N // 2
                left_bank = self.river_poly[:half]
                right_bank = list(reversed(self.river_poly[half:]))

                # Sampling
                def sample_polyline(polyline, step_world=10.0):
                    out = [polyline[0]]
                    acc = 0.0
                    for i in range(1, len(polyline)):
                        ax, ay = polyline[i-1]; bx, by = polyline[i]
                        dx, dy = bx-ax, by-ay
                        d = (dx*dx + dy*dy)**0.5
                        acc += d
                        if acc >= step_world:
                            out.append((bx,by)); acc = 0.0
                    out.append(polyline[-1])
                    return out
                def _resample_by_count(polyline, count):
                    # Uniformly resample a polyline to exactly 'count' vertices (>=2).
                    count = max(2, int(count))
                    if len(polyline) <= 2:
                        return list(polyline)
                    # Build cumulative lengths
                    pts = list(polyline)
                    segs = []
                    total = 0.0
                    for i in range(1, len(pts)):
                        ax, ay = pts[i-1]; bx, by = pts[i]
                        dx, dy = bx-ax, by-ay
                        d = (dx*dx+dy*dy)**0.5
                        segs.append((d, (ax,ay), (bx,by)))
                        total += d
                    if total <= 1e-6:
                        return [pts[0], pts[-1]]
                    step = total / (count-1)
                    result = [pts[0]]
                    dist = 0.0
                    target = step
                    si = 0
                    while len(result) < count-1 and si < len(segs):
                        d,(ax,ay),(bx,by) = segs[si]
                        if dist + d >= target:
                            # interpolate
                            t = (target - dist)/d if d>0 else 0.0
                            x = ax + (bx-ax)*t
                            y = ay + (by-ay)*t
                            result.append((x,y))
                            target += step
                        else:
                            dist += d
                            si += 1
                    result.append(pts[-1])
                    return result

                step = max(10.0, 14.0 / max(0.5, cam_zoom))
                Ls = sample_polyline(left_bank, step_world=step)
                Rs = sample_polyline(right_bank, step_world=step)
                # Progressive smoothing outward
                Ls_mid = _resample_by_count(Ls, 60)
                Rs_mid = _resample_by_count(Rs, 60)
                Ls_road = _resample_by_count(Ls, 40)
                Rs_road = _resample_by_count(Rs, 40)

                rng = random.Random(424242)

                def offset_path(samples, outward_sign=+1, base_offset=12.0, jitter=2.0):
                    pts = []
                    for i in range(1, len(samples)):
                        P = samples[i-1]; Q = samples[i]
                        vx, vy = (Q[0]-P[0], Q[1]-P[1])
                        L = (vx*vx + vy*vy)**0.5 or 1.0
                        nx, ny = (vy/L, -vx/L)  # outward-ish normal
                        j = rng.uniform(-jitter, jitter)
                        off = base_offset + j
                        if i == 1:
                            pts.append((P[0] + nx*off*outward_sign, P[1] + ny*off*outward_sign))
                        pts.append((Q[0] + nx*off*outward_sign, Q[1] + ny*off*outward_sign))
                    return pts

                # Two visual "bank roads" on the river layer (to cover gaps below)
                bank_col = (164, 168, 176)
                road_w = max(6, int(8*cam_zoom))  # thicker
                edge_w = max(1, road_w//5)
                dark = (38,38,38)      # asphalt
                edge = (210,210,210)   # edge stripe

                left_path  = offset_path(Ls_road, outward_sign=-1, base_offset=12.0, jitter=0.35)
                right_path = offset_path(Rs_road, outward_sign=+1, base_offset=12.0, jitter=0.35)

                if len(left_path) >= 2:
                    pygame.draw.lines(screen, bank_col, False, to_screen(left_path), road_w)
                if len(right_path) >= 2:
                    pygame.draw.lines(screen, bank_col, False, to_screen(right_path), road_w)

                # --- Solid fill between water edge and bank roads (covers entire gap) ---
                bank_fill = (198, 214, 187)  # soft greenish bank color
                def offset_polyline(samples, outward_sign, off):
                    pts = []
                    for i in range(1, len(samples)):
                        P = samples[i-1]; Q = samples[i]
                        vx, vy = (Q[0]-P[0], Q[1]-P[1])
                        L = (vx*vx + vy*vy)**0.5 or 1.0
                        nx, ny = (vy/L, -vx/L)
                        if i == 1:
                            pts.append((P[0] + nx*off*outward_sign, P[1] + ny*off*outward_sign))
                        pts.append((Q[0] + nx*off*outward_sign, Q[1] + ny*off*outward_sign))
                    return pts

                inner_off = -1.5  # extend slightly into water edge to hide seams
                outer_off = 12.8  # a bit wider under the bank road

                # Left bank: outward_sign currently -1 (v10 flip), so corridor extends with -1
                left_inner  = offset_polyline(Ls_mid, outward_sign=-1, off=inner_off)
                left_outer  = offset_polyline(Ls_mid, outward_sign=-1, off=outer_off)
                if len(left_inner) >= 2 and len(left_outer) >= 2:
                    poly = to_screen(left_inner + list(reversed(left_outer)))
                    import pygame as _pg
                    _pg.draw.polygon(screen, bank_fill, poly)

                # Right bank: outward_sign currently +1
                right_inner = offset_polyline(Rs_mid, outward_sign=+1, off=inner_off)
                right_outer = offset_polyline(Rs_mid, outward_sign=+1, off=outer_off)
                if len(right_inner) >= 2 and len(right_outer) >= 2:
                    poly = to_screen(right_inner + list(reversed(right_outer)))
                    import pygame as _pg
                    _pg.draw.polygon(screen, bank_fill, poly)


                # --- Gap fill on CITY side (outside the bank roads) with rocks & grass ---
                from geometry import point_in_poly
                def _rock_poly(center, size, rng2):
                    import math
                    k = rng2.randint(5,8)
                    verts = []
                    ang0 = rng2.uniform(0, math.tau)
                    for j in range(k):
                        ang = ang0 + (j/k) * math.tau + rng2.uniform(-0.2, 0.2)
                        rad = size * (0.6 + rng2.uniform(0.0, 0.6))
                        px = center[0] + math.cos(ang)*rad
                        py = center[1] + math.sin(ang)*rad
                        verts.append(world_to_screen((px,py)))
                    return verts

                def _grass_blob(center, size, rng3):
                    import math
                    k = rng3.randint(6,10)
                    verts = []
                    ang0 = rng3.uniform(0, math.tau)
                    for j in range(k):
                        ang = ang0 + (j/k) * math.tau + rng3.uniform(-0.25, 0.25)
                        rad = size * (0.6 + rng3.uniform(0.0, 0.6))
                        px = center[0] + math.cos(ang)*rad
                        py = center[1] + math.sin(ang)*rad
                        verts.append(world_to_screen((px,py)))
                    return verts

                def scatter_gap_fill(samples, outward_sign, rock_density=0.80, grass_density=0.50, seed=606):
                    rng2 = random.Random(seed)
                    stepW = 7.0
                    for i in range(1, len(samples)):
                        ax,ay = samples[i-1]; bx,by = samples[i]
                        dx,dy = (bx-ax, by-ay)
                        segL = (dx*dx+dy*dy)**0.5 or 1.0
                        nx,ny = (dy/segL, -dx/segL)  # outward approx
                        # march along the segment and place items in the band between bank road and city
                        s = 0.0
                        while s < segL:
                            s += stepW
                            t = min(1.0, s/segL)
                            x = ax + dx*t; y = ay + dy*t
                            # distance band starts just outside the bank line width and extends a bit into city
                            band_min = (road_w * 0.25) + 0.3
                            band_max = band_min + 7.5
                            off = rng2.uniform(band_min, band_max)
                            lat = rng2.uniform(-4.0, 4.0)  # slight along-bank jitter
                            cx = x + nx * off * outward_sign + (-ny) * lat
                            cy = y + ny * off * outward_sign + ( nx) * lat
                            # ensure not inside water polygon
                            if point_in_poly((cx,cy), self.river_poly):
                                continue
                            # choose rock or grass
                            if False:
                                sz = _zoom_px(rng2.uniform(5.0, 13.0))
                                poly = _rock_poly((cx,cy), sz, rng2)
                                pygame.draw.polygon(screen, (112,112,112), poly)
                                pygame.draw.polygon(screen, (90,90,90), poly, 1)
                            if rng2.random() < grass_density:
                                szg = _zoom_px(rng2.uniform(6.0, 18.0))
                                poly = _grass_blob((cx,cy), szg, rng2)
                                # soft green with slight variation
                                g = min(255, max(0, int(150 + rng2.uniform(-25, 25))))
                                color = (86, g, 87)
                                pygame.draw.polygon(screen, color, poly)

                # left bank outward_sign currently -1 -> city side is opposite (+1)
                # scatter_gap_fill(Ls_mid, outward_sign=+1, seed=707)
                # right bank outward_sign currently +1 -> city side is opposite (-1)
                # scatter_gap_fill(Rs_mid, outward_sign=-1, seed=708)
# Decorative thin shoulders near water edge (optional subtle)
                # --- Rocks: irregular polygons placed ON the bank line, slightly INSIDE the water
                def scatter_rocks_on_inside(polyline, inward_sign, density=0.90, seed=777, off_min=0.2, off_max=2.0):
                    # Re-purposed: place sparse rocks OUTSIDE the river only
                    from geometry import point_in_poly
                    rng2 = random.Random(seed)
                    stepW = 7.0
                    acc = 0.0
                    for i in range(1, len(polyline)):
                        ax,ay = polyline[i-1]; bx,by=polyline[i]
                        dx,dy = (bx-ax, by-ay)
                        segL = (dx*dx+dy*dy)**0.5 or 1.0
                        nx,ny = (dy/segL, -dx/segL)  # outward approx
                        acc += segL
                        while acc >= stepW:
                            acc -= stepW
                            t = rng2.uniform(0.15, 0.85)
                            x = ax + dx*t; y = ay + dy*t
                            # place OUTSIDE: reverse the previous 'inward' sign
                            off = rng2.uniform(off_min, off_max)
                            cx, cy = (x + nx*off*inward_sign, y + ny*off*inward_sign)
                            # skip if it would cover water
                            if point_in_poly((cx,cy), self.river_poly):
                                continue
                            if rng2.random() < density:
                                k = rng2.randint(5,7)
                                R = _zoom_px(rng2.uniform(2.0, 5.0))
                                verts = []
                                ang0 = rng2.uniform(0, math.tau)
                                for j in range(k):
                                    ang = ang0 + (j/k) * math.tau + rng2.uniform(-0.2, 0.2)
                                    rad = R * (0.6 + rng2.uniform(0.0, 0.5))
                                    px = cx + math.cos(ang)*rad
                                    py = cy + math.sin(ang)*rad
                                    verts.append(world_to_screen((px,py)))
                                pygame.draw.polygon(screen, (112,112,112), verts)
                                pygame.draw.polygon(screen, (90,90,90), verts, 1)
        # rocks disabled

                # right bank outward=-1 => inward_sign = +1
        # rocks disabled

                # inner copy closer to river
        # rocks disabled

        # rocks disabled

                # --- Grass: organic green blobs (filled, no stroke), INSIDE the river edge
                def scatter_grass_blobs(polyline, inward_sign, density=0.80, seed=909, off_min=0.2, off_max=2.5):
                    # Re-purposed: place sparse organic grass blobs OUTSIDE only
                    from geometry import point_in_poly
                    rng3 = random.Random(seed)
                    stepW = 9.0
                    for i in range(1, len(polyline)):
                        ax,ay = polyline[i-1]; bx,by=polyline[i]
                        dx,dy = (bx-ax, by-ay)
                        segL = (dx*dx+dy*dy)**0.5 or 1.0
                        nx,ny = (dy/segL, -dx/segL)
                        s = 0.0
                        while s < segL:
                            s += stepW
                            t = min(1.0, s/segL)
                            x = ax + dx*t; y = ay + dy*t
                            off = rng3.uniform(off_min, off_max)
                            lat = rng3.uniform(-3.0, 3.0)
                            cx, cy = (x + nx*off*inward_sign - ny*lat, y + ny*off*inward_sign + nx*lat)
                            if point_in_poly((cx,cy), self.river_poly):
                                continue
                            if rng3.random() < density:
                                k = rng3.randint(6,9)
                                R = _zoom_px(rng3.uniform(2.5, 6.0))
                                verts = []
                                ang0 = rng3.uniform(0, math.tau)
                                for j in range(k):
                                    ang = ang0 + (j/k) * math.tau + rng3.uniform(-0.25, 0.25)
                                    rad = R * (0.6 + rng3.uniform(0.0, 0.5))
                                    px = cx + math.cos(ang)*rad
                                    py = cy + math.sin(ang)*rad
                                    verts.append(world_to_screen((px,py)))
                                g = min(255, max(0, int(150 + rng3.uniform(-18, 18))))
                                color = (86, g, 87)
                                pygame.draw.polygon(screen, color, verts)
                # # scatter_grass_blobs(Ls_mid, inward_sign=-1, density=0.30, seed=301)
                # # scatter_grass_blobs(Rs_mid, inward_sign=+1, density=0.30, seed=302)
                # inner copy closer to river
                # # scatter_grass_blobs(Ls_mid, inward_sign=-1, density=0.30, seed=311, off_min=0.05, off_max=1.0)
                # # scatter_grass_blobs(Rs_mid, inward_sign=+1, density=0.30, seed=312, off_min=0.05, off_max=1.0)

        if self.sea_poly:
            poly = to_screen(self.sea_poly)
            pygame.draw.polygon(screen, WATER_COLOR, poly)
            pygame.draw.polygon(screen, (80,120,180), poly, max(2, int(4*cam_zoom)))
    def __init__(self, map_size):
        self.MAP_WIDTH, self.HEIGHT = map_size
        self.river_poly = None
        self.sea_poly = None

    def reset(self):
        # clear internal caches for bushes & connectors
        for k in ('_bush_pos','_bush_key','_bush_sprite','_bush_zoom','_bush_zoom_bucket','_bush_sprite_half',
                  '_conn_polys','_conn_key'):
            if hasattr(self, k):
                setattr(self, k, None)
        self.river_poly = None
        self.sea_poly = None

    
    
    def generate_river(self, seed_x=None):
        """Spline-based organic river: fewer verts (≈1/16 of original control path), very smooth."""
        import math, random

        w, h = self.MAP_WIDTH, self.HEIGHT
        rng = random.Random(random.randint(0, 1_000_000))

        # --- Build a noisy center path in local UV (many control samples), then rotate/translate
        nseg = max(120, int(max(w, h) * 0.3))
        amp  = (0.10 + rng.random()*0.20) * w
        amp2 = (0.04 + rng.random()*0.10) * w
        kink = rng.uniform(0.8, 1.4)
        rot_deg = rng.uniform(0.0, 180.0)
        rot = math.radians(rot_deg)
        cx0 = seed_x if seed_x is not None else w*0.5 + rng.uniform(-w*0.12, w*0.12)

        K = 12
        anchors = [rng.uniform(-1.0, 1.0) for _ in range(K+1)]
        def smooth1d(t):
            t = max(0.0, min(1.0, t))
            x = t * K
            i = int(math.floor(x))
            f = x - i
            a = anchors[max(0, min(K, i))]
            b = anchors[max(0, min(K, i+1))]
            mu = (1 - math.cos(f*math.pi)) * 0.5
            return a*(1-mu) + b*mu

        uv = []
        for i in range(nseg+1):
            t = i / nseg
            v = (t - 0.5) * (h * 1.8)
            dx = ( math.sin(t*2.6*kink + rng.uniform(-0.3,0.3)) * amp
                 + math.sin(t*5.7 + 1.2) * amp2
                 + smooth1d(t) * (0.10*w) )
            u = (cx0 - w*0.5) + dx
            uv.append((u, v))

        cosr, sinr = math.cos(rot), math.sin(rot)
        cx, cy = w*0.5, h*0.5
        raw = [(u*cosr - v*sinr + cx, u*sinr + v*cosr + cy) for (u,v) in uv]

        # --- Resample raw path to a small set of control points (≈ 1/16 of original), then Catmull-Rom spline
        def arclen(points):
            L = [0.0]
            for i in range(1, len(points)):
                ax,ay = points[i-1]; bx,by = points[i]
                L.append(L[-1] + ((bx-ax)**2 + (by-ay)**2)**0.5)
            return L

        L = arclen(raw)
        total = L[-1] or 1.0
        ctrl_count = max(10, len(raw)//16)   # ~1/16
        step = total / (ctrl_count-1)
        controls = [raw[0]]
        target = step
        i = 1
        while len(controls) < ctrl_count-1 and i < len(raw):
            while i < len(raw) and L[i] < target:
                i += 1
            if i >= len(raw): break
            a = raw[i-1]; b = raw[i]
            la = L[i-1]; lb = L[i]
            t = (target - la) / max(1e-9, lb-la)
            controls.append((a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t))
            target += step
        controls.append(raw[-1])

        # Centripetal Catmull-Rom spline evaluation
        def catmull_rom(pts, samples_per_seg=4):
            if len(pts) < 4: return pts
            out = [pts[0]]
            def tj(ti, pi, pj, alpha=0.5):
                dx, dy = pj[0]-pi[0], pj[1]-pi[1]
                return (dx*dx + dy*dy)**(alpha*0.5) + ti
            for i in range(len(pts)-3):
                p0, p1, p2, p3 = pts[i], pts[i+1], pts[i+2], pts[i+3]
                t0 = 0.0
                t1 = tj(t0, p0, p1)
                t2 = tj(t1, p1, p2)
                t3 = tj(t2, p2, p3)
                for s in range(1, samples_per_seg+1):
                    t = t1 + (t2-t1) * (s/samples_per_seg)
                    def C(pa, pb, ta, tb, t):
                        if tb-ta == 0: return pb
                        return ((tb - t)/(tb - ta))*pa[0] + ((t - ta)/(tb - ta))*pb[0], ((tb - t)/(tb - ta))*pa[1] + ((t - ta)/(tb - ta))*pb[1]
                    A1 = C(p0, p1, t0, t1, t)
                    A2 = C(p1, p2, t1, t2, t)
                    A3 = C(p2, p3, t2, t3, t)
                    B1 = C(A1, A2, t0, t2, t)
                    B2 = C(A2, A3, t1, t3, t)
                    Cc = C(B1, B2, t1, t2, t)
                    out.append(Cc)
            out.append(pts[-1])
            return out

        # Pad ends for spline (duplicate endpoints)
        if len(controls) < 4:
            center = controls
        else:
            padded = [controls[0]] + controls + [controls[-1]]
            center = catmull_rom(padded, samples_per_seg=4)

        # Reduce to a clean, smooth low-vertex chain (halve again relative to last build)
        # Keep ~half of spline samples uniformly
        if len(center) > 20:
            keep = max(14, len(center)//2)
            # uniform resample by count
            SL = arclen(center); T = SL[-1] or 1.0; st = T/(keep-1)
            out=[center[0]]; tgt=st; j=1
            while len(out) < keep-1 and j < len(center):
                while j < len(center) and SL[j] < tgt: j += 1
                if j >= len(center): break
                a=center[j-1]; b=center[j]; la=SL[j-1]; lb=SL[j]
                tt=(tgt-la)/max(1e-9, lb-la)
                out.append((a[0]+(b[0]-a[0])*tt, a[1]+(b[1]-a[1])*tt))
                tgt += st
            out.append(center[-1])
            center = out

        # Clamp local angle (≥160°) and run a light moving average twice
        def angle(p,q,r):
            vx1,vy1=p[0]-q[0],p[1]-q[1]; vx2,vy2=r[0]-q[0],r[1]-q[1]
            L1=(vx1*vx1+vy1*vy1)**0.5 or 1.0; L2=(vx2*vx2+vy2*vy2)**0.5 or 1.0
            d=max(-1.0,min(1.0,(vx1*vx2+vy1*vy2)/(L1*L2)))
            return math.degrees(math.acos(d))
        for _ in range(2):
            for i in range(1,len(center)-1):
                if angle(center[i-1],center[i],center[i+1]) < 120.0:
                    ax,ay=center[i-1]; bx,by=center[i]; cx,cy=center[i+1]
                    center[i]=((ax+2*bx+cx)/4.0,(ay+2*by+cy)/4.0)


        # Enforce a strict minimum internal angle along the centerline
        def _internal_angle(p, q, r):
            vx1, vy1 = p[0]-q[0], p[1]-q[1]
            vx2, vy2 = r[0]-q[0], r[1]-q[1]
            L1 = (vx1*vx1 + vy1*vy1)**0.5 or 1.0
            L2 = (vx2*vx2 + vy2*vy2)**0.5 or 1.0
            dot = max(-1.0, min(1.0, (vx1*vx2 + vy1*vy2)/(L1*L2)))
            return math.degrees(math.acos(dot))

        def _relax_min_turn(poly, min_deg=120.0, passes=10):
            pts = list(poly)
            lim = max(60.0, min(175.0, float(min_deg)))
            for _ in range(passes):
                changed = False
                for i in range(1, len(pts)-1):
                    a, b, c = pts[i-1], pts[i], pts[i+1]
                    ang = _internal_angle(a, b, c)
                    if ang < lim:
                        # Move the middle point toward the secant midpoint to open the angle
                        mx, my = ( (a[0] + c[0]) * 0.5, (a[1] + c[1]) * 0.5 )
                        bx, by = b
                        # Heavier pull when angle is sharper
                        t = min(0.85, max(0.35, (lim - ang)/lim))
                        pts[i] = (bx*(1.0 - t) + mx*t, by*(1.0 - t) + my*t)
                        changed = True
                if not changed:
                    break
            return pts

        center = _relax_min_turn(center, min_deg=120.0, passes=12)
        # Width profile (smooth)
        widths=[]
        for i in range(len(center)):
            t=i/max(1,len(center)-1)
            base=w*(0.018+0.010*rng.random())
            swell1=(math.sin(t*2.0+rng.uniform(0,1.0))*0.5+0.5)*(w*0.014)
            swell2=(math.sin(t*5.0+rng.uniform(0,6.0))*0.5+0.5)*(w*0.010)
            widths.append(base+swell1+swell2)
        for k in range(2):
            for i in range(1,len(widths)-1):
                widths[i]=(widths[i-1]*0.25+widths[i]*0.5+widths[i+1]*0.25)

        
        # Slightly widen river
        widths = [v * 1.15 for v in widths]
# Build banks with minimal jitter
        left,right=[],[]
        for i in range(len(center)):
            p=center[i]; p0=center[max(0,i-1)]; p1=center[min(len(center)-1,i+1)]
            vx=p1[0]-p0[0]; vy=p1[1]-p0[1]
            L=(vx*vx+vy*vy)**0.5 or 1.0
            nx,ny=vy/L,-vx/L
            hw=widths[i]*0.5
            left.append((p[0]-nx*hw, p[1]-ny*hw))
            right.append((p[0]+nx*hw, p[1]+ny*hw))

        self.river_poly = left + right[::-1]
    
    def generate_sea(self, side='bottom'):
        w,h=self.MAP_WIDTH, self.HEIGHT
        pad = int(h*0.12)
        if side=='bottom':
            self.sea_poly = [(0,h-pad),(w,h-pad),(w,h),(0,h)]
        elif side=='top':
            self.sea_poly = [(0,0),(w,0),(w,pad),(0,pad)]
        elif side=='left':
            self.sea_poly = [(0,0),(pad,0),(pad,h),(0,h)]
        else:
            self.sea_poly = [(w-pad,0),(w,0),(w,h),(w-pad,h)]
    
    def draw(self, screen, world_to_screen, cam_zoom):
        import pygame
        if self.river_poly:
            poly = [world_to_screen(p) for p in self.river_poly]
            pygame.draw.polygon(screen, WATER_COLOR, poly)
            pygame.draw.polygon(screen, (80,120,180), poly, max(1,int(2*cam_zoom)))
        if self.sea_poly:
            poly = [world_to_screen(p) for p in self.sea_poly]
            pygame.draw.polygon(screen, WATER_COLOR, poly)
            pygame.draw.polygon(screen, (80,120,180), poly, max(1,int(2*cam_zoom)))

    def serialize_state(self):
        return {
            "river_poly": [tuple(p) for p in self.river_poly] if self.river_poly else None,
            "sea_poly": [tuple(p) for p in self.sea_poly] if self.sea_poly else None,
        }
    def restore_state(self, state):
        self.river_poly = [tuple(p) for p in state.get("river_poly")] if state.get("river_poly") else None
        self.sea_poly   = [tuple(p) for p in state.get("sea_poly")] if state.get("sea_poly") else None

def rebuild_roads_around_river(roads, water, bank_offset):
    if not water or not water.river_poly: return
    rp = water.river_poly
    from collections import defaultdict
    from quadtree import Quadtree
    from geometry import seg_intersection, segment_intersects_polygon, point_in_poly, seg_aabb
    from math import hypot

    saved = {
        "SNAP_RADIUS_NODE": roads.params.get("SNAP_RADIUS_NODE"),
        "SNAP_RADIUS_SEG":  roads.params.get("SNAP_RADIUS_SEG"),
        "HOUSE_STEP_MIN":   roads.params.get("HOUSE_STEP_MIN"),
        "MIN_PARALLEL_ANGLE_DEG": roads.params.get("MIN_PARALLEL_ANGLE_DEG"),
    }
    roads.params["SNAP_RADIUS_NODE"] = max(4, int(min(10, roads.params.get("SNAP_RADIUS_NODE", 12))))
    roads.params["SNAP_RADIUS_SEG"]  = max(4, int(min(10, roads.params.get("SNAP_RADIUS_SEG", 12))))
    roads.params["HOUSE_STEP_MIN"]   = 0
    roads.params["MIN_PARALLEL_ANGLE_DEG"] = 0
    try:
        had_roads = any(s.active for s in roads.segments)
        kept=[]; cut_pts=[]
        def clip_segment_outside(a,b,poly):
            pts=[]; n=len(poly)
            for i in range(n):
                p1=poly[i]; p2=poly[(i+1)%n]
                ok,P,t,_=seg_intersection(a,b,p1,p2)
                if ok and P is not None and 0.0<=t<=1.0: pts.append((t,P))
            pts.sort(key=lambda x:x[0])
            inside = point_in_poly(a, poly)
            out=[]; t_prev=0.0
            for t_curr,P in pts:
                if not inside:
                    S=(a[0]+(b[0]-a[0])*t_prev, a[1]+(b[1]-a[1])*t_prev)
                    out.append((S,P))
                inside = not inside; t_prev=t_curr
            if not inside:
                S=(a[0]+(b[0]-a[0])*t_prev, a[1]+(b[1]-a[1])*t_prev)
                out.append((S,b))
            return out
        for s in list(roads.segments):
            if not s.active: continue
            a,b=s.a,s.b
            if segment_intersects_polygon(a,b,rp):
                pieces=clip_segment_outside(a,b,rp)
                pts=[]; n=len(rp)
                for i in range(n):
                    p1=rp[i]; p2=rp[(i+1)%n]
                    ok,P,t,_=seg_intersection(a,b,p1,p2)
                    if ok and P is not None and 0.0<=t<=1.0: pts.append(P)
                for (A,B) in pieces:
                    if hypot(B[0]-A[0], B[1]-A[1])>1.0: kept.append((A,B,s.level))
                cut_pts.extend(pts)
            else:
                kept.append((a,b,s.level))
        roads.nodes=[]; roads.road_graph=defaultdict(list)
        roads.qt = Quadtree((roads.WORLD_X, roads.WORLD_Y, roads.WORLD_WIDTH, roads.HEIGHT*2))
        roads.segments=[]; roads.segment_names={}
        roads.river_bridge_segments.clear(); roads.river_skip_segments.clear(); roads.river_cross_count=0
        for (A,B,lvl) in kept:
            roads.add_manual_segment(A,B, level=lvl, water=water, min_spacing_override=7)
        if not had_roads: return
        left,right = _poly_split_left_right(rp)
        offL = _offset_polyline_outside(left, rp, bank_offset)
        offR = _offset_polyline_outside(right, rp, bank_offset)
        def stitch(poly):
            if len(poly) < 2: return
            step = max(18.0, bank_offset*1.5)
            poly = _resample_polyline(poly, maxlen=step)
            for i in range(len(poly)-1):
                A,B=poly[i], poly[i+1]
                if hypot(B[0]-A[0], B[1]-A[1])<1.0: continue
                roads.add_manual_segment(A,B, level=1, water=water, min_spacing_override=7)
        stitch(offL); stitch(offR)
        def poly_min_dist(P, poly):
            best=1e9
            for i in range(len(poly)):
                ax,ay=poly[i]; bx,by=poly[(i+1)%len(poly)]
                vx,vy=(bx-ax,by-ay); wx,wy=(P[0]-ax,P[1]-ay)
                L2=vx*vx+vy*vy; t=0.0 if L2<=1e-9 else max(0.0,min(1.0,(wx*vx+wy*vy)/L2))
                proj=(ax+vx*t, ay+vy*t)
                d=((proj[0]-P[0])**2 + (proj[1]-P[1])**2)**0.5
                if d<best: best=d
            return best
        def nearest_point_on_boundary(P, poly):
            best=None; best_d=1e9; best_edge=None
            for i in range(len(poly)):
                a=poly[i]; b=poly[(i+1)%len(poly)]
                ax,ay=a; bx,by=b; px,py=P
                vx,vy=(bx-ax,by-ay); wx,wy=(px-ax,py-ay)
                L2=vx*vx+vy*vy
                t=0.0 if L2<=1e-9 else max(0.0,min(1.0,(wx*vx+wy*vy)/L2))
                Q=(ax+vx*t, ay+vy*t)
                d=((Q[0]-px)**2 + (Q[1]-py)**2)**0.5
                if d<best_d: best=(Q,i,t); best_d=d; best_edge=(a,b)
            ax,ay=best_edge[0]; bx,by=best_edge[1]
            ex,ey=(bx-ax,by-ay); L=(ex*ex+ey*ey)**0.5 or 1.0
            nx,ny=(ey/L, -ex/L)
            from geometry import point_in_poly
            test=(best[0][0]+nx*2.0, best[0][1]+ny*2.0)
            if point_in_poly(test, poly): nx,ny=(-nx,-ny)
            return best[0], (nx,ny)
        near_nodes=[]
        for idx,nbrs in roads.road_graph.items():
            if len(nbrs)==1:
                P=roads.nodes[idx]
                if poly_min_dist(P,rp)<=bank_offset*2.25: near_nodes.append(P)
        cands = near_nodes + cut_pts
        for P in cands:
            # Multi-offset outward targets from river boundary (more chances to attach)
            Q, nrm = nearest_point_on_boundary(P, rp)
            # Try three distances outward
            for d in (12.0, 13.0, 14.0):
                T0 = (Q[0] + nrm[0]*d, Q[1] + nrm[1]*d)
                # Prefer snapping to tertiary (level>=2) segments if present
                proj, seg, t = roads.try_snap_segment(T0)
                T = T0
                if seg is not None and getattr(seg, "level", 2) >= 2:
                    T = proj
                ok = roads.add_manual_segment(P, T, level=2, water=water, min_spacing_override=0)
                if ok:
                    break
    finally:
        roads.params.update(saved)


def _resample_polyline(poly, maxlen):
    # Uniformly resample a polyline so adjacent vertices are at most maxlen apart.
    from math import hypot
    if len(poly) < 2: return list(poly)
    out=[poly[0]]
    for i in range(len(poly)-1):
        ax,ay = poly[i]; bx,by = poly[i+1]
        dx,dy = (bx-ax, by-ay); d = hypot(dx,dy)
        if d <= 1e-6: continue
        n = int(d // maxlen)
        for k in range(1, n+1):
            t = min(1.0, (k*maxlen)/d)
            out.append((ax + dx*t, ay + dy*t))
    if out[-1] != poly[-1]: out.append(poly[-1])
    return out

def _proj_point_to_polyline(P, poly):
    # Return nearest projected point on a polyline
    best=None; best_d=1e9
    for i in range(len(poly)-1):
        ax,ay=poly[i]; bx,by=poly[i+1]
        vx,vy=(bx-ax,by-ay); wx,wy=(P[0]-ax,P[1]-ay)
        L2=vx*vx+vy*vy
        t=0.0 if L2<=1e-9 else max(0.0, min(1.0, (wx*vx+wy*vy)/L2))
        proj=(ax+vx*t, ay+vy*t)
        d=((proj[0]-P[0])**2 + (proj[1]-P[1])**2)**0.5
        if d<best_d: best=(proj,i,t); best_d=d
    return best[0] if best else poly[0]
def _poly_split_left_right(river_poly):
    n=len(river_poly); k=n//2
    left=river_poly[:k]; right=list(reversed(river_poly[k:]))
    return left,right

def _offset_polyline_outside(polyline, poly, offset):
    from geometry import point_in_poly
    out=[]; n=len(polyline)
    if n==0: return out
    for i in range(n):
        p=polyline[i]; j=i+1 if i+1<n else i; pj=polyline[j]
        vx,vy=(pj[0]-p[0], pj[1]-p[1]); L=(vx*vx+vy*vy)**0.5 or 1.0
        ux,uy=(vx/L,vy/L); nx1,ny1=(uy,-ux); nx2,ny2=(-uy,ux)
        test1=(p[0]+nx1*2.0, p[1]+ny1*2.0)
        nx,ny=(nx1,ny1) if not point_in_poly(test1,poly) else (nx2,ny2)
        out.append((p[0]+nx*offset, p[1]+ny*offset))
    return out


# --------- visual smooth connectors drawn in the water layer (not part of road graph) ---------

def draw_connectors(screen, world_to_screen, cam_zoom, roads, water):
    """Visual-only smooth bezier connectors from river to nearest road.
    Kept subtle: max 3 connectors; skip far targets; avoid ends of the river.
    """
    import pygame as _pg
    from geometry import point_seg_dist
    
    if not water or not getattr(water, "river_poly", None) or not roads:
        return
    
    rp = list(water.river_poly)
    if len(rp) < 8:
        return
    
    # sample three anchor positions away from the ends of the river polyline
    candidate_idx = []
    for t in (0.25, 0.5, 0.75):
        idx = int(t * (len(rp)-1))
        idx = max(3, min(len(rp)-4, idx))
        candidate_idx.append(idx)
    anchors = [rp[i] for i in candidate_idx]
    
    def nearest(P):
        best=None; best_d=1e9; best_seg=None
        for s in roads.segments:
            if not getattr(s,'active',True): 
                continue
            if hasattr(roads,'decorative_segments') and s in roads.decorative_segments: 
                continue
            if hasattr(roads,'river_bridge_segments') and s in roads.river_bridge_segments: 
                continue
            d, proj, t = point_seg_dist(P, s.a, s.b)
            if t is None: 
                continue
            if d < best_d:
                best_d=d; best=proj; best_seg=s
        return best, best_seg, best_d
    
    def draw_bezier(C0, C1, C2, C3, width):
        S = max(14, int((((C3[0]-C0[0])**2 + (C3[1]-C0[1])**2)**0.5) / 14))
        pts=[]
        for i in range(S+1):
            t=i/float(S); u=1.0-t
            x = (u*u*u*C0[0] + 3*u*u*t*C1[0] + 3*u*t*t*C2[0] + t*t*t*C3[0])
            y = (u*u*u*C0[1] + 3*u*u*t*C1[1] + 3*u*t*t*C2[1] + t*t*t*C3[1])
            pts.append(world_to_screen((x,y)))
        _pg.draw.lines(screen, (150,154,162), False, pts, max(1, int(width)))
        _pg.draw.lines(screen, (110,113,120), False, pts, max(1, int(width*0.15)))
    
    MAX_CONNECTORS = 3
    MAX_DIST = 180.0  # skip projections too far away (prevents 'spitting')
    count=0
    for P in anchors:
        proj, seg, d = nearest(P)
        if proj is None or d > MAX_DIST:
            continue
        # local river tangent/normal
        idx = rp.index(P)
        p0 = rp[max(0, idx-1)]; p1 = rp[min(len(rp)-1, idx+1)]
        vx, vy = (p1[0]-p0[0], p1[1]-p0[1]); Lv = (vx*vx+vy*vy)**0.5 or 1.0
        tx, ty = (vx/Lv, vy/Lv); nx, ny = (ty, -tx)
        if (proj[0]-P[0])*nx + (proj[1]-P[1])*ny < 0: nx,ny = -nx,-ny
        # road direction at proj
        if seg is not None:
            rx, ry = (seg.b[0]-seg.a[0], seg.b[1]-seg.a[1]); Lr = (rx*rx+ry*ry)**0.5 or 1.0
            rx, ry = rx/Lr, ry/Lr
        else:
            rx, ry = nx, ny
        dist = (((proj[0]-P[0])**2 + (proj[1]-P[1])**2)**0.5)
        h1 = dist * 0.33; h2 = dist * 0.33
        C0 = P; C1 = (P[0] + nx*h1, P[1] + ny*h1)
        C3 = proj; C2 = (proj[0] - rx*h2, proj[1] - ry*h2)
        width = float(roads.params['ROAD_WIDTH'].get(2, 4))
        draw_bezier(C0, C1, C2, C3, width)
        count += 1
        if count >= MAX_CONNECTORS:
            break


# --------- visual smooth connectors at bridge ends to nearest road ---------


def draw_bridge_connectors(screen, world_to_screen, cam_zoom, roads, water, bridges_mod):
    import pygame as _pg
    from geometry import point_seg_dist
    if not roads or not water or not getattr(bridges_mod, '_BRIDGE_GEOMS', None):
        return

    MAX_DIST = 260.0  # skip far targets

    def width_px():
        base_w = float(roads.params['ROAD_WIDTH'].get(1, 6))
        return max(2, int(base_w * cam_zoom * 1.6))  # scale with zoom

    def nearest(P):
        best=None; best_d=1e9; best_seg=None
        for s in roads.segments:
            if not getattr(s,'active',True): 
                continue
            if hasattr(roads,'decorative_segments') and s in roads.decorative_segments: 
                continue
            if hasattr(roads,'river_bridge_segments') and s in roads.river_bridge_segments: 
                continue
            d, proj, t = point_seg_dist(P, s.a, s.b)
            if t is None:
                continue
            if d < best_d:
                best_d=d; best=proj; best_seg=s
        return best, best_seg, best_d

    def blend_dir(ax, ay, bx, by, k=0.45):
        # Blend two unit vectors
        import math
        L1 = math.hypot(ax, ay) or 1.0; L2 = math.hypot(bx, by) or 1.0
        ax, ay = ax/L1, ay/L1; bx, by = bx/L2, by/L2
        cx, cy = (ax*(1-k) + bx*k, ay*(1-k) + by*k)
        L = (cx*cx+cy*cy)**0.5 or 1.0
        return (cx/L, cy/L)

    def draw_bezier(C0, C1, C2, C3):
        S = max(36, int((((C3[0]-C0[0])**2 + (C3[1]-C0[1])**2)**0.5) / 6))
        pts=[]
        for i in range(S+1):
            t=i/float(S); u=1.0-t
            x = (u*u*u*C0[0] + 3*u*u*t*C1[0] + 3*u*t*t*C2[0] + t*t*t*C3[0])
            y = (u*u*u*C0[1] + 3*u*u*t*C1[1] + 3*u*t*t*C2[1] + t*t*t*C3[1])
            pts.append(world_to_screen((x,y)))
        _pg.draw.lines(screen, (170,174,186), False, pts, width_px())  # single stroke, no dark center

    for (A,B, lvl) in getattr(bridges_mod, '_BRIDGE_GEOMS', []):
        dx, dy = (B[0]-A[0], B[1]-A[1])
        L = (dx*dx+dy*dy)**0.5 or 1.0
        tx, ty = dx/L, dy/L
        for P, sign in ((A, -1), (B, +1)):
            sx, sy = (tx*sign, ty*sign)
            proj, seg, dist = nearest(P)
            if proj is None or dist > MAX_DIST:
                continue
            if seg is not None:
                rx, ry = (seg.b[0]-seg.a[0], seg.b[1]-seg.a[1])
                Lr = (rx*rx+ry*ry)**0.5 or 1.0
                rx, ry = rx/Lr, ry/Lr
            else:
                rx, ry = sx, sy
            # blend start handle toward the target vector to avoid kink
            to_tx, to_ty = (proj[0]-P[0], proj[1]-P[1])
            Lt = (to_tx*to_tx + to_ty*to_ty)**0.5 or 1.0
            to_tx, to_ty = to_tx/Lt, to_ty/Lt
            hx, hy = blend_dir(sx, sy, to_tx, to_ty, k=0.5)
            d = Lt
            h1 = d * 0.65  # longer handles for smoother turn
            h2 = d * 0.65
            C0 = P
            C1 = (P[0] + hx*h1, P[1] + hy*h1)
            C3 = proj
            C2 = (proj[0] - rx*h2, proj[1] - ry*h2)
            draw_bezier(C0, C1, C2, C3)






def draw_grass_top(screen, world_to_screen, cam_zoom, water):
    """Draw bushes as instances of a single prototype sprite for speed."""
    import pygame as _pg, math
    if not water or not getattr(water, 'river_poly', None): return
    rp = list(water.river_poly)
    if len(rp) < 8: return
    def _river_key(poly):
        m=0
        for x,y in poly: m = (m*1315423911 ^ int(x*1000) ^ (int(y*1000)<<11)) & 0xffffffff
        return m
    def poly_len(poly):
        L=0.0
        for i in range(1,len(poly)):
            ax,ay=poly[i-1]; bx,by=poly[i]; L += math.hypot(bx-ax, by-ay)
        return L
    def point_at(poly, dist):
        acc=0.0
        for i in range(1,len(poly)):
            ax,ay=poly[i-1]; bx,by=poly[i]; seg=math.hypot(bx-ax, by-ay)
            if acc+seg >= dist:
                s=(dist-acc)/(seg or 1.0); x=ax+(bx-ax)*s; y=ay+(by-ay)*s
                tx,ty=(bx-ax,by-ay); L=max(1e-9, math.hypot(tx,ty)); return (x,y),(tx/L,ty/L)
            acc+=seg
        ax,ay=poly[-2]; bx,by=poly[-1]; tx,ty=(bx-ax,by-ay); L=max(1e-9, math.hypot(tx,ty)); return poly[-1],(tx/L,ty/L)
    def rng(ix,iy,seed): return ((ix*2246822519 ^ iy*3266489917 ^ seed) & 0xffffffff)/4294967296.0
    key=_river_key(rp)
    if getattr(water,'_bush_key',None)!=key or not hasattr(water,'_bush_pos'):
        n=len(rp); k=n//2; left=rp[:k]; right=list(reversed(rp[k:]))
        def gen(poly, sign, seed):
            out=[]; total=poly_len(poly); SPACING=72.0; d=0.0
            while d<total:
                P,T=point_at(poly,d); nx,ny=(T[1],-T[0]); nx*=sign; ny*=sign
                ix=int(P[0]//SPACING); iy=int(P[1]//SPACING)
                r2=rng(ix+7,iy-9,seed^0x9E3779B9)
                r3=rng(ix-3,iy+11,seed^0x85EBCA6B)
                along=(r2-0.5)*0.6*SPACING; off=4.0+10.0*r3
                cx=P[0]+T[0]*along+nx*off; cy=P[1]+T[1]*along+ny*off
                out.append((cx,cy)); d += SPACING*(0.85+0.6*r2)
            return out
        water._bush_pos = gen(left,-1,0x1234ABCD)+gen(right,+1,0xBEEF00D5)
        water._bush_key = key; water._bush_sprite=None; water._bush_zoom=None
    zb=int(cam_zoom*100); R=4.0
    if getattr(water,'_bush_sprite',None) is None or getattr(water,'_bush_zoom',None)!=zb:
        radius_px=max(2,int(R*cam_zoom)); pad=2; size=radius_px*2+pad*2
        surf=_pg.Surface((size,size)); surf.fill((255,0,255)); surf.set_colorkey((255,0,255))
        cx=cy=size//2; base=[(0.00,-1.00),(0.86,-0.48),(0.78,0.34),(0.00,1.00),(-0.72,0.42),(-0.92,-0.18)]
        pts=[(int(cx+ux*radius_px),int(cy+uy*radius_px)) for ux,uy in base]
        _pg.draw.polygon(surf,(84,176,96),pts)
        water._bush_sprite=surf; water._bush_half=(size//2,size//2); water._bush_zoom=zb
    spr=water._bush_sprite; hx,hy=water._bush_half; W=screen.get_width(); H=screen.get_height()
    for wx,wy in water._bush_pos:
        sx,sy=world_to_screen((wx,wy))
        if -hx<=sx<=W+hx and -hy<=sy<=H+hy: screen.blit(spr,(sx-hx,sy-hy))
