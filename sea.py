
import math, random
from geometry import point_in_poly

# Version stamp for sanity in console
SEA_MODULE_VERSION = "sea_v9.0"
print("[Sea]", SEA_MODULE_VERSION)

WATER_COLOR = (120, 180, 255)
BANK_COLOR  = (164, 168, 176)   # match road gray   # brown dust road (landside)
BAND_FILL   = (198, 214, 187)  # soft bank fill (landside patch)

class SeaSystem:
    def __init__(self, map_size):
        self.MAP_WIDTH, self.HEIGHT = map_size
        self.sea_poly   = None
        self.coast_path = None
        self.harbor_breakwaters = []
        self.harbor_piers = []
        self.harbor_cement = []

    def reset(self):
        self.sea_poly = None
        self.coast_path = None
        self.harbor_breakwaters = []
        self.harbor_piers = []
        self.harbor_cement = []

    # ---------------- Coast Generation ----------------
    def generate(self, side='bottom'):
        """Build an organic coastline (spline), clamp sharp turns, set sea polygon.
           Auto-generates a harbor afterward."""
        print("[Sea] generate", side)
        # reset harbor structures
        self.harbor_breakwaters = []
        self.harbor_piers = []
        self.harbor_cement = []

        w, h = self.MAP_WIDTH, self.HEIGHT
        rng = random.Random(random.randint(0, 1_000_000))

        # Push sea further inland: smaller pad, higher drift amplitude
        pad = max(18.0, 0.22 * (h if side in ('top','bottom') else w))

        N = 160   # raw samples along edge
        K = 10    # noise anchors
        anchors = [rng.uniform(-1.0, 1.0) for _ in range(K+1)]
        def noise01(t):
            t = max(0.0, min(1.0, t))
            x = t*K
            i = int(x); f = x - i
            a = anchors[max(0, min(K, i))]
            b = anchors[max(0, min(K, i+1))]
            mu = (1 - math.cos(f*math.pi)) * 0.5
            return a*(1-mu) + b*mu

        raw = []
        if side in ('bottom','top'):
            for i in range(N+1):
                t = i/N
                x = t*w
                y0 = (h - pad) if side=='bottom' else pad
                drift = (math.sin(t*4.5 + 1.0)*0.4 + math.sin(t*8.0+2.0)*0.25 + noise01(t)*0.6)
                # amplitude 0.09*h (was 0.08)
                y = y0 + (drift * (0.09*h if side=='bottom' else -0.09*h))
                raw.append((x,y))
        else:
            for i in range(N+1):
                t = i/N
                y = t*h
                x0 = pad if side=='left' else (w - pad)
                drift = (math.sin(t*4.2 + 1.3)*0.4 + math.sin(t*7.6+2.1)*0.25 + noise01(t)*0.6)
                x = x0 + (drift * (0.09*w if side=='right' else -0.09*w))
                raw.append((x,y))

        # Arc-length controls and Catmull–Rom
        def arclen(pts):
            L=[0.0]
            for i in range(1,len(pts)):
                ax,ay=pts[i-1]; bx,by=pts[i]
                L.append(L[-1] + ((bx-ax)**2+(by-ay)**2)**0.5)
            return L
        L = arclen(raw); T = L[-1] or 1.0
        ctrl = max(12, len(raw)//16)
        step = T/(ctrl-1)
        controls=[raw[0]]; tgt=step; j=1
        while len(controls)<ctrl-1 and j<len(raw):
            while j<len(raw) and L[j]<tgt: j+=1
            if j>=len(raw): break
            a=raw[j-1]; b=raw[j]; la=L[j-1]; lb=L[j]
            t=(tgt-la)/max(1e-9, lb-la)
            controls.append((a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t))
            tgt+=step
        controls.append(raw[-1])

        def catmull_rom(pts, samples_per_seg=6):
            if len(pts)<4: return pts
            out=[pts[0]]
            def tj(ti, pi, pj, alpha=0.5):
                dx,dy=pj[0]-pi[0], pj[1]-pi[1]
                return (dx*dx+dy*dy)**(alpha*0.5)+ti
            for i in range(len(pts)-3):
                p0,p1,p2,p3=pts[i],pts[i+1],pts[i+2],pts[i+3]
                t0=0.0; t1=tj(t0,p0,p1); t2=tj(t1,p1,p2); t3=tj(t2,p2,p3)
                for s in range(1, samples_per_seg+1):
                    t=t1+(t2-t1)*(s/samples_per_seg)
                    def C(pa,pb,ta,tb,t):
                        if tb-ta==0: return pb
                        return ((tb-t)/(tb-ta))*pa[0]+((t-ta)/(tb-ta))*pb[0], ((tb-t)/(tb-ta))*pa[1]+((t-ta)/(tb-ta))*pb[1]
                    A1=C(p0,p1,t0,t1,t); A2=C(p1,p2,t1,t2,t); A3=C(p2,p3,t2,t3,t)
                    B1=C(A1,A2,t0,t2,t); B2=C(A2,A3,t1,t3,t)
                    out.append(C(B1,B2,t1,t2,t))
            out.append(pts[-1]); return out

        coast = controls if len(controls)<4 else catmull_rom([controls[0]]+controls+[controls[-1]], 6)

        # Relax sharp turns to >= 120°
        def angle(p,q,r):
            vx1,vy1=p[0]-q[0],p[1]-q[1]; vx2,vy2=r[0]-q[0],r[1]-q[1]
            L1=(vx1*vx1+vy1*vy1)**0.5 or 1.0; L2=(vx2*vx2+vy2*vy2)**0.5 or 1.0
            d=max(-1.0,min(1.0,(vx1*vx2+vy1*vy2)/(L1*L2)))
            return math.degrees(math.acos(d))
        for _ in range(12):
            changed=False
            for i in range(1,len(coast)-1):
                if angle(coast[i-1],coast[i],coast[i+1])<120.0:
                    ax,ay=coast[i-1]; bx,by=coast[i]; cx,cy=coast[i+1]
                    coast[i]=((ax+2*bx+cx)/4.0,(ay+2*by+cy)/4.0); changed=True
            if not changed: break

        # Build sea polygon from selected side
        if side=='bottom':
            sea = [(0,h),(w,h)] + list(reversed(coast)) + [(0,h)]
        elif side=='top':
            sea = [(0,0),(w,0)] + coast + [(0,0)]
        elif side=='left':
            sea = [(0,0),(0,h)] + list(reversed(coast)) + [(0,0)]
        else:  # right
            sea = [(w,0),(w,h)] + coast + [(w,0)]

        self.coast_path = coast
        self.sea_poly   = sea

        # Auto-generate a harbor
        self.generate_harbor()

    # ---------------- Overlay Drawing (above roads) ----------------
    def draw_overlay(self, screen, world_to_screen, cam_zoom):
        import pygame
        if not self.sea_poly or not self.coast_path:
            return

        def to_screen(poly):
            return [world_to_screen(p) for p in poly]
        def _px(n): return n / max(0.001, cam_zoom)
        def _zoom_px(n): return (n * (0.8 + 0.6*cam_zoom)) / max(0.001, cam_zoom)

        # Base sea fill + outline
        poly = to_screen(self.sea_poly)
        pygame.draw.polygon(screen, WATER_COLOR, poly)
        pygame.draw.polygon(screen, (80,120,180), poly, max(2, int(4*cam_zoom)))

        coast = list(self.coast_path)

        # Thin coast line for visibility
        if len(coast) >= 2:
            pygame.draw.lines(screen, (60,100,160), False, [world_to_screen(p) for p in coast], max(1, int(1*cam_zoom)))

        # Resample helpers
        def sample_polyline(polyline, step_world=10.0):
            out=[polyline[0]]; acc=0.0
            for i in range(1,len(polyline)):
                ax,ay=polyline[i-1]; bx,by=polyline[i]
                dx,dy=bx-ax, by-ay; d=(dx*dx+dy*dy)**0.5; acc+=d
                if acc>=step_world: out.append((bx,by)); acc=0.0
            out.append(polyline[-1]); return out
        def _resample_by_count(polyline, count):
            pts=list(polyline); count=max(2,int(count))
            if len(pts)<=2: return pts
            L=[0.0]
            for i in range(1,len(pts)):
                ax,ay=pts[i-1]; bx,by=pts[i]
                L.append(L[-1]+((bx-ax)**2+(by-ay)**2)**0.5)
            total=L[-1] or 1.0; step=total/(count-1)
            out=[pts[0]]; tgt=step; j=1
            while len(out)<count-1 and j<len(pts):
                while j<len(pts) and L[j]<tgt: j+=1
                if j>=len(pts): break
                a=pts[j-1]; b=pts[j]; la=L[j-1]; lb=L[j]
                t=(tgt-la)/max(1e-9, lb-la)
                out.append((a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)); tgt+=step
            out.append(pts[-1]); return out
        def offset_polyline(samples, outward_sign, off):
            out=[]
            for i in range(1,len(samples)):
                P=samples[i-1]; Q=samples[i]
                vx,vy=Q[0]-P[0], Q[1]-P[1]; L=(vx*vx+vy*vy)**0.5 or 1.0
                nx,ny=vy/L, -vx/L
                if i==1: out.append((P[0]+nx*off*outward_sign, P[1]+ny*off*outward_sign))
                out.append((Q[0]+nx*off*outward_sign, Q[1]+ny*off*outward_sign))
            return out
        def offset_path(samples, outward_sign=+1, base_offset=12.0, jitter=0.35):
            out=[]; rng=random.Random(1337)
            for i in range(1,len(samples)):
                P=samples[i-1]; Q=samples[i]
                vx,vy=Q[0]-P[0], Q[1]-P[1]; L=(vx*vx+vy*vy)**0.5 or 1.0
                nx,ny=vy/L, -vx/L; off = base_offset + rng.uniform(-jitter, jitter)
                if i==1: out.append((P[0]+nx*off*outward_sign, P[1]+ny*off*outward_sign))
                out.append((Q[0]+nx*off*outward_sign, Q[1]+ny*off*outward_sign))
            return out

        # Determine landward normal robustly (test both sides)
        def land_sign(polyline):
            if len(polyline) < 2:
                return +1
            ax, ay = polyline[0]; bx, by = polyline[1]
            dx, dy = bx-ax, by-ay; L=(dx*dx+dy*dy)**0.5 or 1.0
            nx, ny = dy/L, -dx/L
            d = 40.0
            left  = (ax + nx*d, ay + ny*d)
            right = (ax - nx*d, ay - ny*d)
            left_in  = point_in_poly(left,  self.sea_poly)
            right_in = point_in_poly(right, self.sea_poly)
            if left_in and not right_in:  # left points to sea -> land is -1
                return -1
            if right_in and not left_in:  # right points to sea -> land is +1
                return +1
            return +1

        step = max(10.0, 14.0 / max(0.5, cam_zoom))
        C = sample_polyline(coast, step_world=step)
        C_mid  = _resample_by_count(C, 60)
        C_road = _resample_by_count(C, 40)
        sign = land_sign(C)

        # Styles
        road_w   = max(6, int(8*cam_zoom))
        inner_off = -1.5
        outer_off = 12.8

        # Patch band + coast road
        C_inner  = offset_polyline(C_mid, outward_sign=sign, off=inner_off)
        C_outer  = offset_polyline(C_mid, outward_sign=sign, off=outer_off)
        C_path   = offset_path(C_road, outward_sign=sign, base_offset=12.0, jitter=0.35)

        if len(C_inner)>=2 and len(C_outer)>=2:
            pygame.draw.polygon(screen, BAND_FILL, to_screen(C_inner + list(reversed(C_outer))))
        if len(C_path)>=2:
            pygame.draw.lines(screen, BANK_COLOR, False, to_screen(C_path), road_w)

        # Decorations (lighter than river)
        def _rock_poly(center, size, rng2):
            k = rng2.randint(5,8); verts=[]; ang0=rng2.uniform(0, math.tau)
            for j in range(k):
                ang = ang0 + (j/k)*math.tau + rng2.uniform(-0.2, 0.2)
                rad = size*(0.6 + rng2.uniform(0.0, 0.6))
                px = center[0] + math.cos(ang)*rad
                py = center[1] + math.sin(ang)*rad
                verts.append(world_to_screen((px,py)))
            return verts
        def _grass_blob(center, size, rng3):
            k = rng3.randint(6,10); verts=[]; ang0=rng3.uniform(0, math.tau)
            for j in range(k):
                ang = ang0 + (j/k)*math.tau + rng3.uniform(-0.25,0.25)
                rad = size*(0.55 + rng3.uniform(0.0,0.65))
                px = center[0] + math.cos(ang)*rad
                py = center[1] + math.sin(ang)*rad
                verts.append(world_to_screen((px,py)))
            return verts

        def scatter_rocks_on_inside(polyline, inward_sign, density=0.16, seed=401, off_min=0.2, off_max=2.0):
            rng2 = random.Random(seed)
            stepW = 8.0
            for i in range(1, len(polyline)):
                ax,ay = polyline[i-1]; bx,by = polyline[i]
                vx,vy=bx-ax, by-ay; L=(vx*vx+vy*vy)**0.5 or 1.0
                nx,ny=vy/L, -vx/L
                seg_len=((bx-ax)**2+(by-ay)**2)**0.5
                n=max(1,int(seg_len/stepW))
                for k in range(n):
                    if rng2.random() > density: continue
                    t = (k + rng2.random()) / n
                    x = ax + (bx-ax)*t; y = ay + (by-ay)*t
                    off = rng2.uniform(off_min, off_max)
                    cx = x + nx * off * inward_sign
                    cy = y + ny * off * inward_sign
                    # Skip cement & water
                    if self.harbor_cement:
                        inside_cement=False
                        for _cp in self.harbor_cement:
                            if point_in_poly((cx,cy), _cp): inside_cement=True; break
                        if inside_cement: continue
                    if point_in_poly((cx,cy), self.sea_poly): continue
                    R = _zoom_px(rng2.uniform(2.5, 6.5))
                    poly = _rock_poly((cx,cy), R, rng2)
                    pygame.draw.polygon(screen, (110,120,120), poly)

        def scatter_grass_blobs(polyline, inward_sign, density=0.10, seed=403, off_min=0.2, off_max=2.5):
            rng3 = random.Random(seed)
            stepW = 9.0
            for i in range(1,len(polyline)):
                ax,ay = polyline[i-1]; bx,by = polyline[i]
                vx,vy=bx-ax, by-ay; L=(vx*vx+vy*vy)**0.5 or 1.0
                nx,ny=vy/L, -vx/L
                seg_len=((bx-ax)**2+(by-ay)**2)**0.5
                n=max(1,int(seg_len/stepW))
                for k in range(n):
                    if rng3.random() > density: continue
                    t = (k + rng3.random()) / n
                    x = ax + (bx-ax)*t; y = ay + (by-ay)*t
                    off = rng3.uniform(off_min, off_max)
                    cx = x + nx * off * inward_sign
                    cy = y + ny * off * inward_sign
                    if self.harbor_cement:
                        inside_cement=False
                        for _cp in self.harbor_cement:
                            if point_in_poly((cx,cy), _cp): inside_cement=True; break
                        if inside_cement: continue
                    if point_in_poly((cx,cy), self.sea_poly): continue
                    R = _zoom_px(rng3.uniform(3.0, 9.0))
                    pygame.draw.polygon(screen, (116, 158, 108), _grass_blob((cx,cy), R, rng3))

        def scatter_gap_fill(samples, outward_sign, rock_density=0.16, grass_density=0.12):
            rng2 = random.Random(4242)
            stepW = 7.0
            for i in range(1,len(samples)):
                ax,ay=samples[i-1]; bx,by=samples[i]
                vx,vy=bx-ax, by-ay; L=(vx*vx+vy*vy)**0.5 or 1.0
                nx,ny=vy/L,-vx/L
                seg_len=((bx-ax)**2+(by-ay)**2)**0.5
                n=max(1,int(seg_len/stepW))
                for k in range(n):
                    t=(k + rng2.random())/n
                    x=ax+(bx-ax)*t; y=ay+(by-ay)*t
                    band_min = (road_w * 0.25) + 0.3
                    band_max = band_min + 7.5
                    off = rng2.uniform(band_min, band_max)
                    lat = rng2.uniform(-4.0, 4.0)
                    cx = x + nx * off * outward_sign + (-ny) * lat
                    cy = y + ny * off * outward_sign + ( nx) * lat
                    if self.harbor_cement:
                        inside_cement=False
                        for _cp in self.harbor_cement:
                            if point_in_poly((cx,cy), _cp): inside_cement=True; break
                        if inside_cement: continue
                    if point_in_poly((cx,cy), self.sea_poly):
                        continue
                    if False:
                        sz = _zoom_px(rng2.uniform(2.5, 6.5))
                        poly = _rock_poly((cx,cy), sz, rng2)
                        pygame.draw.polygon(screen, (110,120,120), poly)
                    elif rng2.random() < grass_density:
                        szg = _zoom_px(rng2.uniform(3.0, 9.0))
                        pygame.draw.polygon(screen, (116, 158, 108), _grass_blob((cx,cy), szg, rng2))
        # rocks disabled

        # rocks disabled

        # scatter_grass_blobs(C_mid, inward_sign=sign, density=0.10, seed=403)
        # scatter_grass_blobs(C_mid, inward_sign=sign, density=0.08, seed=404, off_min=0.05, off_max=1.0)
        scatter_gap_fill(C_mid, outward_sign=sign, rock_density=0.16, grass_density=0.12)

        # Draw cement apron ABOVE patch+decor
        if self.harbor_cement:
            import pygame as _pg
            for cpoly in self.harbor_cement:
                _pg.draw.polygon(screen, (180,180,180), [world_to_screen(p) for p in cpoly])
                _pg.draw.polygon(screen, (130,130,130), [world_to_screen(p) for p in cpoly], max(1, int(2*cam_zoom)))

        # Harbor overlay
        self._draw_harbor(screen, world_to_screen, cam_zoom)

    # ---------------- Harbor ----------------
    def generate_harbor(self):
        """Create harbor geometry along the coast (breakwaters, piers, cement apron)."""
        print("[Sea] generate_harbor")
        if not self.sea_poly or not self.coast_path:
            return
        import math, random
        w, h = self.MAP_WIDTH, self.HEIGHT
        coast = self.coast_path
        rng = random.Random(random.randint(0, 1_000_000))
        if len(coast) < 4:
            return

        # anchor along mid 60% of coast
        i0 = int(len(coast) * rng.uniform(0.2, 0.8))
        i1 = min(len(coast)-1, i0+1)
        A = coast[i0]; B = coast[i1]
        vx, vy = (B[0]-A[0], B[1]-A[1])
        L = (vx*vx + vy*vy)**0.5 or 1.0
        nx, ny = (vy/L, -vx/L)  # left normal

        # sea vs land
        dtest = 40.0
        left  = (A[0] + nx*dtest, A[1] + ny*dtest)
        right = (A[0] - nx*dtest, A[1] - ny*dtest)
        left_in  = point_in_poly(left,  self.sea_poly)
        right_in = point_in_poly(right, self.sea_poly)
        sea_sign  = +1 if left_in else -1 if right_in else +1
        land_sign = -sea_sign
        n_sea_x, n_sea_y = nx*sea_sign, ny*sea_sign

        def poly_rect(center, dirx, diry, length, width):
            hx, hy = dirx*length*0.5, diry*length*0.5
            wx, wy = -diry*width*0.5, dirx*width*0.5
            cx, cy = center
            return [
                (cx - hx - wx, cy - hy - wy),
                (cx - hx + wx, cy - hy + wy),
                (cx + hx + wx, cy + hy + wy),
                (cx + hx - wx, cy + hy - wy),
            ]

        # reset
        self.harbor_breakwaters = []
        self.harbor_piers = []
        self.harbor_cement = []

        
        # Breakwater (single straight arm)
        arm_len = max(120.0, min(260.0, 0.28*max(w,h)))
        arm_w   = 16.0
        dirx, diry = n_sea_x, n_sea_y  # into sea
        arm_center = (A[0] + dirx*(arm_len*0.5), A[1] + diry*(arm_len*0.5))
        arm_poly = poly_rect(arm_center, dirx, diry, arm_len, arm_w)

        # Store exactly one breakwater polygon
        self.harbor_breakwaters = [arm_poly]


        # Cement apron (land side band near anchor)
        seg_lo = max(0, i0 - 18); seg_hi = min(len(coast)-1, i0 + 18)
        seg = coast[seg_lo:seg_hi+1]

        def _resample_by_count(polyline, count):
            pts=list(polyline); count=max(2,int(count))
            if len(pts)<=2: return pts
            L=[0.0]
            for i in range(1,len(pts)):
                ax,ay=pts[i-1]; bx,by=pts[i]
                L.append(L[-1]+((bx-ax)**2+(by-ay)**2)**0.5)
            total=L[-1] or 1.0; step=total/(count-1)
            out=[pts[0]]; tgt=step; j=1
            while len(out)<count-1 and j<len(pts):
                while j<len(pts) and L[j]<tgt: j+=1
                if j>=len(pts): break
                a=pts[j-1]; b=pts[j]; la=L[j-1]; lb=L[j]
                t=(tgt-la)/max(1e-9, lb-la)
                out.append((a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t)); tgt+=step
            out.append(pts[-1]); return out

        seg = _resample_by_count(seg, 30)

        def _offset_polyline(samples, outward_sign, off):
            out=[]
            for i in range(1,len(samples)):
                P=samples[i-1]; Q=samples[i]
                vx,vy=Q[0]-P[0], Q[1]-P[1]; L=(vx*vx+vy*vy)**0.5 or 1.0
                nx,ny=vy/L, -vx/L
                if i==1: out.append((P[0]+nx*off*outward_sign, P[1]+ny*off*outward_sign))
                out.append((Q[0]+nx*off*outward_sign, Q[1]+ny*off*outward_sign))
            return out

        apron_sea  = _offset_polyline(seg, outward_sign=sea_sign,  off= 3.0)
        apron_road = _offset_polyline(seg, outward_sign=land_sign, off=12.0)
        self.harbor_cement = [apron_sea + list(reversed(apron_road))]

        # Piers pointing into sea
        n_piers = rng.randint(5,8)
        for k in range(n_piers):
            t = (k - (n_piers-1)/2.0) * rng.uniform(20.0, 30.0)
            base = (A[0] + (vx/L)*t, A[1] + (vy/L)*t)
            plen = rng.uniform(50.0, 80.0)
            pwid = rng.uniform(6.0, 10.0)
            px, py = n_sea_x, n_sea_y
            pier_center = (base[0] + px*(plen*0.5), base[1] + py*(plen*0.5))
            poly = poly_rect(pier_center, px, py, plen, pwid)
            self.harbor_piers.append(poly)

    def _draw_harbor(self, screen, world_to_screen, cam_zoom):
        import pygame
        if not self.harbor_breakwaters:
            return
        # Breakwaters
        for poly in self.harbor_breakwaters:
            pygame.draw.polygon(screen, (90, 92, 96), [world_to_screen(p) for p in poly])
            pygame.draw.polygon(screen, (60, 62, 66), [world_to_screen(p) for p in poly], max(1, int(2*cam_zoom)))
        # Piers
        for poly in self.harbor_piers:
            pygame.draw.polygon(screen, (185,160,120), [world_to_screen(p) for p in poly])
            pygame.draw.polygon(screen, (140,120, 90), [world_to_screen(p) for p in poly], max(1, int(2*cam_zoom)))

    # ---------------- State ----------------
    def serialize_state(self):
        return {
            "sea_poly": [tuple(p) for p in self.sea_poly] if self.sea_poly else None,
            "coast_path": [tuple(p) for p in self.coast_path] if self.coast_path else None,
            "harbor_breakwaters": self.harbor_breakwaters if self.harbor_breakwaters else None,
            "harbor_piers": self.harbor_piers if self.harbor_piers else None,
            "harbor_cement": self.harbor_cement if self.harbor_cement else None,
        }

    def restore_state(self, st):
        self.sea_poly   = [tuple(p) for p in st.get("sea_poly")] if st.get("sea_poly") else None
        self.coast_path = [tuple(p) for p in st.get("coast_path")] if st.get("coast_path") else None
        self.harbor_breakwaters = st.get("harbor_breakwaters") or []
        self.harbor_piers       = st.get("harbor_piers") or []
        self.harbor_cement      = st.get("harbor_cement") or []
