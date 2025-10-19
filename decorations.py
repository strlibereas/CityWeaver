
import math, random

try:
    import pygame
except Exception:
    pygame = None

class DecorationsSystem:
    def _inside_any_cement(self, x, y):
        for poly in getattr(self, '_cement_polys', []) or []:
            if self._point_in_poly((x,y), poly):
                return True
        return False

    def _inside_map(self, x, y):
        return 0 <= x <= self.MAP_WIDTH and 0 <= y <= self.HEIGHT

    """
    Lightweight decorative objects (grass & rocks) drawn as GPU-blitted sprites
    instead of re-generated polygons every frame. Instances share a few pre-rendered
    textures and are culled to the viewport.
    """
    def __init__(self, map_size):
        self.MAP_WIDTH, self.HEIGHT = map_size
        self.instances = []   # list of dicts: {type, x, y, size, angle}
        self._key = None      # signature of (river_poly, sea_poly)
        self._sprites = {}
        self._cement_polys = []    # {(zoom_bucket, type, size_id): Surface}
        self._zoom_bucket = None

    @staticmethod
    def _signed_area(poly):
        a = 0.0
        for i in range(len(poly)):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
            a += x1*y2 - x2*y1
        return 0.5*a

    @staticmethod
    def _point_in_poly(pt, poly):
        # ray casting
        x,y = pt; inside = False
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
            if ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/( (y2-y1) if (y2!=y1) else 1e-9 ) + x1):
                inside = not inside
        return inside

    @staticmethod
    def _poly_length(poly):
        L=0.0
        for i in range(1,len(poly)):
            ax,ay=poly[i-1]; bx,by=poly[i]
            L += math.hypot(bx-ax, by-ay)
        return L

    def _sample_boundary(self, poly, step=10.0):
        # returns [(x,y, nx,ny)] approx every step along boundary with outward normal
        out = []
        if not poly or len(poly)<2: return out
        # orientation
        area = self._signed_area(poly)
        ccw = area > 0  # CCW positive
        for i in range(len(poly)):
            ax,ay = poly[i-1]; bx,by = poly[i]
            vx,vy = bx-ax, by-ay
            seg_len = math.hypot(vx,vy) or 1.0
            nx,ny = vy/seg_len, -vx/seg_len  # left-hand normal
            # Determine outward by testing a point
            # Left-hand normal points outward for CW polygons; invert if needed
            test_x = (ax+bx)*0.5 + nx*2.0
            test_y = (ay+by)*0.5 + ny*2.0
            if self._point_in_poly((test_x,test_y), poly):
                nx,ny = -nx, -ny
            n = max(1, int(seg_len/step))
            for k in range(n):
                t = (k + 0.5) / n
                x = ax + vx*t; y = ay + vy*t
                out.append((x,y,nx,ny))
        return out

    def _rebuild(self, river_poly, sea_poly, cement_polys=None, rng_seed=12345):
        sea_cement = []
        try:
            from sea import SeaSystem
        except ImportError:
            sea_cement = []
        # If sea instance is available later we'll filter again

        rng = random.Random(rng_seed)
        instances = []

        def add_band_decor(boundary_samples, band_min=1.0, band_max=18.0,
                           rock_density=0.36, grass_density=0.44):
            for (x,y,nx,ny) in boundary_samples:
                # rocks
                if rng.random() < rock_density:
                    off = rng.uniform(band_min, band_max)
                    cx = x + nx*off; cy = y + ny*off
                    size = rng.uniform(4.0, 10.0)
                    ang  = rng.uniform(0, math.tau)
                    if not self._inside_any_cement(cx, cy) and self._inside_map(cx, cy):
                        instances.append({"type":"rock","x":cx,"y":cy,"size":size,"angle":ang})
                # grass
                if rng.random() < grass_density:
                    off = rng.uniform(band_min, band_max)
                    cx = x + nx*off; cy = y + ny*off
                    size = rng.uniform(6.0, 16.0)
                    ang  = rng.uniform(0, math.tau)
                    if not self._inside_any_cement(cx, cy) and self._inside_map(cx, cy):
                        instances.append({"type":"grass","x":cx,"y":cy,"size":size,"angle":ang})
            


        # River banks: decorate OUTSIDE the river polygon
        if river_poly:
            samples = self._sample_boundary(river_poly, step=8.0)
            # Filter samples that are outside the river
            out_samples = []
            for (x,y,nx,ny) in samples:
                # Point just outside
                ox,oy = x+nx*2.0, y+ny*2.0
                if not self._point_in_poly((ox,oy), river_poly):
                    out_samples.append((x,y,nx,ny))
            add_band_decor(out_samples, band_min=1.0, band_max=20.0,
                           rock_density=0.40, grass_density=0.52)

        # Sea coast: decorate OUTSIDE the sea polygon
        if sea_poly:
            samples = self._sample_boundary(sea_poly, step=9.0)
            out_samples = []
            for (x,y,nx,ny) in samples:
                ox,oy = x+nx*2.0, y+ny*2.0
                if not self._point_in_poly((ox,oy), sea_poly):
                    out_samples.append((x,y,nx,ny))
            add_band_decor(out_samples, band_min=2.0, band_max=28.0,
                           rock_density=0.30, grass_density=0.40)

        self.instances = [i for i in instances if self._inside_map(i["x"], i["y"])]

    def ensure_synced(self, water, sea):
        river_poly = getattr(water, "river_poly", None)
        sea_poly   = getattr(sea, "sea_poly",   None)
        cement     = getattr(sea, "harbor_cement", []) or []

        def _norm_point(pt):
            x, y = pt
            return (round(x, 2), round(y, 2))

        river_key = tuple(_norm_point(pt) for pt in river_poly) if river_poly else None
        sea_key   = tuple(_norm_point(pt) for pt in sea_poly)   if sea_poly   else None
        cement_key = tuple(tuple(_norm_point(pt) for pt in poly) for poly in cement)

        key = (river_key, sea_key, cement_key)
        if key != self._key:
            self._cement_polys = cement
            self._rebuild(river_poly, sea_poly)
            self._key = key





    # --- Rendering ---

    def _zoom_bucket_for(self, cam_zoom: float) -> int:
        # bucket zoom to limit re-renders of sprites
        # 0.50-0.74 -> 50, 0.75-1.24 -> 100, etc.
        return int(round(cam_zoom * 100))

    def _get_sprite(self, typ: str, size_px: int, zoom_bucket: int):
        if pygame is None:
            return None
        key = (zoom_bucket, typ, size_px)
        srf = self._sprites.get(key)
        if srf is not None:
            return srf
        # Build sprite
        pad = 2
        w = h = max(8, size_px) + pad*2
        surf = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
        cx, cy = w/2, h/2
        rnd = random.Random( (hash((typ,size_px,zoom_bucket)) & 0xffffffff) )
        import math
        if typ == "rock":
            k = rnd.randint(5,8)
            ang0 = rnd.uniform(0, math.tau)
            pts = []
            for j in range(k):
                ang = ang0 + (j/k)*math.tau + rnd.uniform(-0.2,0.2)
                rad = (size_px/2) * (0.6 + rnd.uniform(0.0,0.6))
                px = cx + math.cos(ang)*rad
                py = cy + math.sin(ang)*rad
                pts.append((px,py))
            c1 = (112,112,112); c2 = (90,90,90)
            pygame.draw.polygon(surf, c1, pts)
            pygame.draw.polygon(surf, c2, pts, 1)
        else:  # grass
            k = rnd.randint(5,7)
            ang0 = rnd.uniform(0, math.tau)
            pts = []
            for j in range(k):
                ang = ang0 + (j/k)*math.tau + rnd.uniform(-0.3,0.3)
                rad = (size_px/2) * (0.8 + rnd.uniform(-0.1,0.3))
                px = cx + math.cos(ang)*rad
                py = cy + math.sin(ang)*rad
                pts.append((px,py))
            g = max(0, min(255, int(150 + rnd.uniform(-25, 25))))
            color = (86, g, 87)
            pygame.draw.polygon(surf, color, pts)
        self._sprites[key] = surf
        return surf

    def draw(self, screen, world_to_screen, cam_zoom):
        if pygame is None or not self.instances:
            return
        zoom_bucket = self._zoom_bucket_for(cam_zoom)
        self._zoom_bucket = zoom_bucket

        # Viewport culling in world space
        sw, sh = screen.get_size()
        # compute approximate world-rect visible (assumes world_to_screen is affine-like)
        # we'll inverse by sampling (0,0) and (sw,sh) with an approximate local scale
        # The caller provides cam_zoom; world units per pixel is 1/cam_zoom
        # So inflate the rect slightly
        # We can't easily recover camera offset here; rely on world_to_screen by checking each instance's screen pos.
        for inst in self.instances:
            sx, sy = world_to_screen((inst["x"], inst["y"]))
            if sx < -50 or sy < -50 or sx > sw+50 or sy > sh+50:
                continue
            # convert desired world size to pixel size
            size_px = max(6, int(inst["size"] * cam_zoom))
            # Quantize to a few buckets to enable instancing
            if size_px < 9: size_id = 8
            elif size_px < 13: size_id = 12
            elif size_px < 18: size_id = 16
            else: size_id = 20
            surf = self._get_sprite(inst["type"], size_id, zoom_bucket)
            if surf is None: continue
            rx = sx - surf.get_width()/2
            ry = sy - surf.get_height()/2
            screen.blit(surf, (rx, ry))
