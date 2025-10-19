
QT_MAX_OBJECTS = 32
QT_MAX_LEVELS  = 10

class Quadtree:
    def __init__(self, bounds, depth=0):
        self.x,self.y,self.w,self.h = bounds
        self.depth = depth
        self.items = []
        self.divided = False
        self.NW=self.NE=self.SW=self.SE=None

    def _intersects(self, a, b):
        ax,ay,aw,ah = a; bx,by,bw,bh = b
        return not (ax+aw<bx or bx+bw<ax or ay+ah<by or by+bh<ay)

    def _subdivide(self):
        hx,hy = self.w/2, self.h/2; x,y = self.x, self.y; d = self.depth+1
        self.NW = Quadtree((x,      y,      hx,hy), d)
        self.NE = Quadtree((x+hx,   y,      hx,hy), d)
        self.SW = Quadtree((x,      y+hy,   hx,hy), d)
        self.SE = Quadtree((x+hx,   y+hy,   hx,hy), d)
        self.divided = True

    def _child(self, rect):
        cx = rect[0] + rect[2]/2; cy = rect[1] + rect[3]/2
        left = cx < self.x + self.w/2; top = cy < self.y + self.h/2
        return self.NW if top and left else self.NE if top and not left else self.SW if not top and left else self.SE

    def insert(self, rect, payload):
        if not self._intersects(rect, (self.x,self.y,self.w,self.h)): return False
        if len(self.items) < QT_MAX_OBJECTS or self.depth >= QT_MAX_LEVELS:
            self.items.append((rect, payload)); return True
        if not self.divided: self._subdivide()
        return self._child(rect).insert(rect, payload)

    def query(self, rect, out):
        if not self._intersects(rect, (self.x,self.y,self.w,self.h)): return
        for r,p in self.items:
            if self._intersects(r, rect): out.append(p)
        if self.divided:
            self.NW.query(rect, out); self.NE.query(rect, out)
            self.SW.query(rect, out); self.SE.query(rect, out)
