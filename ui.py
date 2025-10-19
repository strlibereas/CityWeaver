
import pygame

class Slider:
    def __init__(self, label, min_val, max_val, value, step=0.01, fmt="{:.2f}", on_change=None):
        self.label=label; self.min=float(min_val); self.max=float(max_val)
        self.value=float(value); self.step=float(step); self.fmt=fmt; self.on_change=on_change
        self.rect = pygame.Rect(0,0,0,0); self.dragging=False

    def set_value(self, v):
        v=max(self.min, min(self.max, float(v)))
        if abs(v-self.value) >= 1e-9:
            self.value=v
            if self.on_change: self.on_change(self.value)

    def draw(self, screen, font, x, y, width):
        label = font.render(f"{self.label}: {self.fmt.format(self.value)}", True, (30,30,30))
        screen.blit(label, (x,y))
        track_y = int(y + label.get_height() + 6); track_h=6
        track = pygame.Rect(x, track_y, width, track_h)
        pygame.draw.rect(screen, (220,220,220), track, border_radius=3)
        t = 0.0 if self.max==self.min else (self.value-self.min)/(self.max-self.min)
        hx = int(x + t*width); hr=7
        pygame.draw.circle(screen, (80,80,80), (hx, track_y+track_h//2), hr)
        self.rect = pygame.Rect(x, y, width, hr*2 + label.get_height() + 6)
        return self.rect.bottom + 10

    def handle_event(self, event, area_x, area_w):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button==1:
            if self.rect.collidepoint(*event.pos):
                self.dragging=True; return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button==1:
            self.dragging=False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mx,_ = event.pos; t = (mx-area_x)/max(1,area_w); t=max(0,min(1,t))
            raw = self.min + t*(self.max-self.min)
            steps = round((raw-self.min)/self.step); val = self.min + steps*self.step
            self.set_value(val); return True
        return False

class SidebarUI:
    def __init__(self, rect):
        self.rect = pygame.Rect(rect); self.controls=[]; self.font=None
        self.scroll=0; self.inner_h=0
    def set_font(self, font): self.font=font
    def add_slider(self, s): self.controls.append(s)
    def draw_bg(self, screen):
        pygame.draw.rect(screen, (250,250,250), self.rect)
        pygame.draw.line(screen, (200,200,200), (self.rect.left,0),(self.rect.left,self.rect.bottom),2)
    def draw(self, screen):
        if not self.font: return
        x = self.rect.left + 20; y = self.rect.top + 20 - self.scroll; w = self.rect.width - 40
        self.inner_h = 0
        clip_prev = screen.get_clip(); screen.set_clip(self.rect)
        for ctl in self.controls:
            y = ctl.draw(screen, self.font, x, y, w)
            self.inner_h = max(self.inner_h, y - (self.rect.top - self.scroll))
        screen.set_clip(clip_prev)
    def handle_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(*pygame.mouse.get_pos()):
                self.scroll = max(0, min(max(0, self.inner_h - self.rect.height), self.scroll - event.y*20))
                return True
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            if not self.rect.collidepoint(*pygame.mouse.get_pos()): return False
        consumed=False; x=self.rect.left+20; w=self.rect.width-40
        for ctl in self.controls:
            consumed = ctl.handle_event(event, x, w) or consumed
        return consumed
