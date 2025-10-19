"""
railway.py
================

This module defines a very lightweight ``RailwaySystem`` that can be used
within the existing city weaver to draw a stylised railway across the
city.  The railway is represented as a list of world‑space points forming a
polyline.  A small station and train are placed on the line and drawn at
the appropriate zoom level.

The API is intentionally minimal: call ``generate()`` to build a new
railway path and ``draw()`` to render it on the pygame surface.  Repeated
calls to ``generate()`` will toggle the railway on and off; a call when
a railway already exists will clear it.

The path generation uses a simple cubic Bézier construction with two
intermediate control points to create a gentle curve.  The control points
are chosen based on the map size and a small random vertical jitter so
that each generation feels slightly different.
"""

from __future__ import annotations

import random
import math
from typing import List, Tuple, Optional

try:
    import pygame
except ImportError:
    # pygame will be available in the main program context.  If not
    # available at import time we fall back to None and skip drawing.
    pygame = None


class RailwaySystem:
    """A system for generating and drawing a simple railway line.

    Parameters
    ----------
    map_size : tuple[int, int]
        The width and height of the map in pixels.  The railway will
        traverse the full width of the map.

    seed : Optional[int]
        Optional random seed to make railway generation deterministic
        across runs.  If ``None`` a new random seed is used.
    """

    def __init__(self, map_size: Tuple[int, int], seed: Optional[int] = None) -> None:
        self.MAP_WIDTH, self.HEIGHT = map_size
        # Path is stored as a list of world‑space (x,y) tuples
        self.path: List[Tuple[float, float]] = []
        self.station_pos: Optional[Tuple[float, float]] = None
        self.train_pos: Optional[Tuple[float, float]] = None
        self._rng = random.Random(seed)

    def clear(self) -> None:
        """Remove any existing railway path, train and station."""
        self.path.clear()
        self.station_pos = None
        self.train_pos = None

    def _sample_bezier(self, p0, p1, p2, p3, n=80) -> List[Tuple[float, float]]:
        """Return ``n`` samples along a cubic Bézier curve defined by the
        points ``p0``..``p3``.  This helper uses a simple parameter sweep.
        """
        pts = []
        for i in range(n + 1):
            t = i / n
            mt = 1.0 - t
            # De Casteljau's algorithm for cubic Bézier
            x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
            y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
            pts.append((x, y))
        return pts

    def generate(self) -> None:
        """Generate a railway path.  If a railway already exists it will
        be cleared.  The new path starts off the left edge of the map
        and ends off the right edge, with a gentle vertical curve in
        between.  A station is placed roughly in the middle and a train
        toward the first quarter of the route.
        """
        # Toggle behaviour: clear if already present
        if self.path:
            self.clear()
            return

        # Choose a random base y position near the vertical centre of the map
        base_y = self.HEIGHT * 0.5 + self._rng.uniform(-self.HEIGHT * 0.1, self.HEIGHT * 0.1)
        # Control points for the Bézier: a gentle S‑curve across the map
        start = (-0.2 * self.MAP_WIDTH, base_y + self._rng.uniform(-30, 30))
        end   = (1.2 * self.MAP_WIDTH, base_y + self._rng.uniform(-30, 30))
        cp1_y_offset = self.HEIGHT * 0.15 * self._rng.uniform(-1.0, 1.0)
        cp2_y_offset = self.HEIGHT * 0.15 * self._rng.uniform(-1.0, 1.0)
        cp1 = (self.MAP_WIDTH * 0.3, base_y + cp1_y_offset)
        cp2 = (self.MAP_WIDTH * 0.7, base_y + cp2_y_offset)
        # Sample the curve
        self.path = self._sample_bezier(start, cp1, cp2, end, n=150)
        # Pick station and train positions along the path
        if self.path:
            self.station_pos = self.path[int(len(self.path) * 0.5)]
            self.train_pos   = self.path[int(len(self.path) * 0.25)]

    def draw(self, screen, world_to_screen, cam_zoom: float) -> None:
        """Draw the railway, station and train to the given ``screen``.

        Parameters
        ----------
        screen : ``pygame.Surface``
            The destination surface on which to render.
        world_to_screen : callable
            A function mapping world‑space (x,y) tuples to screen coordinates.
        cam_zoom : float
            Current camera zoom factor.  Used to scale the thickness of
            drawn elements.
        """
        if not pygame or not self.path:
            return
        # Draw the rail line as a yellow polyline.  We draw individual
        # segments for simplicity.  Thickness scales with zoom.
        color = (240, 200, 30)  # warm yellow
        width = max(2, int(4 * cam_zoom))
        for a, b in zip(self.path[:-1], self.path[1:]):
            ax, ay = world_to_screen(a)
            bx, by = world_to_screen(b)
            pygame.draw.line(screen, color, (ax, ay), (bx, by), width)
        # Draw the station as a small brownish block
        if self.station_pos:
            sx, sy = world_to_screen(self.station_pos)
            size = max(4, int(14 * cam_zoom))
            rect = pygame.Rect(int(sx - size / 2), int(sy - size / 2), size, size)
            pygame.draw.rect(screen, (180, 150, 90), rect)
            pygame.draw.rect(screen, (90, 60, 30), rect, 1)
        # Draw the train as a small red rectangle aligned along the track
        if self.train_pos:
            # Determine direction by looking ahead on the path
            idx = self.path.index(self.train_pos)
            next_idx = min(len(self.path) - 1, idx + 5)
            tx, ty = self.train_pos
            nx, ny = self.path[next_idx]
            # Angle of the path segment
            angle = math.atan2(ny - ty, nx - tx)
            # Train dimensions scale with zoom
            length = max(6, int(18 * cam_zoom))
            height = max(4, int(10 * cam_zoom))
            # Local corners of the rectangle centred at origin
            half_l = length / 2.0
            half_h = height / 2.0
            local_pts = [(-half_l, -half_h), (half_l, -half_h), (half_l, half_h), (-half_l, half_h)]
            # Rotate and translate
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            screen_pts = []
            for (lx, ly) in local_pts:
                x = tx + lx * cos_a - ly * sin_a
                y = ty + lx * sin_a + ly * cos_a
                screen_pts.append(world_to_screen((x, y)))
            pygame.draw.polygon(screen, (200, 40, 40), screen_pts)
            pygame.draw.polygon(screen, (80, 20, 20), screen_pts, 1)
