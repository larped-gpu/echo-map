"""Occupancy + material grid mapping, wall extraction, map rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import (
    MAP_CELLS,
    MAP_RESOLUTION_M,
    MAP_SIZE_M,
    MATERIALS,
    MATERIAL_TO_IDX,
)


@dataclass
class EchoMap:
    """Two-layer map: occupancy (free/occupied/unknown) + material belief."""

    resolution_m: float = MAP_RESOLUTION_M
    size_m: float = MAP_SIZE_M
    cells: int = MAP_CELLS

    # -1 = unknown, 0 = free, 1 = occupied
    occupancy: np.ndarray = field(default_factory=lambda: np.full((MAP_CELLS, MAP_CELLS), -1, dtype=np.int8))

    # Per-cell material class index (-1 = unknown)
    material: np.ndarray = field(default_factory=lambda: np.full((MAP_CELLS, MAP_CELLS), -1, dtype=np.int8))

    # Per-cell material confidence [0, 1]
    material_conf: np.ndarray = field(default_factory=lambda: np.zeros((MAP_CELLS, MAP_CELLS), dtype=np.float32))

    # Robot pose (meters from origin, heading degrees)
    robot_x: float = MAP_SIZE_M / 2
    robot_y: float = MAP_SIZE_M / 2
    robot_heading: float = 0.0

    def world_to_grid(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Convert world coordinates (m) to grid indices."""
        gx = int(x_m / self.resolution_m)
        gy = int(y_m / self.resolution_m)
        gx = np.clip(gx, 0, self.cells - 1)
        gy = np.clip(gy, 0, self.cells - 1)
        return gx, gy

    def update_occupancy(
        self,
        range_m: float,
        bearing_deg: float,
        occupied: bool = True,
    ) -> None:
        """
        Mark cells along a ray from robot pose to detected range.

        Free cells along ray, occupied at endpoint.
        """
        heading_rad = np.radians(self.robot_heading + bearing_deg)
        n_steps = int(range_m / self.resolution_m)

        for step in range(1, n_steps):
            dist = step * self.resolution_m
            wx = self.robot_x + dist * np.cos(heading_rad)
            wy = self.robot_y + dist * np.sin(heading_rad)
            gx, gy = self.world_to_grid(wx, wy)
            if self.occupancy[gy, gx] == -1:
                self.occupancy[gy, gx] = 0  # free

        # Endpoint
        wx = self.robot_x + range_m * np.cos(heading_rad)
        wy = self.robot_y + range_m * np.sin(heading_rad)
        gx, gy = self.world_to_grid(wx, wy)
        if occupied:
            self.occupancy[gy, gx] = 1

    def update_material(
        self,
        range_m: float,
        bearing_deg: float,
        material_idx: int,
        confidence: float,
    ) -> None:
        """Bayesian-style update on material belief at wall endpoint."""
        heading_rad = np.radians(self.robot_heading + bearing_deg)
        wx = self.robot_x + range_m * np.cos(heading_rad)
        wy = self.robot_y + range_m * np.sin(heading_rad)
        gx, gy = self.world_to_grid(wx, wy)

        if confidence > self.material_conf[gy, gx]:
            self.material[gy, gx] = material_idx
            self.material_conf[gy, gx] = confidence

    def frontier_cells(self) -> list[tuple[int, int]]:
        """Find unknown cells adjacent to free space (exploration targets)."""
        frontiers = []
        for gy in range(1, self.cells - 1):
            for gx in range(1, self.cells - 1):
                if self.occupancy[gy, gx] != -1:
                    continue
                neighbors = self.occupancy[gy - 1 : gy + 2, gx - 1 : gx + 2]
                if np.any(neighbors == 0):
                    frontiers.append((gx, gy))
        return frontiers

    def lowest_confidence_wall(self) -> tuple[int, int, float] | None:
        """Find occupied cell with lowest material confidence."""
        mask = self.occupancy == 1
        if not np.any(mask):
            return None
        confs = np.where(mask, self.material_conf, 2.0)
        min_idx = np.unravel_index(np.argmin(confs), confs.shape)
        gy, gx = min_idx
        return gx, gy, float(self.material_conf[gy, gx])

    def map_converged(self, conf_threshold: float = 0.85) -> bool:
        """True if no frontiers and all walls have high material confidence."""
        if self.frontier_cells():
            return False
        mask = self.occupancy == 1
        if not np.any(mask):
            return False
        wall_confs = self.material_conf[mask]
        return bool(np.all(wall_confs >= conf_threshold))

    def to_rgb_image(self) -> np.ndarray:
        """
        Render map as RGB image for visualization.

        Colors: unknown=gray, free=white, walls=color-coded by material.
        """
        material_colors = {
            0: (200, 180, 140),  # drywall - tan
            1: (139, 90, 43),    # wood - brown
            2: (135, 206, 250),  # glass - light blue
            3: (192, 192, 192),  # metal - silver
            4: (160, 120, 80),   # carpet - dark tan
            5: (128, 128, 128),  # concrete - gray
        }

        img = np.full((self.cells, self.cells, 3), 200, dtype=np.uint8)  # unknown = gray

        free_mask = self.occupancy == 0
        img[free_mask] = (255, 255, 255)

        wall_mask = self.occupancy == 1
        for mat_idx, color in material_colors.items():
            mat_mask = wall_mask & (self.material == mat_idx)
            img[mat_mask] = color

        # Unknown-material walls = dark red
        unknown_wall = wall_mask & (self.material == -1)
        img[unknown_wall] = (180, 60, 60)

        return img

    def save_png(self, path: str) -> None:
        """Save map as PNG file."""
        try:
            from PIL import Image

            img = self.to_rgb_image()
            Image.fromarray(img).save(path)
        except ImportError:
            import matplotlib.pyplot as plt

            img = self.to_rgb_image()
            plt.imsave(path, img)
