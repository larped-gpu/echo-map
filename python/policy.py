"""Rule-based adaptive chirp policy: jointly selects pose and chirp mode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    APPROACH_DISTANCE_M,
    CHIRP_MODES,
    GLASS_KURTOSIS_THRESHOLD,
    MAP_RESOLUTION_M,
    MATERIAL_CONFIDENCE_THRESHOLD,
    MATERIAL_STOP_THRESHOLD,
)
from mapping import EchoMap


@dataclass
class PolicyAction:
    """Output of the adaptive chirp policy."""

    target_x: float
    target_y: float
    target_heading: float
    chirp_mode: str
    servo_angle_deg: float = 0.0
    reason: str = ""


def select_action(
    echomap: EchoMap,
    last_glass_hint: bool = False,
    last_glass_conf: float = 0.0,
) -> PolicyAction | None:
    """
    Rule-based policy from proposal section 04.

    Priority:
    1. Glass probe if glass signature detected
    2. Material chirp if wall confidence < threshold
    3. Geometry chirp toward largest frontier
    4. Stop if map converged
    """
    # Stop condition
    if echomap.map_converged(conf_threshold=MATERIAL_STOP_THRESHOLD):
        return None

    # Glass probe: sharp early peak + low spectral spread
    if last_glass_hint and last_glass_conf > 0.5:
        wall = echomap.lowest_confidence_wall()
        if wall:
            gx, gy, _ = wall
            tx = gx * MAP_RESOLUTION_M
            ty = gy * MAP_RESOLUTION_M
            return PolicyAction(
                target_x=tx,
                target_y=ty,
                target_heading=echomap.robot_heading,
                chirp_mode="GLASS_PROBE",
                servo_angle_deg=37.5,  # 30-45 deg oblique
                reason="glass signature detected, probing obliquely",
            )

    # Material confidence low: move closer and emit MATERIAL chirp
    wall = echomap.lowest_confidence_wall()
    if wall is not None:
        gx, gy, conf = wall
        if conf < MATERIAL_CONFIDENCE_THRESHOLD:
            tx = gx * MAP_RESOLUTION_M
            ty = gy * MAP_RESOLUTION_M
            # Move closer: interpolate toward wall
            dx = tx - echomap.robot_x
            dy = ty - echomap.robot_y
            dist = np.hypot(dx, dy)
            if dist > APPROACH_DISTANCE_M:
                scale = (dist - APPROACH_DISTANCE_M) / dist
                tx = echomap.robot_x + dx * scale
                ty = echomap.robot_y + dy * scale
            heading = float(np.degrees(np.arctan2(dy, dx)))
            return PolicyAction(
                target_x=tx,
                target_y=ty,
                target_heading=heading,
                chirp_mode="MATERIAL",
                reason=f"material confidence {conf:.2f} < {MATERIAL_CONFIDENCE_THRESHOLD}",
            )

    # Unmapped frontier: move toward it with GEOMETRY chirp
    frontiers = echomap.frontier_cells()
    if frontiers:
        # Pick frontier closest to robot
        best = min(
            frontiers,
            key=lambda f: np.hypot(
                f[0] * MAP_RESOLUTION_M - echomap.robot_x,
                f[1] * MAP_RESOLUTION_M - echomap.robot_y,
            ),
        )
        tx = best[0] * MAP_RESOLUTION_M
        ty = best[1] * MAP_RESOLUTION_M
        dx = tx - echomap.robot_x
        dy = ty - echomap.robot_y
        heading = float(np.degrees(np.arctan2(dy, dx)))
        return PolicyAction(
            target_x=tx,
            target_y=ty,
            target_heading=heading,
            chirp_mode="GEOMETRY",
            reason="exploring unmapped frontier",
        )

    # Fallback: sweep geometry chirp in current heading
    return PolicyAction(
        target_x=echomap.robot_x,
        target_y=echomap.robot_y,
        target_heading=echomap.robot_heading,
        chirp_mode="GEOMETRY",
        reason="no frontiers, scanning current heading",
    )
