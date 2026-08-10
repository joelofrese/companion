"""The body-frame movement command shared by all control layers."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame velocity and yaw-rate setpoints."""

    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0


def body_to_ned(command: VelocityCommand, heading_rad: float):
    """Convert one body-frame command to PX4's local NED frame."""

    cosine = math.cos(heading_rad)
    sine = math.sin(heading_rad)
    return (
        command.forward_m_s * cosine - command.right_m_s * sine,
        command.forward_m_s * sine + command.right_m_s * cosine,
        command.down_m_s,
    )


def ned_to_body(north_m_s, east_m_s, down_m_s, heading_rad: float):
    """Convert one PX4 local NED velocity to body coordinates."""

    cosine = math.cos(heading_rad)
    sine = math.sin(heading_rad)
    return (
        north_m_s * cosine + east_m_s * sine,
        -north_m_s * sine + east_m_s * cosine,
        down_m_s,
    )
