"""Velocity command shared by reactive control and vehicle adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityCommand:
    """North-east-down velocity and yaw in the PX4/MAVSDK convention."""

    north_m_s: float = 0.0
    east_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_deg: float = 0.0
