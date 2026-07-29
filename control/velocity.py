"""The velocity command shared by all control layers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VelocityCommand:
    """North, east, down, and yaw setpoints."""

    north_m_s: float = 0.0
    east_m_s: float = 0.0
    down_m_s: float = 0.0
    yaw_deg: float = 0.0
