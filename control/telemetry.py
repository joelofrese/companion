"""Vehicle telemetry shared by the brain and body links."""

from dataclasses import dataclass
from typing import Optional

from control.velocity import VelocityCommand


@dataclass(frozen=True)
class Telemetry:
    """The vehicle information the brain can use."""

    obstacle_distance_m: Optional[float] = None
    last_command: Optional[VelocityCommand] = None
    forward_velocity_m_s: Optional[float] = None
    right_velocity_m_s: Optional[float] = None
    down_velocity_m_s: Optional[float] = None
    heading_rad: Optional[float] = None
