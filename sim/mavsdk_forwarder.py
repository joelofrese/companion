"""Forward simulated CM5 commands to PX4 through MAVSDK."""

import math
from typing import Callable, Optional

from mavsdk.offboard import VelocityNedYaw

from control.velocity import VelocityCommand, body_to_ned


class MavsdkVelocityForwarder:
    """Send one simulated velocity setpoint to PX4."""

    def __init__(self, drone, heading_provider: Callable[[], Optional[float]]):
        self.drone = drone
        self.heading_provider = heading_provider

    async def send(self, command: VelocityCommand):
        """Send one velocity setpoint."""

        heading_deg = self.heading_provider()
        if heading_deg is None or not math.isfinite(heading_deg):
            north_m_s = east_m_s = down_m_s = 0.0
            heading_deg = 0.0
        else:
            north_m_s, east_m_s, down_m_s = body_to_ned(
                command,
                math.radians(heading_deg),
            )
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(
                north_m_s,
                east_m_s,
                down_m_s,
                heading_deg,
            )
        )
