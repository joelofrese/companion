"""Forward simulated CM5 commands to PX4 through MAVSDK."""

import math

from mavsdk.offboard import VelocityNedYaw

from control.velocity import VelocityCommand, body_to_ned


class MavsdkVelocityForwarder:
    """Send one simulated velocity setpoint to PX4."""

    def __init__(self, drone, heading_deg):
        self.drone = drone
        self.heading_deg = heading_deg

    async def send(self, command: VelocityCommand):
        """Send one velocity setpoint."""

        north_m_s, east_m_s, down_m_s = body_to_ned(
            command,
            math.radians(self.heading_deg),
        )
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(
                north_m_s,
                east_m_s,
                down_m_s,
                self.heading_deg,
            )
        )
