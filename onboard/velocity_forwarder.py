"""MAVSDK velocity sink used by the CM5-to-PX4 integration seam."""

from mavsdk.offboard import VelocityNedYaw

from control.velocity import VelocityCommand


class MavsdkVelocityForwarder:
    """Forward only already-approved NED commands to an active MAVSDK vehicle."""

    def __init__(self, drone):
        self.drone = drone

    async def send(self, command: VelocityCommand):
        """Send one velocity setpoint; mode and arming remain flight-loop concerns."""

        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(
                command.north_m_s,
                command.east_m_s,
                command.down_m_s,
                command.yaw_deg,
            )
        )
