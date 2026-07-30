"""Forward simulated CM5 commands to PX4 through MAVSDK."""

from mavsdk.offboard import VelocityNedYaw

from control.velocity import VelocityCommand


class MavsdkVelocityForwarder:
    """Send one simulated velocity setpoint to PX4."""

    def __init__(self, drone):
        self.drone = drone

    async def send(self, command: VelocityCommand):
        """Send one velocity setpoint."""

        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(
                command.north_m_s,
                command.east_m_s,
                command.down_m_s,
                command.yaw_deg,
            )
        )
