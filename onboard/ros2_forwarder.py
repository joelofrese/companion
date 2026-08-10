"""Publish velocity commands to PX4 through ROS 2."""

import math
from typing import Callable, Optional

from control.velocity import VelocityCommand, body_to_ned


class Ros2VelocityForwarder:
    """Publish velocity setpoints and the PX4 offboard heartbeat."""

    def __init__(
        self,
        heartbeat_publisher,
        setpoint_publisher,
        heartbeat_factory,
        setpoint_factory,
        timestamp_us: Callable[[], int],
        heading_provider: Callable[[], Optional[float]],
    ):
        self.heartbeat_publisher = heartbeat_publisher
        self.setpoint_publisher = setpoint_publisher
        self.heartbeat_factory = heartbeat_factory
        self.setpoint_factory = setpoint_factory
        self.timestamp_us = timestamp_us
        self.heading_provider = heading_provider

    async def send(self, command: VelocityCommand):
        """Publish one heartbeat and setpoint."""

        timestamp_us = self.timestamp_us()
        heartbeat = self.heartbeat_factory()
        heartbeat.timestamp = timestamp_us
        heartbeat.position = False
        heartbeat.velocity = True
        heartbeat.acceleration = False
        heartbeat.attitude = False
        heartbeat.body_rate = False
        heartbeat.thrust_and_torque = False
        heartbeat.direct_actuator = False
        self.heartbeat_publisher.publish(heartbeat)

        setpoint = self.setpoint_factory()
        setpoint.timestamp = timestamp_us
        setpoint.position = [math.nan, math.nan, math.nan]
        heading = self.heading_provider()
        if heading is None or not math.isfinite(heading):
            north_m_s = east_m_s = down_m_s = 0.0
            yaw_rate_rad_s = 0.0
        else:
            north_m_s, east_m_s, down_m_s = body_to_ned(command, heading)
            yaw_rate_rad_s = math.radians(command.yaw_rate_deg_s)
        setpoint.velocity = [
            north_m_s,
            east_m_s,
            down_m_s,
        ]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        # NaN leaves heading unset; yawspeed controls optional turning.
        setpoint.yaw = math.nan
        setpoint.yawspeed = yaw_rate_rad_s
        self.setpoint_publisher.publish(setpoint)
