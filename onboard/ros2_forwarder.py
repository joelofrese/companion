"""Publish velocity commands to PX4 through ROS 2."""

import math
from typing import Callable

from control.velocity import VelocityCommand


class Ros2VelocityForwarder:
    """Publish velocity setpoints and the PX4 offboard heartbeat."""

    def __init__(
        self,
        heartbeat_publisher,
        setpoint_publisher,
        heartbeat_factory,
        setpoint_factory,
        timestamp_us: Callable[[], int],
    ):
        self.heartbeat_publisher = heartbeat_publisher
        self.setpoint_publisher = setpoint_publisher
        self.heartbeat_factory = heartbeat_factory
        self.setpoint_factory = setpoint_factory
        self.timestamp_us = timestamp_us

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
        setpoint.velocity = [
            command.north_m_s,
            command.east_m_s,
            command.down_m_s,
        ]
        setpoint.acceleration = [math.nan, math.nan, math.nan]
        setpoint.yaw = math.radians(command.yaw_deg)
        setpoint.yawspeed = math.nan
        self.setpoint_publisher.publish(setpoint)
