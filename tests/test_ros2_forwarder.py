import asyncio
import math
import unittest

from control.velocity import VelocityCommand
from onboard.ros2_forwarder import Ros2VelocityForwarder


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeHeartbeat:
    pass


class FakeSetpoint:
    pass


class Ros2VelocityForwarderTests(unittest.TestCase):
    def test_publishes_velocity_only_heartbeat_and_ned_setpoint(self):
        heartbeat_publisher = FakePublisher()
        setpoint_publisher = FakePublisher()
        forwarder = Ros2VelocityForwarder(
            heartbeat_publisher,
            setpoint_publisher,
            FakeHeartbeat,
            FakeSetpoint,
            timestamp_us=lambda: 123,
        )

        asyncio.run(forwarder.send(VelocityCommand(0.3, -0.1, 0.05, 12.0)))

        heartbeat = heartbeat_publisher.messages[0]
        self.assertEqual(heartbeat.timestamp, 123)
        self.assertFalse(heartbeat.position)
        self.assertTrue(heartbeat.velocity)
        self.assertFalse(heartbeat.acceleration)
        self.assertFalse(heartbeat.attitude)
        self.assertFalse(heartbeat.body_rate)
        self.assertFalse(heartbeat.thrust_and_torque)
        self.assertFalse(heartbeat.direct_actuator)

        setpoint = setpoint_publisher.messages[0]
        self.assertEqual(setpoint.timestamp, 123)
        self.assertEqual(setpoint.velocity, [0.3, -0.1, 0.05])
        self.assertEqual(setpoint.yaw, math.radians(12.0))
        self.assertTrue(math.isnan(setpoint.yawspeed))
        self.assertTrue(all(math.isnan(value) for value in setpoint.position))
        self.assertTrue(all(math.isnan(value) for value in setpoint.acceleration))


if __name__ == "__main__":
    unittest.main()
