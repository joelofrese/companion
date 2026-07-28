import asyncio
import unittest

from control.velocity import VelocityCommand
from onboard.velocity_forwarder import MavsdkVelocityForwarder


class FakeOffboard:
    def __init__(self):
        self.commands = []

    async def set_velocity_ned(self, command):
        self.commands.append(command)


class FakeDrone:
    def __init__(self):
        self.offboard = FakeOffboard()


class MavsdkVelocityForwarderTests(unittest.TestCase):
    def test_forwards_ned_velocity_fields(self):
        drone = FakeDrone()
        command = VelocityCommand(0.3, -0.1, 0.05, 12.0)
        asyncio.run(MavsdkVelocityForwarder(drone).send(command))
        forwarded = drone.offboard.commands[0]
        self.assertEqual(forwarded.north_m_s, 0.3)
        self.assertEqual(forwarded.east_m_s, -0.1)
        self.assertEqual(forwarded.down_m_s, 0.05)
        self.assertEqual(forwarded.yaw_deg, 12.0)


if __name__ == "__main__":
    unittest.main()
