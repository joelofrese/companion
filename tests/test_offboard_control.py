import unittest

from control.velocity import VelocityCommand
from sim.offboard_control import demo_velocity


class DemoVelocityTests(unittest.TestCase):
    def test_forward_motion_is_slow_and_ned_aligned(self):
        self.assertEqual(demo_velocity(1.0), VelocityCommand(north_m_s=0.5))

    def test_profile_returns_to_hover(self):
        self.assertEqual(demo_velocity(4.0), VelocityCommand())
        self.assertEqual(demo_velocity(8.0), VelocityCommand())

    def test_negative_elapsed_time_is_safe(self):
        self.assertEqual(demo_velocity(-1.0), VelocityCommand())


if __name__ == "__main__":
    unittest.main()
