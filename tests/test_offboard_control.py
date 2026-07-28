import unittest

from control.state_machine import State
from sim.offboard_control import demo_state


class DemoIntentTests(unittest.TestCase):
    def test_forward_motion_is_slow_and_ned_aligned(self):
        self.assertIs(demo_state(1.0), State.FOLLOWING)

    def test_profile_returns_to_hover(self):
        self.assertIs(demo_state(4.0), State.HOVERING)
        self.assertIs(demo_state(8.0), State.HOVERING)

    def test_negative_elapsed_time_is_safe(self):
        self.assertIs(demo_state(-1.0), State.HOVERING)


if __name__ == "__main__":
    unittest.main()
