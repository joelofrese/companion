import unittest

from control.state_machine import State
from sim.offboard_control import VideoDemoVision, demo_obstacle_distance_m, demo_state


class DemoIntentTests(unittest.TestCase):
    def test_forward_motion_is_slow_and_ned_aligned(self):
        self.assertIs(demo_state(1.0), State.FOLLOWING)

    def test_profile_returns_to_hover(self):
        self.assertIs(demo_state(4.0), State.HOVERING)
        self.assertIs(demo_state(8.0), State.HOVERING)

    def test_negative_elapsed_time_is_safe(self):
        self.assertIs(demo_state(-1.0), State.HOVERING)

    def test_demo_intent_uses_voice_state_mapping(self):
        self.assertIs(demo_state(1.0), State.FOLLOWING)
        self.assertIs(demo_state(5.0), State.HOVERING)

    def test_demo_obstacle_profile_has_one_forward_sensor_event(self):
        self.assertEqual(demo_obstacle_distance_m(1.9), 2.0)
        self.assertEqual(demo_obstacle_distance_m(2.0), 0.5)
        self.assertEqual(demo_obstacle_distance_m(3.0), 2.0)

    def test_video_demo_vision_uses_decoded_frame_dimensions(self):
        frame = type("Frame", (), {"shape": (48, 64, 3)})()
        estimate = VideoDemoVision().process(frame, 1.0)
        self.assertEqual(estimate.x_px, 32.0)
        self.assertEqual(estimate.target_height_px, 6.0)


if __name__ == "__main__":
    unittest.main()
