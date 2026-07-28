import unittest

from control.following import FollowConfig, VisualFollower
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


def target(x_px, height_px):
    return TrackEstimate(x_px, 240.0, 0.0, 0.0, x_px, 240.0, target_height_px=height_px)


class VisualFollowerTests(unittest.TestCase):
    def test_small_centered_target_moves_forward(self):
        follower = VisualFollower()
        self.assertEqual(follower.command(target(320.0, 60.0)), VelocityCommand(north_m_s=0.25))

    def test_large_target_backs_off(self):
        follower = VisualFollower()
        self.assertEqual(follower.command(target(320.0, 240.0)), VelocityCommand(north_m_s=-0.5))

    def test_right_of_center_moves_east_and_is_bounded(self):
        follower = VisualFollower(FollowConfig(frame_width_px=640.0))
        self.assertEqual(follower.command(target(640.0, 120.0)), VelocityCommand(east_m_s=0.3))

    def test_missing_size_holds(self):
        self.assertEqual(VisualFollower().command(target(320.0, 0.0)), VelocityCommand())


if __name__ == "__main__":
    unittest.main()
