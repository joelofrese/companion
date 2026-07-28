import unittest

from control.following import FollowConfig, VisualFollower
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


def target(x_px, height_px):
    return TrackEstimate(x_px, 240.0, 0.0, 0.0, x_px, 240.0, target_height_px=height_px)


class VisualFollowerTests(unittest.TestCase):
    def test_large_target_holds_instead_of_reversing(self):
        follower = VisualFollower()
        self.assertEqual(follower.command(target(320.0, 240.0)), VelocityCommand())

    def test_invalid_target_geometry_holds(self):
        follower = VisualFollower()
        for height in (float("nan"), float("inf"), True, "unknown"):
            self.assertEqual(follower.command(target(320.0, height)), VelocityCommand())
        self.assertEqual(follower.command(target(float("nan"), 120.0)), VelocityCommand())

if __name__ == "__main__":
    unittest.main()
