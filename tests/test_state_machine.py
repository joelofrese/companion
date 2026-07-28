import unittest

from control.state_machine import ReactiveController, State
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


class ReactiveControllerTests(unittest.TestCase):
    def test_intent_drives_following_velocity(self):
        controller = ReactiveController()
        controller.set_intent(State.FOLLOWING)
        self.assertEqual(
            controller.command(
                obstacle_distance_m=2.0,
                target_age_s=0.0,
                target=TrackEstimate(320.0, 240.0, 0.0, 0.0, 320.0, 240.0, target_height_px=60.0),
            ),
            VelocityCommand(north_m_s=0.25),
        )

    def test_following_without_target_holds_position(self):
        controller = ReactiveController()
        controller.set_intent(State.FOLLOWING)
        self.assertEqual(controller.command(obstacle_distance_m=2.0), VelocityCommand())

    def test_stale_target_holds_position(self):
        controller = ReactiveController()
        controller.set_intent(State.FOLLOWING)
        self.assertEqual(controller.command(target_age_s=0.6), VelocityCommand())

    def test_obstacle_overrides_intent_and_backs_off(self):
        controller = ReactiveController()
        controller.set_intent(State.FOLLOWING)
        self.assertEqual(controller.command(obstacle_distance_m=0.5), VelocityCommand(north_m_s=-0.2))
        self.assertIs(controller.state, State.AVOIDING)

    def test_obstacle_override_recovers_to_saved_intent(self):
        controller = ReactiveController()
        controller.set_intent(State.FOLLOWING)
        controller.command(obstacle_distance_m=0.5)
        command = controller.command(
            obstacle_distance_m=2.0,
            target_age_s=0.0,
            target=TrackEstimate(320.0, 240.0, 0.0, 0.0, 320.0, 240.0, target_height_px=60.0),
        )
        self.assertEqual(command, VelocityCommand(north_m_s=0.25))
        self.assertIs(controller.state, State.FOLLOWING)

    def test_invalid_obstacle_reading_holds_and_preserves_intent(self):
        controller = ReactiveController()
        controller.set_intent(State.FOLLOWING)
        target = TrackEstimate(320.0, 240.0, 0.0, 0.0, 320.0, 240.0, target_height_px=60.0)
        for reading in (float("nan"), float("inf"), True, "unknown"):
            self.assertEqual(
                controller.command(obstacle_distance_m=reading, target_age_s=0.0, target=target),
                VelocityCommand(),
            )
            self.assertIs(controller.state, State.FOLLOWING)

    def test_no_obstacle_holds_for_non_following_states(self):
        controller = ReactiveController()
        controller.set_intent(State.RESPONDING)
        self.assertEqual(controller.command(obstacle_distance_m=2.0), VelocityCommand())


if __name__ == "__main__":
    unittest.main()
