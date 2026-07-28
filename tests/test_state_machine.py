import unittest

from control.state_machine import ReactiveController, State
from control.velocity import VelocityCommand


class ReactiveControllerTests(unittest.TestCase):
    def test_intent_drives_following_velocity(self):
        controller = ReactiveController()
        controller.set_intent(State.FOLLOWING)
        self.assertEqual(
            controller.command(obstacle_distance_m=2.0, target_age_s=0.0),
            VelocityCommand(north_m_s=0.5),
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

    def test_no_obstacle_holds_for_non_following_states(self):
        controller = ReactiveController()
        controller.set_intent(State.RESPONDING)
        self.assertEqual(controller.command(obstacle_distance_m=2.0), VelocityCommand())


if __name__ == "__main__":
    unittest.main()
