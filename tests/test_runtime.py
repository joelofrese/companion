import unittest

from control.runtime import CompanionRuntime
from control.state_machine import State
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


TARGET = TrackEstimate(100.0, 100.0, 0.0, 0.0, 100.0, 100.0)


class CompanionRuntimeTests(unittest.TestCase):
    def test_fresh_target_and_follow_intent_generate_motion(self):
        runtime = CompanionRuntime()
        runtime.set_intent(State.FOLLOWING)
        runtime.update_target(TARGET, timestamp_s=10.0)
        self.assertEqual(runtime.command(timestamp_s=10.4), VelocityCommand(north_m_s=0.5))

    def test_missing_target_holds(self):
        runtime = CompanionRuntime()
        runtime.set_intent(State.FOLLOWING)
        self.assertEqual(runtime.command(timestamp_s=10.0), VelocityCommand())

    def test_target_expires_from_runtime_clock(self):
        runtime = CompanionRuntime()
        runtime.set_intent(State.FOLLOWING)
        runtime.update_target(TARGET, timestamp_s=10.0)
        self.assertEqual(runtime.command(timestamp_s=10.6), VelocityCommand())

    def test_obstacle_still_overrides_a_fresh_target(self):
        runtime = CompanionRuntime()
        runtime.set_intent(State.FOLLOWING)
        runtime.update_target(TARGET, timestamp_s=10.0)
        self.assertEqual(runtime.command(10.1, obstacle_distance_m=0.5), VelocityCommand(north_m_s=-0.2))

    def test_unknown_intent_does_not_change_safe_default(self):
        runtime = CompanionRuntime()
        runtime.set_intent(None)
        self.assertEqual(runtime.command(timestamp_s=0.0), VelocityCommand())


if __name__ == "__main__":
    unittest.main()
