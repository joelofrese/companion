import asyncio
import unittest

from control.loop import CompanionControlLoop
from control.step import CompanionControlStep
from control.state_machine import State
from control.tracking import Detection
from control.velocity import VelocityCommand
from sim.offboard import _shutdown_command
from vision.pipeline import PersonVisionPipeline


class FakeDetector:
    def detect(self, frame, timestamp_s):
        return Detection(320.0, 240.0, timestamp_s, height_px=60.0)


class CompanionControlLoopTests(unittest.TestCase):
    def test_tick_returns_heartbeat_protected_reactive_command(self):
        step = CompanionControlStep(PersonVisionPipeline(FakeDetector()))
        loop = CompanionControlLoop(step)
        self.assertEqual(loop.tick("frame", 1.0, State.FOLLOWING), VelocityCommand(north_m_s=0.25))

    def test_missed_tick_latches_zero_command(self):
        step = CompanionControlStep(PersonVisionPipeline(FakeDetector()))
        loop = CompanionControlLoop(step)
        loop.tick("frame", 1.0, State.FOLLOWING)
        self.assertEqual(loop.tick("frame", 1.2, State.FOLLOWING), VelocityCommand())
        self.assertTrue(loop.watchdog.tripped)

    def test_obstacle_is_applied_before_watchdog(self):
        step = CompanionControlStep(PersonVisionPipeline(FakeDetector()))
        loop = CompanionControlLoop(step)
        self.assertEqual(
            loop.tick("frame", 1.0, State.FOLLOWING, obstacle_distance_m=0.5),
            VelocityCommand(north_m_s=-0.2),
        )

    def test_offboard_shutdown_fails_safe_when_frame_reader_ends(self):
        class FakeControlLoop:
            def tick(self, **kwargs):
                raise AssertionError("an ended reader must not produce a control tick")

        async def ended_reader():
            return None

        command = asyncio.run(_shutdown_command(FakeControlLoop(), ended_reader))
        self.assertEqual(command, VelocityCommand())


if __name__ == "__main__":
    unittest.main()
