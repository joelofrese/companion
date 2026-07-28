import asyncio
import unittest

from control.runtime import CompanionRuntime
from control.state_machine import State
from control.tracking import Detection, TrackEstimate
from control.velocity import VelocityCommand
from sim.offboard import _shutdown_command
from vision.pipeline import PersonVisionPipeline


class FakeDetector:
    def __init__(self, detection):
        self.detection = detection

    def detect(self, frame, timestamp_s):
        return self.detection


class StaticVision:
    def __init__(self, estimate):
        self.estimate = estimate

    def process(self, frame, timestamp_s):
        return self.estimate


class CompanionRuntimeTests(unittest.TestCase):
    def test_tick_connects_vision_intent_and_reactive_motion(self):
        runtime = CompanionRuntime(
            PersonVisionPipeline(FakeDetector(Detection(320.0, 240.0, 1.0, height_px=60.0)))
        )
        command = runtime.tick("frame", 1.0, State.FOLLOWING)
        self.assertGreater(command.north_m_s, 0.0)

    def test_missing_target_holds(self):
        runtime = CompanionRuntime(StaticVision(None))
        self.assertEqual(runtime.tick(None, 10.0, State.FOLLOWING), VelocityCommand())

    def test_target_expires_from_runtime_clock(self):
        target = TrackEstimate(320.0, 240.0, 0.0, 0.0, 320.0, 240.0, age_s=0.6, target_height_px=60.0)
        runtime = CompanionRuntime(StaticVision(target))
        self.assertEqual(runtime.tick(None, 10.0, State.FOLLOWING), VelocityCommand())

    def test_obstacle_still_overrides_a_fresh_target(self):
        target = TrackEstimate(320.0, 240.0, 0.0, 0.0, 320.0, 240.0, target_height_px=60.0)
        runtime = CompanionRuntime(StaticVision(target))
        self.assertEqual(runtime.tick(None, 10.0, State.FOLLOWING, 0.5), VelocityCommand(north_m_s=-0.2))

    def test_missed_tick_latches_zero(self):
        runtime = CompanionRuntime(StaticVision(None))
        runtime.tick(None, 1.0)
        self.assertEqual(runtime.tick(None, 1.2), VelocityCommand())
        self.assertTrue(runtime.watchdog.tripped)

    def test_offboard_shutdown_holds_when_frame_reader_ends(self):
        class NeverCalledControl:
            def tick(self, **kwargs):
                raise AssertionError("an ended reader must not produce a control tick")

        async def ended_reader():
            return None

        self.assertEqual(
            asyncio.run(_shutdown_command(NeverCalledControl(), ended_reader)),
            VelocityCommand(),
        )


if __name__ == "__main__":
    unittest.main()
