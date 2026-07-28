import unittest

from control.step import CompanionControlStep
from control.state_machine import State
from control.tracking import Detection
from control.velocity import VelocityCommand
from vision.pipeline import PersonVisionPipeline


class FakeDetector:
    def __init__(self, detections):
        self.detections = iter(detections)

    def detect(self, frame, timestamp_s):
        return next(self.detections)


class CompanionControlStepTests(unittest.TestCase):
    def test_frame_and_follow_intent_produce_velocity(self):
        vision = PersonVisionPipeline(FakeDetector([Detection(100.0, 100.0, 1.0)]))
        step = CompanionControlStep(vision)
        self.assertEqual(
            step.process("frame", 1.0, intent=State.FOLLOWING),
            VelocityCommand(north_m_s=0.5),
        )

    def test_missing_frame_target_holds(self):
        vision = PersonVisionPipeline(FakeDetector([None]))
        step = CompanionControlStep(vision)
        self.assertEqual(step.process("frame", 1.0, intent=State.FOLLOWING), VelocityCommand())

    def test_obstacle_wins_over_vision_target(self):
        vision = PersonVisionPipeline(FakeDetector([Detection(100.0, 100.0, 1.0)]))
        step = CompanionControlStep(vision)
        self.assertEqual(
            step.process("frame", 1.0, intent=State.FOLLOWING, obstacle_distance_m=0.5),
            VelocityCommand(north_m_s=-0.2),
        )


if __name__ == "__main__":
    unittest.main()
