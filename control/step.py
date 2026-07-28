"""One synchronous frame-to-command step for the companion control loop."""

from typing import Any, Optional, Protocol

from control.runtime import CompanionRuntime
from control.state_machine import State
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


class VisionPipeline(Protocol):
    def process(self, frame: Any, timestamp_s: float) -> Optional[TrackEstimate]:
        ...


class CompanionControlStep:
    """Compose one vision frame, cognitive intent, and sensor snapshot."""

    def __init__(self, vision: VisionPipeline, runtime: Optional[CompanionRuntime] = None):
        self.vision = vision
        self.runtime = runtime or CompanionRuntime()

    def process(
        self,
        frame: Any,
        timestamp_s: float,
        intent: Optional[State] = None,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        """Return the only command that may leave the cognitive/reactive stack."""

        estimate = self.vision.process(frame, timestamp_s)
        self.runtime.set_intent(intent)
        self.runtime.update_target(estimate, timestamp_s)
        return self.runtime.command(timestamp_s, obstacle_distance_m=obstacle_distance_m)
