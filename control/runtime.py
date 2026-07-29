"""Run one Mac control step."""

from typing import Any, Optional, Protocol

from control.state_machine import ReactiveController, State
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand
from control.watchdog import SetpointWatchdog


class VisionPipeline(Protocol):
    def process(self, frame: Any, timestamp_s: float) -> Optional[TrackEstimate]:
        ...


class CompanionRuntime:
    """Turn one frame, intent, and sensor reading into one command."""

    def __init__(
        self,
        vision: VisionPipeline,
        controller: Optional[ReactiveController] = None,
        watchdog: Optional[SetpointWatchdog] = None,
    ):
        self.vision = vision
        self.controller = controller or ReactiveController()
        self.watchdog = watchdog or SetpointWatchdog()

    def tick(
        self,
        frame: Any,
        timestamp_s: float,
        intent: Optional[State] = None,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        """Return the safe command for this control step."""

        target = self.vision.process(frame, timestamp_s)
        if intent is not None:
            self.controller.set_intent(intent)
        desired = self.controller.command(
            obstacle_distance_m=obstacle_distance_m,
            target_age_s=target.age_s if target is not None else None,
            target=target,
        )
        return self.watchdog.emit(timestamp_s, desired)
