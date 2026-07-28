"""Reusable one-tick scheduler for the reactive companion command path."""

from typing import Any, Optional

from control.state_machine import State
from control.step import CompanionControlStep
from control.velocity import VelocityCommand
from control.watchdog import SetpointWatchdog


class CompanionControlLoop:
    """Compute and heartbeat-protect one command at a time."""

    def __init__(self, step: CompanionControlStep, watchdog: Optional[SetpointWatchdog] = None):
        self.step = step
        self.watchdog = watchdog or SetpointWatchdog()

    def tick(
        self,
        frame: Any,
        timestamp_s: float,
        intent: Optional[State] = None,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        desired = self.step.process(
            frame=frame,
            timestamp_s=timestamp_s,
            intent=intent,
            obstacle_distance_m=obstacle_distance_m,
        )
        return self.watchdog.emit(timestamp_s, desired)
