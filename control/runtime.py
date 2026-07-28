"""Coordinator between cognitive intent, vision freshness, and reactive safety."""

from typing import Optional

from control.state_machine import ReactiveController, State
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


class CompanionRuntime:
    """Hold the latest cognitive state and target while emitting safe commands."""

    def __init__(self, controller: Optional[ReactiveController] = None):
        self.controller = controller or ReactiveController()
        self._target: Optional[TrackEstimate] = None
        self._target_updated_at_s: Optional[float] = None

    def set_intent(self, state: Optional[State]):
        """Apply a recognized cognitive intent; unknown voice stays a no-op."""

        if state is not None:
            self.controller.set_intent(state)

    def update_target(self, estimate: Optional[TrackEstimate], timestamp_s: float):
        """Store the latest vision estimate and the monotonic time it was received."""

        self._target = estimate
        self._target_updated_at_s = timestamp_s if estimate is not None else None

    def command(self, timestamp_s: float, obstacle_distance_m: Optional[float] = None) -> VelocityCommand:
        """Generate one reactive command with target age derived from current time."""

        target_age_s = None
        if self._target is not None and self._target_updated_at_s is not None:
            target_age_s = self._target.age_s + max(0.0, timestamp_s - self._target_updated_at_s)
        return self.controller.command(
            obstacle_distance_m=obstacle_distance_m,
            target_age_s=target_age_s,
        )
