"""Minimal reactive state machine: intent selects behavior, obstacles override it."""

import math
from enum import Enum, auto
from numbers import Real
from typing import Optional

from control.following import VisualFollower
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


class State(Enum):
    IDLE = auto()
    FOLLOWING = auto()
    AVOIDING = auto()
    HOVERING = auto()
    RESPONDING = auto()


OBSTACLE_STOP_M = 0.6
BACKOFF_SPEED_M_S = 0.2
TARGET_MAX_AGE_S = 0.5


class ReactiveController:
    def __init__(self, follower: Optional[VisualFollower] = None):
        self.state = State.IDLE
        self._intent_state = State.IDLE
        self.follower = follower or VisualFollower()

    def set_intent(self, state: State):
        """Apply cognitive intent; obstacle safety is evaluated when commanding."""

        self._intent_state = state
        self.state = state

    def command(
        self,
        obstacle_distance_m: Optional[float] = None,
        target_age_s: Optional[float] = None,
        target: Optional[TrackEstimate] = None,
    ) -> VelocityCommand:
        """Return one safe velocity command for the current sensor snapshot."""

        if obstacle_distance_m is not None:
            if isinstance(obstacle_distance_m, bool) or not isinstance(obstacle_distance_m, Real):
                return VelocityCommand()
            if not math.isfinite(obstacle_distance_m):
                return VelocityCommand()
            if obstacle_distance_m < OBSTACLE_STOP_M:
                self.state = State.AVOIDING
                return VelocityCommand(north_m_s=-BACKOFF_SPEED_M_S)

        if self.state is State.AVOIDING:
            self.state = self._intent_state
        if self.state is State.FOLLOWING and target_age_s is not None and 0.0 <= target_age_s <= TARGET_MAX_AGE_S and target is not None:
            return self.follower.command(target)
        return VelocityCommand()
