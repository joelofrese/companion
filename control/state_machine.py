"""Turn intent and sensors into one safe command."""

import math
from enum import Enum, auto
from numbers import Real
from typing import Optional

from control.following import VisualFollower
from control.safety_limits import BACKOFF_SPEED_M_S, OBSTACLE_STOP_M
from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


class State(Enum):
    IDLE = auto()
    FOLLOWING = auto()
    HOVERING = auto()


TARGET_MAX_AGE_S = 0.5


class ReactiveController:
    def __init__(self, follower: Optional[VisualFollower] = None):
        self.intent = State.IDLE
        self.follower = follower or VisualFollower()

    def set_intent(self, state: State):
        """Set the requested behavior."""

        if not isinstance(state, State):
            raise ValueError("intent must be a State")
        self.intent = state

    def command(
        self,
        obstacle_distance_m: Optional[float] = None,
        target_age_s: Optional[float] = None,
        target: Optional[TrackEstimate] = None,
    ) -> VelocityCommand:
        """Return one safe velocity command."""

        if obstacle_distance_m is not None:
            if isinstance(obstacle_distance_m, bool) or not isinstance(obstacle_distance_m, Real):
                return VelocityCommand()
            if not math.isfinite(obstacle_distance_m):
                return VelocityCommand()
            if obstacle_distance_m < 0.0:
                return VelocityCommand()
            if obstacle_distance_m < OBSTACLE_STOP_M:
                return VelocityCommand(north_m_s=-BACKOFF_SPEED_M_S)

        if (
            self.intent is State.FOLLOWING
            and target_age_s is not None
            and 0.0 <= target_age_s <= TARGET_MAX_AGE_S
            and target is not None
        ):
            return self.follower.command(target)
        return VelocityCommand()
