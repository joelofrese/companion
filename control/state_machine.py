"""Minimal reactive state machine: intent selects behavior, obstacles override it."""

from enum import Enum, auto
from typing import Optional

from control.velocity import VelocityCommand


class State(Enum):
    IDLE = auto()
    FOLLOWING = auto()
    AVOIDING = auto()
    HOVERING = auto()
    RESPONDING = auto()


OBSTACLE_STOP_M = 0.6
BACKOFF_SPEED_M_S = 0.2


class ReactiveController:
    def __init__(self):
        self.state = State.IDLE

    def set_intent(self, state: State):
        """Apply cognitive intent; obstacle safety is evaluated when commanding."""

        self.state = state

    def command(self, obstacle_distance_m: Optional[float] = None) -> VelocityCommand:
        """Return one safe velocity command for the current sensor snapshot."""

        if obstacle_distance_m is not None and obstacle_distance_m < OBSTACLE_STOP_M:
            self.state = State.AVOIDING
            return VelocityCommand(north_m_s=-BACKOFF_SPEED_M_S)

        if self.state is State.FOLLOWING:
            return VelocityCommand(north_m_s=0.5)
        return VelocityCommand()
