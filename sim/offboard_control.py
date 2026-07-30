"""The fixed intent and sensor schedule used by SITL."""

import math
from dataclasses import dataclass

SETPOINT_PERIOD_S = 0.05
PROFILE_DURATION_S = 32.0
FOLLOW_END_S = 4.0
SECOND_FOLLOW_START_S = 12.0
SECOND_FOLLOW_END_S = 16.0
THIRD_FOLLOW_START_S = 24.0
THIRD_FOLLOW_END_S = 28.0
INVALID_DISTANCE_START_S = 13.0
INVALID_DISTANCE_END_S = 13.5
TARGET_LOST_START_S = 14.0
TARGET_LOST_END_S = 14.7
COMMAND_DROPOUT_START_S = 24.0
COMMAND_DROPOUT_END_S = 24.5


@dataclass(frozen=True)
class DistanceMessage:
    """Distance data used by the simulation."""

    current_distance: float
    min_distance: float = 0.0
    max_distance: float = 10.0


def demo_state(elapsed_s: float) -> str:
    """Repeat following and hover intents."""

    following = (
        0.0 <= elapsed_s < FOLLOW_END_S
        or SECOND_FOLLOW_START_S <= elapsed_s < SECOND_FOLLOW_END_S
        or THIRD_FOLLOW_START_S <= elapsed_s < THIRD_FOLLOW_END_S
    )
    return "following" if following else "hover"


def demo_obstacle_distance_m(elapsed_s: float) -> float:
    """Return the simulated obstacle distance."""

    if INVALID_DISTANCE_START_S <= elapsed_s < INVALID_DISTANCE_END_S:
        return math.nan
    return 0.5 if 2.0 <= elapsed_s < 3.0 else 2.0
