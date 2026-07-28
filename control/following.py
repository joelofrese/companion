"""Bounded image-space person-following behavior."""

import math
from dataclasses import dataclass
from numbers import Real

from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


MAX_FORWARD_SPEED_M_S = 0.5
MAX_LATERAL_SPEED_M_S = 0.3


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


@dataclass(frozen=True)
class FollowConfig:
    frame_width_px: float = 640.0
    desired_target_height_px: float = 120.0
    max_forward_speed_m_s: float = MAX_FORWARD_SPEED_M_S
    max_lateral_speed_m_s: float = MAX_LATERAL_SPEED_M_S


class VisualFollower:
    """Use target size for distance and horizontal error for lateral motion."""

    def __init__(self, config: FollowConfig = FollowConfig()):
        if (
            not _finite(config.frame_width_px)
            or not _finite(config.desired_target_height_px)
            or not _finite(config.max_forward_speed_m_s)
            or not _finite(config.max_lateral_speed_m_s)
            or config.frame_width_px <= 0.0
            or config.desired_target_height_px <= 0.0
            or config.max_forward_speed_m_s < 0.0
            or config.max_lateral_speed_m_s < 0.0
            or config.max_forward_speed_m_s > MAX_FORWARD_SPEED_M_S
            or config.max_lateral_speed_m_s > MAX_LATERAL_SPEED_M_S
        ):
            raise ValueError("follow configuration must stay within the vehicle speed envelope")
        self.config = config

    def command(self, target: TrackEstimate) -> VelocityCommand:
        if not _finite(target.predicted_x_px) or not _finite(target.target_height_px) or target.target_height_px <= 0.0:
            return VelocityCommand()
        distance_error = (
            self.config.desired_target_height_px - target.target_height_px
        ) / self.config.desired_target_height_px
        horizontal_error = (
            target.predicted_x_px - self.config.frame_width_px / 2.0
        ) / (self.config.frame_width_px / 2.0)
        return VelocityCommand(
            north_m_s=_clamp(distance_error) * self.config.max_forward_speed_m_s,
            east_m_s=_clamp(horizontal_error) * self.config.max_lateral_speed_m_s,
        )
