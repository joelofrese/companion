"""Bounded image-space person-following behavior."""

from dataclasses import dataclass

from control.tracking import TrackEstimate
from control.velocity import VelocityCommand


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass(frozen=True)
class FollowConfig:
    frame_width_px: float = 640.0
    desired_target_height_px: float = 120.0
    max_forward_speed_m_s: float = 0.5
    max_lateral_speed_m_s: float = 0.3


class VisualFollower:
    """Use target size for distance and horizontal error for lateral motion."""

    def __init__(self, config: FollowConfig = FollowConfig()):
        if config.frame_width_px <= 0.0 or config.desired_target_height_px <= 0.0:
            raise ValueError("camera dimensions and desired target size must be positive")
        self.config = config

    def command(self, target: TrackEstimate) -> VelocityCommand:
        if target.target_height_px <= 0.0:
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
