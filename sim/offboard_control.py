"""Deterministic cognitive intent profile for SITL bring-up."""

from dataclasses import dataclass

from control.state_machine import State
from control.tracking import TrackEstimate
from voice.pipeline import VoiceCommandPipeline


SETPOINT_PERIOD_S = 0.05
PROFILE_DURATION_S = 32.0
FOLLOW_END_S = 4.0
SECOND_FOLLOW_START_S = 12.0
SECOND_FOLLOW_END_S = 16.0
THIRD_FOLLOW_START_S = 24.0
THIRD_FOLLOW_END_S = 28.0


@dataclass(frozen=True)
class DistanceMessage:
    """Small simulation equivalent of PX4's distance-sensor message."""

    current_distance: float
    min_distance: float = 0.0
    max_distance: float = 10.0


class _DemoTranscriber:
    def transcribe(self, transcript):
        return transcript


_demo_voice = VoiceCommandPipeline(_DemoTranscriber())


def demo_state(elapsed_s: float) -> State:
    """Repeat a short follow-and-hover behavior cycle."""

    following = (
        0.0 <= elapsed_s < FOLLOW_END_S
        or SECOND_FOLLOW_START_S <= elapsed_s < SECOND_FOLLOW_END_S
        or THIRD_FOLLOW_START_S <= elapsed_s < THIRD_FOLLOW_END_S
    )
    transcript = "follow me" if following else "hover"
    state = _demo_voice.handle(transcript)
    if state is None:
        raise RuntimeError(f"demo transcript produced no state: {transcript!r}")
    return state


def demo_target() -> TrackEstimate:
    """Represent a fresh, stable target for the deterministic flight scenario."""

    return TrackEstimate(320.0, 240.0, 0.0, 0.0, 320.0, 240.0, target_height_px=60.0)


def demo_obstacle_distance_m(elapsed_s: float) -> float:
    """Simulate a forward TOF obstacle for one second during following."""

    return 0.5 if 2.0 <= elapsed_s < 3.0 else 2.0


class DemoVision:
    """Synthetic camera provider used to exercise the shared control step in SITL."""

    def process(self, frame, timestamp_s):
        return demo_target()
