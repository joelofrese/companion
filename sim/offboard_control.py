"""Deterministic cognitive intent profile for SITL bring-up."""

from control.state_machine import State
from control.tracking import Detection, TrackEstimate
from voice.pipeline import VoiceCommandPipeline
from vision.pipeline import PersonVisionPipeline


class _DemoTranscriber:
    def transcribe(self, transcript):
        return transcript


_demo_voice = VoiceCommandPipeline(_DemoTranscriber())


def demo_state(elapsed_s: float) -> State:
    """Follow for four seconds, then ask the reactive layer to hover."""

    transcript = "follow me" if 0.0 <= elapsed_s < 4.0 else "hover"
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


class _VideoDemoDetector:
    """Turn any decoded test frame into a centered synthetic person detection."""

    def detect(self, frame, timestamp_s):
        if frame is None or len(frame.shape) < 2:
            raise ValueError("decoded video frame must have image dimensions")
        height, width = frame.shape[:2]
        return Detection(width / 2.0, height / 2.0, timestamp_s, height_px=height / 8.0)


class VideoDemoVision:
    """Use the real decoded-frame path with a deterministic detector for SITL."""

    def __init__(self):
        self._pipeline = PersonVisionPipeline(_VideoDemoDetector())

    def process(self, frame, timestamp_s):
        return self._pipeline.process(frame, timestamp_s)
