"""Deterministic cognitive intent profile for SITL bring-up."""

from control.state_machine import State
from control.tracking import TrackEstimate
from voice.pipeline import VoiceCommandPipeline


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

    return TrackEstimate(320.0, 240.0, 0.0, 0.0, 320.0, 240.0)
