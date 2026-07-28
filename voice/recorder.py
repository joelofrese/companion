"""Push-to-talk microphone capture for the Mac-side voice pipeline."""

from typing import Any, Optional


class PushToTalkRecorder:
    """Capture one fixed-length utterance as a mono 16 kHz float buffer."""

    def __init__(
        self,
        duration_s: float = 3.0,
        sample_rate: int = 16000,
        audio_backend: Optional[Any] = None,
    ):
        if duration_s <= 0.0 or sample_rate <= 0:
            raise ValueError("duration and sample rate must be positive")
        self.duration_s = duration_s
        self.sample_rate = sample_rate
        self._audio_backend = audio_backend

    def record(self):
        """Block until one utterance is captured and return a 1D float buffer."""

        if self._audio_backend is None:
            import sounddevice

            self._audio_backend = sounddevice
        frames = round(self.duration_s * self.sample_rate)
        audio = self._audio_backend.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocking=True,
        )
        return audio.reshape(-1)
