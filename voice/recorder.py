"""Record one push-to-talk utterance."""


class PushToTalkRecorder:
    """Record a fixed-length mono audio buffer."""

    def __init__(
        self,
        duration_s: float = 3.0,
        sample_rate: int = 16000,
    ):
        if duration_s <= 0.0 or sample_rate <= 0:
            raise ValueError("duration and sample rate must be positive")
        self.duration_s = duration_s
        self.sample_rate = sample_rate

    def record(self):
        """Record one utterance."""

        import sounddevice

        frames = round(self.duration_s * self.sample_rate)
        audio = sounddevice.rec(
            frames,
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocking=True,
        )
        return audio.reshape(-1)
