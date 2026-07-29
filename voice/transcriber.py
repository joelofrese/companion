"""faster-whisper voice transcription for the Mac."""


class WhisperTranscriber:
    """Turn audio into text."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_source) -> str:
        segments, _ = self._model.transcribe(audio_source, beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
