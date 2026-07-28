"""Lazy faster-whisper adapter for Mac-side audio transcription."""

from typing import Any, Optional


class WhisperTranscriber:
    """Transcribe an audio path or compatible audio buffer into plain text."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        model: Optional[Any] = None,
    ):
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._model = model

    def transcribe(self, audio_source: Any) -> str:
        segments, _ = self._model.transcribe(audio_source, beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
