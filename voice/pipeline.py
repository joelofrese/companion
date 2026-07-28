"""Voice transcription and intent parsing boundary."""

from typing import Any, Optional, Protocol

from control.state_machine import State
from voice.intent import parse_intent


class Transcriber(Protocol):
    def transcribe(self, audio_source: Any) -> str:
        ...


class VoiceCommandPipeline:
    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    def handle(self, audio_source: Any) -> Optional[State]:
        """Transcribe one utterance and return only a recognized cognitive intent."""

        return parse_intent(self.transcriber.transcribe(audio_source))
