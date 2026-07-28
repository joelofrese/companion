"""Voice transcription and intent parsing boundary."""

from typing import Any, Optional, Protocol

from control.state_machine import State
from voice.intent import parse_intent


class Transcriber(Protocol):
    def transcribe(self, audio_source: Any) -> str:
        ...


class AudioRecorder(Protocol):
    def record(self) -> Any:
        ...


class VoiceCommandPipeline:
    def __init__(self, transcriber: Transcriber):
        self.transcriber = transcriber

    def handle(self, audio_source: Any) -> Optional[State]:
        """Transcribe one utterance and return only a recognized cognitive intent."""

        return parse_intent(self.transcriber.transcribe(audio_source))


class PushToTalkVoicePipeline:
    """Capture one utterance and convert it into one optional cognitive state."""

    def __init__(self, recorder: AudioRecorder, transcriber: Transcriber):
        self.recorder = recorder
        self.commands = VoiceCommandPipeline(transcriber)

    def listen_once(self) -> Optional[State]:
        return self.commands.handle(self.recorder.record())
