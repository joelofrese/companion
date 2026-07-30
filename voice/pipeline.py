"""Connect one recorded utterance to a safe intent."""

from voice.intent import parse_intent


class PushToTalkVoicePipeline:
    """Capture one utterance and return its intent."""

    def __init__(self, recorder, transcriber):
        self.recorder = recorder
        self.transcriber = transcriber

    def listen_once(self):
        return parse_intent(self.transcriber.transcribe(self.recorder.record()))
