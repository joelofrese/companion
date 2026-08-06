"""Connect one recorded utterance to the companion dialogue path."""


class PushToTalkVoicePipeline:
    """Capture one utterance and return its transcript."""

    def __init__(self, recorder, transcriber):
        self.recorder = recorder
        self.transcriber = transcriber

    def listen_once(self):
        return self.transcriber.transcribe(self.recorder.record()).strip()
