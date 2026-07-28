import unittest

from control.state_machine import State
from voice.intent import parse_intent
from voice.pipeline import VoiceCommandPipeline
from voice.transcriber import WhisperTranscriber


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeWhisperModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return iter([FakeSegment(" follow"), FakeSegment(" me ")]), {"language": "en"}


class FakeTranscriber:
    def __init__(self, transcript):
        self.transcript = transcript
        self.source = None

    def transcribe(self, source):
        self.source = source
        return self.transcript


class VoiceTests(unittest.TestCase):
    def test_parser_maps_commands_to_cognitive_states(self):
        self.assertIs(parse_intent("Follow me!"), State.FOLLOWING)
        self.assertIs(parse_intent("please hover"), State.HOVERING)
        self.assertIs(parse_intent("turn to me"), State.RESPONDING)
        self.assertIs(parse_intent("go idle"), State.IDLE)

    def test_stop_takes_precedence_over_follow(self):
        self.assertIs(parse_intent("stop following"), State.HOVERING)

    def test_unknown_or_empty_transcript_is_rejected(self):
        self.assertIsNone(parse_intent("what is the weather"))
        self.assertIsNone(parse_intent(""))

    def test_pipeline_passes_audio_to_transcriber(self):
        transcriber = FakeTranscriber("follow me")
        pipeline = VoiceCommandPipeline(transcriber)
        self.assertIs(pipeline.handle("audio"), State.FOLLOWING)
        self.assertEqual(transcriber.source, "audio")

    def test_whisper_adapter_joins_segments_and_uses_conservative_options(self):
        model = FakeWhisperModel()
        transcriber = WhisperTranscriber(model=model)
        self.assertEqual(transcriber.transcribe("audio"), "follow me")
        self.assertEqual(model.calls[0], ("audio", {"beam_size": 1, "vad_filter": True}))


if __name__ == "__main__":
    unittest.main()
