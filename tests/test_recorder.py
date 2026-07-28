import unittest

import numpy as np

from voice.recorder import PushToTalkRecorder


class FakeAudioBackend:
    def __init__(self):
        self.calls = []

    def rec(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return np.zeros((args[0], kwargs["channels"]), dtype=np.float32)


class PushToTalkRecorderTests(unittest.TestCase):
    def test_records_mono_float_audio_at_configured_rate(self):
        backend = FakeAudioBackend()
        recorder = PushToTalkRecorder(duration_s=0.25, sample_rate=8000, audio_backend=backend)

        audio = recorder.record()

        self.assertEqual(audio.shape, (2000,))
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(backend.calls[0], ((2000,), {
            "samplerate": 8000,
            "channels": 1,
            "dtype": "float32",
            "blocking": True,
        }))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            PushToTalkRecorder(duration_s=0.0)
        with self.assertRaises(ValueError):
            PushToTalkRecorder(sample_rate=0)


if __name__ == "__main__":
    unittest.main()
