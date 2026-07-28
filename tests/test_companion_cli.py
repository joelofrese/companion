import unittest

from control.companion import build_parser


class CompanionCliTests(unittest.TestCase):
    def test_idle_is_the_safe_default(self):
        args = build_parser().parse_args(["192.168.1.20"])
        self.assertEqual(args.state, "idle")
        self.assertFalse(args.voice_once)
        self.assertEqual(args.command_port, 5001)
        self.assertEqual(args.video_port, 5000)

    def test_following_and_stream_configuration_are_explicit(self):
        args = build_parser().parse_args(
            [
                "drone.local",
                "--state",
                "following",
                "--voice-once",
                "--whisper-model",
                "base.en",
                "--record-duration",
                "2",
                "--command-port",
                "6001",
                "--video-port",
                "6000",
                "--width",
                "320",
                "--height",
                "240",
                "--framerate",
                "20",
                "--target-height",
                "80",
            ]
        )
        self.assertEqual(args.state, "following")
        self.assertTrue(args.voice_once)
        self.assertEqual(args.whisper_model, "base.en")
        self.assertEqual(args.record_duration, 2.0)
        self.assertEqual((args.command_port, args.video_port), (6001, 6000))
        self.assertEqual((args.width, args.height, args.framerate), (320, 240, 20))
        self.assertEqual(args.target_height, 80.0)


if __name__ == "__main__":
    unittest.main()
