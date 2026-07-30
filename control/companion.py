"""Run the temporary YOLO-based Mac control fallback."""

import argparse
import asyncio

from control.following import FollowConfig, VisualFollower
from control.runtime import CompanionRuntime
from control.reactive import ReactiveController, State
from control.udp_control import UdpControlService
from control.udp_sender import UdpCommandSender
from vision.latest import LatestVisionPipeline
from vision.legacy_yolo import YoloPersonDetector
from vision.pipeline import PersonVisionPipeline
from vision.video_stream import AsyncLatestFrameReader, GStreamerH264Receiver, H264StreamConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Run the Mac-side Companion control stack")
    parser.add_argument("cm5_host", help="CM5 IP address or hostname")
    parser.add_argument("--state", choices=("idle", "following"), default="idle")
    parser.add_argument(
        "--voice-once",
        action="store_true",
        help="capture one push-to-talk utterance before starting flight control",
    )
    parser.add_argument("--whisper-model", default="tiny.en")
    parser.add_argument("--record-duration", type=float, default=3.0)
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--command-port", type=int, default=5001)
    parser.add_argument("--video-port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--target-height", type=float, default=120.0)
    return parser


async def run(args):
    state = State.FOLLOWING if args.state == "following" else State.IDLE
    if args.voice_once:
        from voice.pipeline import PushToTalkVoicePipeline
        from voice.recorder import PushToTalkRecorder
        from voice.transcriber import WhisperTranscriber

        def listen_once():
            return PushToTalkVoicePipeline(
                PushToTalkRecorder(duration_s=args.record_duration),
                WhisperTranscriber(model_size=args.whisper_model),
            ).listen_once()

        voice_state = await asyncio.to_thread(listen_once)
        if voice_state is not None:
            state = voice_state

    video_config = H264StreamConfig(
        port=args.video_port,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
    )
    receiver = GStreamerH264Receiver(video_config)
    frame_reader = AsyncLatestFrameReader(receiver)
    vision = LatestVisionPipeline(
        PersonVisionPipeline(YoloPersonDetector(model_path=args.yolo_model))
    )
    control = CompanionRuntime(
        vision,
        ReactiveController(
            VisualFollower(
                FollowConfig(
                    frame_width_px=video_config.width,
                    desired_target_height_px=args.target_height,
                )
            )
        )
    )
    sender = UdpCommandSender(args.cm5_host, args.command_port)
    service = UdpControlService(
        control,
        sender,
        frame_reader.read,
        intent_provider=lambda timestamp_s: state,
    )
    stop_event = asyncio.Event()
    try:
        receiver.start()
        print(
            f"Companion ready: video :{video_config.port}, "
            f"commands {args.cm5_host}:{args.command_port}, state={state.name.lower()}."
        )
        await service.run(stop_event)
    finally:
        receiver.close()
        vision.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
