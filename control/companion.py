"""Run the Mac brain with a temporary YOLO visual fallback."""

import argparse
import asyncio

from control.fallback_brain import IntentLanguageModel
from control.mind import MacMind, Telemetry
from control.mind_runtime import MindRuntime
from control.udp_control import UdpControlService
from control.udp_sender import UdpCommandSender
from vision.legacy_yolo import YoloVisualModel
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
    intent = "following" if args.state == "following" else "idle"
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
            intent = voice_state.name.lower()

    video_config = H264StreamConfig(
        port=args.video_port,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
    )
    receiver = GStreamerH264Receiver(video_config)
    frame_reader = AsyncLatestFrameReader(receiver)
    brain = MacMind(
        YoloVisualModel(
            model_path=args.yolo_model,
            frame_width_px=video_config.width,
            target_height_px=args.target_height,
        ),
        IntentLanguageModel(),
    )
    control = MindRuntime(brain)
    sender = UdpCommandSender(args.cm5_host, args.command_port)
    service = UdpControlService(
        control,
        sender,
        frame_reader.read,
        intent_provider=lambda timestamp_s: intent,
    )
    stop_event = asyncio.Event()
    mind_stop = asyncio.Event()
    mind_task = asyncio.create_task(
        control.think_loop(
            mind_stop,
            telemetry_provider=lambda: Telemetry(),
        )
    )
    try:
        receiver.start()
        print(
            f"Companion ready: video :{video_config.port}, "
            f"commands {args.cm5_host}:{args.command_port}, intent={intent}."
        )
        await service.run(stop_event)
    finally:
        receiver.close()
        mind_stop.set()
        await mind_task
        control.close()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
