"""Run the companion brain beside the CM5 safety bridge."""

import argparse
import asyncio
import sys

from control.memory import CompanionMemory
from control.dialogue import DialogueInput
from control.udp_control import UdpControlService
from control.udp_sender import UdpCommandSender
from onboard.video_sender import GStreamerH264Sender
from vision.video_stream import AsyncLatestFrameReader, GStreamerH264Receiver, H264StreamConfig


DEFAULT_SITUATION = "explore the surroundings"


def build_parser():
    parser = argparse.ArgumentParser(description="Run the Companion control stack")
    parser.add_argument(
        "cm5_host",
        nargs="?",
        default="127.0.0.1",
        help="CM5 IP address or hostname (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run the brain and camera on the CM5 beside its safety bridge",
    )
    parser.add_argument(
        "--intent",
        default=DEFAULT_SITUATION,
        help=(
            "initial situation for Gemini (default: "
            f"{DEFAULT_SITUATION})"
        ),
    )
    parser.add_argument(
        "--voice-once",
        action="store_true",
        help="capture one push-to-talk utterance before starting flight control",
    )
    parser.add_argument(
        "--dialogue",
        action="store_true",
        help="accept typed dialogue while flight control is running",
    )
    parser.add_argument("--whisper-model", default="tiny.en")
    parser.add_argument("--record-duration", type=float, default=3.0)
    parser.add_argument(
        "--memory",
        default="~/.companion/memory.txt",
        help="editable experience-memory file",
    )
    parser.add_argument("--command-port", type=int, default=5001)
    parser.add_argument("--video-port", type=int, default=5000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--framerate", type=int, default=30)
    return parser


async def run(args):
    voice_request = None
    if args.voice_once:
        from voice.recorder import PushToTalkRecorder
        from voice.transcriber import WhisperTranscriber

        def listen_once():
            recorder = PushToTalkRecorder(duration_s=args.record_duration)
            transcriber = WhisperTranscriber(model_size=args.whisper_model)
            return transcriber.transcribe(recorder.record()).strip()

        voice_request = await asyncio.to_thread(listen_once)

    video_config = H264StreamConfig(
        port=args.video_port,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
    )
    receiver = GStreamerH264Receiver(video_config)
    frame_reader = AsyncLatestFrameReader(receiver)
    camera_sender = (
        GStreamerH264Sender("127.0.0.1", video_config)
        if args.local
        else None
    )
    memory = CompanionMemory(args.memory)
    from control.gemini_brain import GeminiRuntime

    control = GeminiRuntime(
        situation=args.intent,
        memory=memory,
    )
    await control.start()
    body_host = "127.0.0.1" if args.local else args.cm5_host
    sender = UdpCommandSender(body_host, args.command_port)
    service = UdpControlService(
        control,
        sender,
        frame_reader.read,
        telemetry_provider=sender.telemetry,
    )
    stop_event = asyncio.Event()
    dialogue_stop = asyncio.Event()
    dialogue_input = (
        DialogueInput(voice_request)
        if args.dialogue or voice_request
        else None
    )
    dialogue_task = None
    if dialogue_input is not None:
        async def route_dialogue():
            while not dialogue_stop.is_set():
                message = dialogue_input.next()
                if message:
                    control.add_dialogue(message)
                try:
                    await asyncio.wait_for(dialogue_stop.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass

        dialogue_task = asyncio.create_task(route_dialogue())
    try:
        receiver.start()
        if camera_sender is not None:
            camera_sender.start()
        print(
            f"Companion ready: video :{video_config.port}, "
            f"commands {body_host}:{args.command_port}, "
            f"situation={args.intent}, Gemini ER 2."
        )
        if args.dialogue:
            dialogue_input.start()
        await service.run(stop_event)
    finally:
        receiver.close()
        if camera_sender is not None:
            camera_sender.close()
        dialogue_stop.set()
        if dialogue_task is not None:
            await dialogue_task
        control.close()
        await control.wait_closed()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(f"companion: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
