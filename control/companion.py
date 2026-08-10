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


DEFAULT_INTENT = "explore the surroundings"


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
        default=DEFAULT_INTENT,
        help=(
            "initial situation for Gemini or intent for the local fallback "
            f"(default: {DEFAULT_INTENT})"
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
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vlm-model", default="moondream")
    parser.add_argument("--llm-model", default="moondream")
    parser.add_argument("--ollama-timeout", type=float, default=60.0)
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="use separate local Ollama VLM and LLM sessions instead of Gemini",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-robotics-er-2-streaming-preview",
    )
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
    use_gemini = not args.ollama
    if use_gemini:
        from control.gemini_brain import GeminiRuntime

        control = GeminiRuntime(
            situation=args.intent,
            model=args.gemini_model,
            memory=memory,
        )
        await control.start()
    else:
        from control.mind import CompanionMind
        from control.mind_runtime import MindRuntime
        from control.ollama_brain import OllamaLanguageModel, OllamaVisionModel
        from control.ollama_client import OllamaClient

        client = OllamaClient(args.ollama_url, timeout_s=args.ollama_timeout)
        await asyncio.to_thread(client.check)
        await asyncio.to_thread(client.preload, args.vlm_model)
        await asyncio.to_thread(client.preload, args.llm_model)
        brain = CompanionMind(
            OllamaVisionModel(client, args.vlm_model),
            OllamaLanguageModel(client, args.llm_model),
            memory=memory,
        )
        brain.set_intent(args.intent)
        control = MindRuntime(brain)
    body_host = "127.0.0.1" if args.local else args.cm5_host
    sender = UdpCommandSender(body_host, args.command_port)
    service = UdpControlService(
        control,
        sender,
        frame_reader.read,
        telemetry_provider=sender.telemetry,
    )
    stop_event = asyncio.Event()
    mind_stop = asyncio.Event()
    dialogue_input = (
        DialogueInput(voice_request)
        if args.dialogue or voice_request
        else None
    )
    mind_task = None
    if use_gemini and dialogue_input is not None:
        async def route_dialogue():
            while not mind_stop.is_set():
                message = dialogue_input.next()
                if message:
                    control.add_dialogue(message)
                try:
                    await asyncio.wait_for(mind_stop.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    pass

        mind_task = asyncio.create_task(route_dialogue())
    elif not use_gemini:
        mind_task = asyncio.create_task(
            control.think_loop(
                mind_stop,
                telemetry_provider=sender.telemetry,
                dialogue_provider=dialogue_input.next if dialogue_input else None,
            )
        )
    try:
        receiver.start()
        if camera_sender is not None:
            camera_sender.start()
        model_name = (
            f"Gemini={args.gemini_model}"
            if use_gemini
            else f"VLM={args.vlm_model}, LLM={args.llm_model}"
        )
        context_name = (
            f"situation={args.intent}"
            if use_gemini
            else f"intent={args.intent}"
        )
        print(
            f"Companion ready: video :{video_config.port}, "
            f"commands {body_host}:{args.command_port}, {context_name}, "
            f"{model_name}."
        )
        if args.dialogue:
            dialogue_input.start()
        await service.run(stop_event)
    finally:
        receiver.close()
        if camera_sender is not None:
            camera_sender.close()
        mind_stop.set()
        if mind_task is not None:
            await mind_task
        control.close()
        if use_gemini:
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
