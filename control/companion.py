"""Run the Mac brain with local VLM and LLM models."""

import argparse
import asyncio

from control.mind import CompanionMemory, MacMind
from control.mind_runtime import MindRuntime
from control.ollama_brain import OllamaClient, OllamaLanguageModel, OllamaVisionModel
from control.dialogue import DialogueInput
from control.udp_control import UdpControlService
from control.udp_sender import UdpCommandSender
from vision.video_stream import AsyncLatestFrameReader, GStreamerH264Receiver, H264StreamConfig


def build_parser():
    parser = argparse.ArgumentParser(description="Run the Mac-side Companion control stack")
    parser.add_argument("cm5_host", help="CM5 IP address or hostname")
    parser.add_argument(
        "--intent",
        default="hover",
        help="initial high-level intent in plain language (default: hover)",
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
    parser.add_argument("--llm-model", default="gemma3:4b")
    parser.add_argument("--ollama-timeout", type=float, default=60.0)
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
    intent = args.intent
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
            intent = voice_state

    client = OllamaClient(args.ollama_url, timeout_s=args.ollama_timeout)
    await asyncio.to_thread(client.check)
    await asyncio.to_thread(client.preload, args.vlm_model)
    await asyncio.to_thread(client.preload, args.llm_model)
    video_config = H264StreamConfig(
        port=args.video_port,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
    )
    receiver = GStreamerH264Receiver(video_config)
    frame_reader = AsyncLatestFrameReader(receiver)
    brain = MacMind(
        OllamaVisionModel(client, args.vlm_model),
        OllamaLanguageModel(client, args.llm_model),
        memory=CompanionMemory(args.memory),
    )
    brain.set_intent(intent)
    control = MindRuntime(brain)
    sender = UdpCommandSender(args.cm5_host, args.command_port)
    service = UdpControlService(
        control,
        sender,
        frame_reader.read,
        telemetry_provider=sender.telemetry,
    )
    stop_event = asyncio.Event()
    mind_stop = asyncio.Event()
    dialogue_input = DialogueInput() if args.dialogue else None
    mind_task = asyncio.create_task(
        control.think_loop(
            mind_stop,
            telemetry_provider=sender.telemetry,
            dialogue_provider=dialogue_input.next if dialogue_input else None,
        )
    )
    try:
        receiver.start()
        print(
            f"Companion ready: video :{video_config.port}, "
            f"commands {args.cm5_host}:{args.command_port}, intent={intent}, "
            f"VLM={args.vlm_model}, LLM={args.llm_model}."
        )
        if dialogue_input:
            dialogue_input.start()
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
