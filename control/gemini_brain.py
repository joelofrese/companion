"""Run Gemini Robotics ER as the companion's streaming brain."""

import asyncio
from collections import deque
from io import BytesIO
import math
import os
import time
from typing import Optional

from PIL import Image

from control.memory import CompanionMemory
from control.mind import ConsciousDecision, Telemetry, VisualObservation
from control.mind_motion import movement_command
from control.safety_limits import MAX_YAW_RATE_DEG_S, OBSTACLE_STOP_M
from control.velocity import VelocityCommand


DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"
DEFAULT_SITUATION = "Observe the indoor environment and decide what to do next."
VIDEO_PERIOD_S = 1.0
MIN_HEARTBEAT_PERIOD_S = 1.0
RESPONSE_TIMEOUT_S = 10.0
START_TIMEOUT_S = 20.0
RECONNECT_DELAY_S = 1.0
MIN_MOVE_S = 0.2
MAX_MOVE_S = 1.0
MIN_TURN_DEG = 15.0
MAX_TURN_DEG = 90.0
TURN_RATE_DEG_S = MAX_YAW_RATE_DEG_S
MAX_IMAGE_WIDTH = 640


class GeminiRuntime:
    """Give one streaming Gemini session a deliberately small body."""

    def __init__(
        self,
        situation: str = DEFAULT_SITUATION,
        model: str = DEFAULT_MODEL,
        memory: Optional[CompanionMemory] = None,
        api_key: Optional[str] = None,
    ):
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError("situation must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Gemini model must be a non-empty string")
        self.situation = situation.strip()
        self.model = model.strip()
        self.memory_store = memory
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._latest_frame = None
        self._telemetry = Telemetry()
        self._dialogue = deque()
        self._movement = "stop"
        self._movement_until_s = 0.0
        self._turn_direction = "stop"
        self._turn_until_s = 0.0
        self._movement_blocked = False
        self.latest_observation: Optional[VisualObservation] = None
        self.latest_decision: Optional[ConsciousDecision] = None
        self.latest_thought = ""
        self.latest_response = ""
        self.latest_action = "stop"
        self._last_heartbeat_at_s: Optional[float] = None
        self.latest_observation_duration_s: Optional[float] = None
        self.latest_decision_duration_s: Optional[float] = None
        self.observation_count = 0
        self.decision_count = 0
        self.turn_count = 0
        self.video_frame_count = 0
        self._response_parts = []
        self._response_thoughts = []
        self._action_summary = ""
        self.thought_count = 0
        self.thought_token_count = 0
        self._memory_sent = False
        self._bootstrap_pending = True
        self._session_handle: Optional[str] = None
        self.session_reconnect_count = 0
        self._reconnect_requested = False
        self._closed = asyncio.Event()
        self._frame_ready = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._task = None
        self._error: Optional[Exception] = None

    async def start(self):
        """Connect the persistent Gemini session before flight begins."""

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is required for Gemini")
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=START_TIMEOUT_S)
        except asyncio.TimeoutError as error:
            self.close()
            raise RuntimeError(
                "Gemini did not connect before the start timeout"
            ) from error
        if self._error is not None:
            raise RuntimeError(
                f"Gemini did not connect: {self._error}"
            ) from self._error

    def add_dialogue(self, message: str):
        """Deliver one user request in the next model heartbeat."""

        if not isinstance(message, str) or not message.strip():
            return
        self._dialogue.append(message.strip())

    def tick(
        self,
        frame,
        timestamp_s: float,
        telemetry: Telemetry = Telemetry(),
    ) -> VelocityCommand:
        """Store fresh state and return the current bounded action."""

        if self._closed.is_set() or self._error is not None:
            return VelocityCommand()
        if frame is not None:
            self._latest_frame = frame
            self._frame_ready.set()
        self._telemetry = telemetry
        now = time.monotonic()
        if now >= self._movement_until_s:
            self._movement = "stop"
        if now >= self._turn_until_s:
            self._turn_direction = "stop"
        if any(
            value is None or not math.isfinite(value)
            for value in (
                telemetry.forward_velocity_m_s,
                telemetry.right_velocity_m_s,
                telemetry.down_velocity_m_s,
            )
        ):
            command = VelocityCommand()
        else:
            movement = movement_command(self._movement, telemetry.obstacle_distance_m)
            yaw_rate = 0.0
            if (
                self._turn_direction != "stop"
                and telemetry.obstacle_distance_m is not None
                and math.isfinite(telemetry.obstacle_distance_m)
                and telemetry.obstacle_distance_m > OBSTACLE_STOP_M
            ):
                yaw_rate = TURN_RATE_DEG_S * (
                    1.0 if self._turn_direction == "right" else -1.0
                )
            command = VelocityCommand(
                movement.forward_m_s,
                movement.right_m_s,
                movement.down_m_s,
                yaw_rate,
            )
        if self._movement != "stop" and command == VelocityCommand():
            self._movement_blocked = True
        return command

    def close(self):
        """Stop movement and end the streaming session."""

        self._movement = "stop"
        self._movement_until_s = 0.0
        self._turn_direction = "stop"
        self._turn_until_s = 0.0
        self._closed.set()
        self._frame_ready.set()

    async def wait_closed(self):
        """Wait for the session task after closing it."""

        if self._task is not None:
            await self._task

    async def _run(self):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            connected = False
            while not self._closed.is_set():
                self._reconnect_requested = False
                if self._session_handle is None:
                    self._memory_sent = False
                    self._bootstrap_pending = True
                try:
                    config = types.LiveConnectConfig(
                        response_modalities=["TEXT"],
                        tools=_tools(),
                        system_instruction=_system_instruction(),
                        thinking_config=types.ThinkingConfig(include_thoughts=True),
                        context_window_compression=(
                            types.ContextWindowCompressionConfig(
                                sliding_window=types.SlidingWindow()
                            )
                        ),
                        session_resumption=types.SessionResumptionConfig(
                            handle=self._session_handle
                        ),
                    )
                    async with client.aio.live.connect(
                        model=self.model,
                        config=config,
                    ) as session:
                        connected = True
                        self._ready.set()
                        await self._frame_ready.wait()
                        if self._closed.is_set():
                            return
                        await self._send_frame(session, types)
                        video_task = asyncio.create_task(
                            self._stream_video(session, types)
                        )
                        try:
                            while (
                                not self._closed.is_set()
                                and not self._reconnect_requested
                            ):
                                sent_at_s = time.monotonic()
                                await self._heartbeat(session)
                                try:
                                    await asyncio.wait_for(
                                        self._receive(session, types),
                                        timeout=RESPONSE_TIMEOUT_S,
                                    )
                                except asyncio.TimeoutError:
                                    pass
                                if video_task.done():
                                    await video_task
                                remaining_s = MIN_HEARTBEAT_PERIOD_S - (
                                    time.monotonic() - sent_at_s
                                )
                                if remaining_s > 0.0:
                                    try:
                                        await asyncio.wait_for(
                                            self._closed.wait(), timeout=remaining_s
                                        )
                                    except asyncio.TimeoutError:
                                        pass
                        finally:
                            video_task.cancel()
                            await asyncio.gather(video_task, return_exceptions=True)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if not connected:
                        raise error
                    if self._closed.is_set():
                        return
                    print(f"Gemini session reconnecting: {error}", flush=True)
                if self._closed.is_set():
                    return
                self.session_reconnect_count += 1
                try:
                    await asyncio.wait_for(
                        self._closed.wait(), timeout=RECONNECT_DELAY_S
                    )
                except asyncio.TimeoutError:
                    pass
        except Exception as error:
            self._error = error
            self._ready.set()

    async def _stream_video(self, session, types):
        """Keep the Live session supplied with the newest camera frame."""

        while not self._closed.is_set():
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=VIDEO_PERIOD_S)
            except asyncio.TimeoutError:
                await self._send_frame(session, types)

    async def _send_frame(self, session, types):
        """Send one current JPEG without waiting for a model decision."""

        frame = self._latest_frame
        if frame is None:
            return
        image_bytes = await asyncio.to_thread(_jpeg, frame)
        async with self._send_lock:
            await session.send_realtime_input(
                video=types.Blob(data=image_bytes, mime_type="image/jpeg")
            )
        self.video_frame_count += 1

    async def _heartbeat(self, session):
        """Ask for one high-level decision while video keeps streaming."""

        self._last_heartbeat_at_s = time.monotonic()
        dialogue = self._dialogue.popleft() if self._dialogue else ""
        async with self._send_lock:
            await session.send_realtime_input(text=self._heartbeat_text(dialogue))

    def _heartbeat_text(self, dialogue: str) -> str:
        memory = ""
        if not self._memory_sent:
            self._memory_sent = True
            if self.memory_store is not None:
                memory = self.memory_store.context()
        start = ""
        if self._bootstrap_pending:
            self._bootstrap_pending = False
            start = f"[START]\nSituation: {self.situation}\n"
        state = (
            f"{start}[STATE]\n"
            f"Vehicle: {_telemetry_text(self._telemetry)}\n"
            "Inspect the latest image and current state. Decide for yourself whether "
            "to move, turn, hover, speak, or do nothing. Never infer that a move or "
            "turn is safe from the image alone. A heartbeat does not require a tool "
            "or a visible response."
        )
        if dialogue:
            state += f"\nUser: {dialogue}"
        if memory:
            state += f"\nMemory:\n{memory}"
        return state

    async def _receive(self, session, types):
        async for message in session.receive():
            update = message.session_resumption_update
            if update is not None and update.resumable and update.new_handle:
                self._session_handle = update.new_handle

            if message.go_away is not None:
                self._reconnect_requested = True

            usage = message.usage_metadata
            if usage is not None and usage.thoughts_token_count is not None:
                self.thought_token_count = usage.thoughts_token_count
            content = message.server_content
            if content is not None:
                turn = content.model_turn
                if turn is not None and turn.parts:
                    for part in turn.parts:
                        if not part.text:
                            continue
                        if getattr(part, "thought", False):
                            self._response_thoughts.append(part.text)
                        else:
                            self._response_parts.append(part.text)
                else:
                    transcript = content.output_transcription
                    if transcript is not None and transcript.text:
                        self._response_parts.append(transcript.text)
                if content.turn_complete:
                    self.turn_count += 1
                    self._finish_turn()
            tool_call = message.tool_call
            if tool_call is not None:
                responses = []
                for call in tool_call.function_calls:
                    result = await self._execute(call.name, call.args or {})
                    responses.append(
                        types.FunctionResponse(
                            name=call.name,
                            response=result,
                            id=call.id,
                        )
                    )
                async with self._send_lock:
                    await session.send_tool_response(function_responses=responses)
            if message.tool_call_cancellation is not None:
                self._movement = "stop"
                self._movement_until_s = 0.0
                self._record_action("cancelled the current action")

            if self._reconnect_requested:
                self._response_parts.clear()
                self._response_thoughts.clear()
                self._action_summary = ""
                return

    async def _execute(self, name: str, args: dict) -> dict:
        if name == "move":
            return await self._move(args)
        if name == "turn":
            return await self._turn(args)
        if name == "hover":
            self._movement = "stop"
            self._movement_until_s = 0.0
            self._turn_direction = "stop"
            self._turn_until_s = 0.0
            self._record_action("hover")
            return {"status": "hovering", "telemetry": _telemetry_text(self._telemetry)}
        if name == "speak":
            message = str(args.get("message", "")).strip()
            if not message:
                return {"status": "rejected", "reason": "message is required"}
            self._record_action(f"speak: {message}")
            print(f"Companion: {message}", flush=True)
            return {"status": "spoken"}
        return {"status": "rejected", "reason": "unknown tool"}

    async def _move(self, args: dict) -> dict:
        direction = str(args.get("direction", "")).strip().lower()
        duration_s = args.get("duration_s")
        if direction not in ("forward", "left", "right"):
            return {
                "status": "rejected",
                "reason": "direction must be forward, left, or right",
            }
        if (
            isinstance(duration_s, bool)
            or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s)
            or not MIN_MOVE_S <= duration_s <= MAX_MOVE_S
        ):
            return {
                "status": "rejected",
                "reason": f"duration_s must be {MIN_MOVE_S} to {MAX_MOVE_S}",
            }
        self._movement = direction
        self._movement_until_s = time.monotonic() + duration_s
        self._movement_blocked = False
        self._record_action(f"move {direction} for {duration_s:.1f}s")
        return {
            "status": "accepted",
            "expires_in_s": duration_s,
            "telemetry": _telemetry_text(self._telemetry),
        }

    async def _turn(self, args: dict) -> dict:
        direction = str(args.get("direction", "")).strip().lower()
        angle_deg = args.get("angle_deg")
        if direction not in ("left", "right"):
            return {
                "status": "rejected",
                "reason": "direction must be left or right",
            }
        if (
            isinstance(angle_deg, bool)
            or not isinstance(angle_deg, (int, float))
            or not math.isfinite(angle_deg)
            or not MIN_TURN_DEG <= angle_deg <= MAX_TURN_DEG
        ):
            return {
                "status": "rejected",
                "reason": f"angle_deg must be {MIN_TURN_DEG} to {MAX_TURN_DEG}",
            }
        self._turn_direction = direction
        self._turn_until_s = time.monotonic() + angle_deg / TURN_RATE_DEG_S
        self._record_action(f"turn {direction} {angle_deg:.0f} degrees")
        return {
            "status": "accepted",
            "expires_in_s": angle_deg / TURN_RATE_DEG_S,
            "telemetry": _telemetry_text(self._telemetry),
        }

    def _record_action(self, action: str):
        action = " ".join(str(action).split())
        if action:
            self._action_summary = (
                f"{self._action_summary}; {action}"
                if self._action_summary
                else action
            )

    def _finish_turn(self):
        thought = " ".join("".join(self._response_thoughts).split())
        response = " ".join("".join(self._response_parts).split())
        action = self._action_summary or "none"
        self._response_thoughts.clear()
        self._response_parts.clear()
        self._action_summary = ""
        self.latest_thought = thought
        self.latest_response = response
        self.latest_action = action
        summary = thought or response or action
        now = time.monotonic()
        latency = (
            max(0.0, now - self._last_heartbeat_at_s)
            if self._last_heartbeat_at_s is not None
            else None
        )
        self.latest_observation = VisualObservation(
            timestamp_s=now,
            description=summary,
            movement=self._movement,
            confidence=1.0,
        )
        self.latest_decision = ConsciousDecision(
            intent="",
            dialogue=response,
            summary=summary,
        )
        self.latest_observation_duration_s = latency
        self.latest_decision_duration_s = latency
        self.observation_count += 1
        self.decision_count += 1
        if thought:
            self.thought_count += 1
            print(f"Gemini thought: {thought}", flush=True)
        if response:
            print(f"Gemini response: {response}", flush=True)
        if self.memory_store is not None:
            self.memory_store.remember(
                f"{_telemetry_text(self._telemetry)}; summary={summary}"
            )


def _tools():
    """Return the high-level actions exposed to Gemini."""

    return [{"function_declarations": [
        {
            "name": "move",
            "description": "Move slowly in one body direction for a short time.",
            "behavior": "NON_BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["forward", "left", "right"],
                    },
                    "duration_s": {
                        "type": "NUMBER",
                        "description": "A duration from 0.2 through 1.0 seconds.",
                    },
                },
                "required": ["direction", "duration_s"],
            },
        },
        {
            "name": "turn",
            "description": "Turn in place slowly to look in another direction.",
            "behavior": "NON_BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["left", "right"],
                    },
                    "angle_deg": {
                        "type": "NUMBER",
                        "description": "A turn from 15 through 90 degrees.",
                    },
                },
                "required": ["direction", "angle_deg"],
            },
        },
        {
            "name": "hover",
            "description": "Stop horizontal motion and hold position.",
            "behavior": "NON_BLOCKING",
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "speak",
            "description": "Say one short message to the nearby user.",
            "behavior": "NON_BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {"message": {"type": "STRING"}},
                "required": ["message"],
            },
        },
    ]}]


def _system_instruction() -> str:
    """State the safety and control contract in plain language."""

    return (
        "You are the high-level brain of an indoor companion drone. Observe the "
        "latest camera image, user dialogue, telemetry, and previous outputs. "
        "Think broadly, but move slowly and deliberately. You have optional tools "
        "for brief "
        "forward or lateral translation, turning in place, hovering, and speech. "
        "Choose freely "
        "whether to use one, use several, or use none; a heartbeat is not a "
        "requirement to act or speak. The CM5 checks every physical action and "
        "may stop it. "
        "Never request altitude, motors, attitude, position, or a long translation. "
        "Turn only through the turn tool. Use hover when stopping is appropriate "
        "or the scene or telemetry is unclear."
    )


def _jpeg(frame) -> bytes:
    """Turn one Gazebo BGR frame into the bounded JPEG sent to Gemini."""

    image = Image.fromarray(frame[:, :, ::-1])
    if image.width > MAX_IMAGE_WIDTH:
        image.thumbnail((MAX_IMAGE_WIDTH, image.height))
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def _telemetry_text(telemetry: Telemetry) -> str:
    """Return the small telemetry record useful to the model and memory."""

    command = telemetry.last_command or VelocityCommand()
    return "; ".join(
        (
            f"obstacle={_number(telemetry.obstacle_distance_m)}",
            "command=" + ",".join(
                _number(value)
                for value in (
                    command.forward_m_s,
                    command.right_m_s,
                    command.down_m_s,
                    command.yaw_rate_deg_s,
                )
            ),
            "velocity=" + ",".join(
                _number(value)
                for value in (
                    telemetry.forward_velocity_m_s,
                    telemetry.right_velocity_m_s,
                    telemetry.down_velocity_m_s,
                )
            ),
        )
    )


def _number(value) -> str:
    """Format one finite telemetry value without pretending unknown is zero."""

    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2f}"
    return "?"
