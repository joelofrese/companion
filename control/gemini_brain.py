"""Run Gemini Robotics ER as the companion's streaming brain."""

import asyncio
from collections import deque
from dataclasses import dataclass
from io import BytesIO
import math
import os
import time
from typing import Optional

from PIL import Image

from control.memory import CompanionMemory
from control.safety_limits import OBSTACLE_STOP_M
from control.telemetry import Telemetry
from control.velocity import VelocityCommand


DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"
DEFAULT_SITUATION = "Explore the indoor surroundings autonomously."
# Give the streaming model a fresh view often enough for short closed-loop moves.
VIDEO_PERIOD_S = 1.0
# Allow a slow ER2 decision to finish before recovering a silent session.
RESPONSE_TIMEOUT_S = 30.0
START_TIMEOUT_S = 20.0
INITIAL_CONNECT_RETRIES = 1
RECONNECT_DELAY_S = 1.0
MIN_MOVE_S = 0.2
MAX_MOVE_S = 2.0
MAX_FORWARD_SPEED_M_S = 0.25
MAX_RIGHT_SPEED_M_S = 0.20
MIN_TURN_S = 0.25
# A short pulse lets the model look again instead of estimating an exact angle.
DEFAULT_TURN_S = 0.75
MAX_TURN_S = 2.0
# A full half-turn is enough to scan the surroundings before translating.
MAX_IN_PLACE_TURN_DEG = 180.0
# Keep the yaw rate low enough for PX4 to settle near the requested heading.
TURN_RATE_DEG_S = 8.0
MIN_TURN_RATE_DEG_S = 1.5
TURN_SLOW_THRESHOLD_DEG = 5.0
MAX_IMAGE_WIDTH = 640
# PX4 may take longer than the commanded yaw rate to settle on a heading.
ACTION_GRACE_S = 5.0
ACTION_SETTLE_S = 1.0
MOVE_SETTLE_S = 0.5
ACTION_STABLE_S = 0.3
# Do not resume an action after safety has held it for too long.
ACTION_SAFETY_HOLD_S = 0.5
HEADING_STABILITY_RAD = math.radians(2.0)
HEADING_TOLERANCE_RAD = math.radians(2.0)
MAX_FRAME_AGE_S = 1.5


@dataclass
class ActiveAction:
    """One physical action that must finish before another can start."""

    kind: str
    direction: str
    duration_s: float
    deadline_s: float
    start_heading_rad: Optional[float] = None
    phase: str = "running"
    stable_since_s: Optional[float] = None
    last_heading_rad: Optional[float] = None
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    yaw_rate_deg_s: float = 0.0
    last_update_s: Optional[float] = None
    last_sample_s: Optional[float] = None
    blocked_since_s: Optional[float] = None
    observed_forward_m: float = 0.0
    observed_right_m: float = 0.0
    completion: Optional[asyncio.Future] = None


class GeminiRuntime:
    """Give one streaming Gemini session a deliberately small body."""

    def __init__(
        self,
        situation: str = DEFAULT_SITUATION,
        memory: Optional[CompanionMemory] = None,
        api_key: Optional[str] = None,
    ):
        if not isinstance(situation, str) or not situation.strip():
            raise ValueError("situation must be a non-empty string")
        self.situation = situation.strip()
        self.memory_store = memory
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._latest_frame = None
        self._latest_frame_at_s: Optional[float] = None
        self._last_frame_sent_at_s: Optional[float] = None
        self._frame_count = 0
        self._last_frame_sent_count = 0
        self._telemetry = Telemetry()
        self._dialogue = deque()
        self._dialogue_in_flight: Optional[str] = None
        self._dialogue_send_complete = False
        self._latest_user_request = self.situation
        self._speech_blocked = False
        self._active_action: Optional[ActiveAction] = None
        self._in_place_turn_deg = 0.0
        self._action_finished_at_s: Optional[float] = None
        self._stop_requested = False
        self._last_action_result = ""
        self.latest_thought = ""
        self.latest_response = ""
        self.latest_action = "stop"
        self.latest_turn_duration_s: Optional[float] = None
        self.action_count = 0
        self.dialogue_sent_count = 0
        self.dialogue_count = 0
        self.video_frame_count = 0
        self._response_parts = []
        self._response_thoughts = []
        self._actions = []
        self.thought_count = 0
        self.thought_token_count = 0
        self._memory_sent = False
        self._bootstrap_pending = True
        self._session_handle: Optional[str] = None
        self._session = None
        self.session_reconnect_count = 0
        self._reconnect_requested = False
        self._last_model_activity_s: Optional[float] = None
        self._response_in_flight = False
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
        message = message.strip()
        self._latest_user_request = message
        self._speech_blocked = False
        if _is_explicit_stop(message):
            self._stop_requested = True
            self._cancel_action("explicit stop request")
        if self._dialogue_in_flight is not None:
            self._dialogue.clear()
            self._dialogue_in_flight = None
            self._dialogue_send_complete = False
        self._dialogue.append(message)

    def request_reconnect(self):
        """Reconnect the live session while stopping any active movement."""

        if self._closed.is_set():
            return
        self._stop_requested = False
        self._cancel_action("Gemini session reconnecting")
        self._reconnect_requested = True
        self._close_session()

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
            self._latest_frame_at_s = time.monotonic()
            self._frame_count += 1
            self._frame_ready.set()
        self._telemetry = telemetry
        self._refresh_action()
        if not self._has_fresh_frame():
            self._cancel_action("camera frame stale")
            return VelocityCommand()
        action = self._active_action
        if any(
            value is None or not math.isfinite(value)
            for value in (
                telemetry.forward_velocity_m_s,
                telemetry.right_velocity_m_s,
                telemetry.down_velocity_m_s,
            )
        ):
            return VelocityCommand()
        if action is None or action.phase != "running":
            return VelocityCommand()
        if action.kind == "move":
            if _obstacle_is_clear(telemetry.obstacle_distance_m):
                return VelocityCommand(
                    forward_m_s=action.forward_m_s,
                    right_m_s=action.right_m_s,
                    yaw_rate_deg_s=action.yaw_rate_deg_s,
                )
            return VelocityCommand()
        if (
            action.kind == "turn"
            and action.phase == "running"
            and _finite(telemetry.heading_rad)
            and _obstacle_is_valid(telemetry.obstacle_distance_m)
        ):
            yaw_rate = self._turn_rate(action)
            if action.direction == "left":
                yaw_rate = -yaw_rate
            return VelocityCommand(yaw_rate_deg_s=yaw_rate)
        return VelocityCommand()

    def close(self):
        """Stop movement and end the streaming session."""

        self._stop_requested = False
        self._cancel_action("brain closed")
        self._closed.set()
        self._frame_ready.set()
        self._close_session()

    def _close_session(self):
        if self._session is not None:
            asyncio.create_task(self._session.close())

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
            initial_connect_attempts = 0
            while not self._closed.is_set():
                self._reconnect_requested = False
                self._session = None
                if self._session_handle is None:
                    self._memory_sent = False
                    self._bootstrap_pending = True
                    self._dialogue_in_flight = None
                    self._dialogue_send_complete = False
                self._response_in_flight = False
                try:
                    config = types.LiveConnectConfig(
                        response_modalities=["TEXT"],
                        temperature=0.0,
                        tools=_tools(),
                        system_instruction=_system_instruction(),
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
                        model=DEFAULT_MODEL,
                        config=config,
                    ) as session:
                        self._session = session
                        self._last_frame_sent_at_s = None
                        connected = True
                        self._ready.set()
                        await self._frame_ready.wait()
                        if self._closed.is_set():
                            return
                        heartbeat_task = asyncio.create_task(
                            self._heartbeat_loop(session, types)
                        )
                        try:
                            receive_task = None
                            response_started_s = None
                            while (
                                not self._closed.is_set()
                                and not self._reconnect_requested
                            ):
                                if receive_task is None:
                                    response_started_s = time.monotonic()
                                    self._last_model_activity_s = response_started_s
                                    receive_task = asyncio.create_task(
                                        self._receive(
                                            session,
                                            types,
                                            response_started_s,
                                        )
                                    )
                                done, _ = await asyncio.wait(
                                    (heartbeat_task, receive_task),
                                    timeout=VIDEO_PERIOD_S,
                                    return_when=asyncio.FIRST_COMPLETED,
                                )
                                if heartbeat_task in done:
                                    await heartbeat_task
                                    break
                                if receive_task in done:
                                    await receive_task
                                    receive_task = None
                                    response_started_s = None
                                    self._last_model_activity_s = None
                                    if self._reconnect_requested:
                                        break
                                elif (
                                    response_started_s is not None
                                    and self._active_action is None
                                    and (
                                        time.monotonic()
                                        - (
                                            self._last_model_activity_s
                                            or response_started_s
                                        ) > RESPONSE_TIMEOUT_S
                                    )
                                ):
                                    print(
                                        "Gemini response stalled; reconnecting the session.",
                                        flush=True,
                                    )
                                    receive_task.cancel()
                                    await asyncio.gather(
                                        receive_task,
                                        return_exceptions=True,
                                    )
                                    self._response_parts.clear()
                                    self._response_thoughts.clear()
                                    self._actions.clear()
                                    self._last_model_activity_s = None
                                    # A silent turn is not a healthy resumable session.
                                    # Restart with the situation and compact memory.
                                    self._session_handle = None
                                    self._memory_sent = False
                                    self._bootstrap_pending = True
                                    self._reconnect_requested = True
                                    break
                        finally:
                            tasks = [heartbeat_task]
                            if receive_task is not None:
                                tasks.append(receive_task)
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(
                                *tasks,
                                return_exceptions=True,
                            )
                            self._last_model_activity_s = None
                        if self._dialogue_in_flight is not None:
                            self._dialogue_in_flight = None
                            self._dialogue_send_complete = False
                        self._session = None
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._session = None
                    if self._closed.is_set():
                        return
                    if self._session_handle and _resume_rejected(error):
                        self._session_handle = None
                        self._memory_sent = False
                        self._bootstrap_pending = True
                        print(
                            "Gemini session resumption rejected; starting a fresh session.",
                            flush=True,
                        )
                    if not connected:
                        if initial_connect_attempts >= INITIAL_CONNECT_RETRIES:
                            raise error
                        initial_connect_attempts += 1
                        print(
                            f"Gemini initial connection failed; retrying: {error}",
                            flush=True,
                        )
                    else:
                        self._cancel_action("Gemini session reconnecting")
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

    async def _send_frame(self, session, types):
        """Send one current JPEG without waiting for a model decision."""

        frame = self._latest_frame
        if frame is None or not self._has_fresh_frame():
            return
        image_bytes = await asyncio.to_thread(_jpeg, frame)
        async with self._send_lock:
            now = time.monotonic()
            if (
                self._last_frame_sent_at_s is not None
                and now - self._last_frame_sent_at_s < VIDEO_PERIOD_S
            ):
                return
            await session.send_realtime_input(
                video=types.Blob(data=image_bytes, mime_type="image/jpeg")
            )
            self._last_frame_sent_at_s = now
            self._last_frame_sent_count = self._frame_count
        self.video_frame_count += 1

    async def _heartbeat_loop(self, session, types):
        """Continuously prompt Gemini with the newest frame and state."""

        while not self._closed.is_set():
            await self._heartbeat(session, types)
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=VIDEO_PERIOD_S)
            except asyncio.TimeoutError:
                pass

    async def _heartbeat(self, session, types):
        """Send the current camera frame and state heartbeat."""

        await self._send_frame(session, types)
        # Keep frames continuous, but let the current model or blocking tool
        # cycle finish. State text is a new user turn and interrupts reasoning;
        # dialogue is the one intentional exception.
        if self._response_in_flight:
            if self._dialogue and self._dialogue_in_flight is None:
                action_result = self._last_action_result
                await self._send_dialogue(session, self._dialogue[0])
                if action_result and self._last_action_result == action_result:
                    self._last_action_result = ""
                if not self._memory_sent:
                    self._memory_sent = True
                if self._bootstrap_pending:
                    self._bootstrap_pending = False
            return
        dialogue = self._dialogue[0] if self._dialogue else ""
        action_result = self._last_action_result
        self._response_in_flight = True
        if dialogue:
            await self._send_dialogue(session, dialogue)
        else:
            async with self._send_lock:
                try:
                    await session.send_realtime_input(
                        text=self._heartbeat_text("")
                    )
                except Exception:
                    self._response_in_flight = False
                    raise
        # Repeat a completed action once in the next heartbeat so the state is
        # easy to see even when the model did not close its turn.
        if action_result and self._last_action_result == action_result:
            self._last_action_result = ""
        if not self._memory_sent:
            self._memory_sent = True
        if self._bootstrap_pending:
            self._bootstrap_pending = False

    async def _send_dialogue(self, session, dialogue: str):
        """Send user dialogue even when a model response is still running."""

        self._dialogue_in_flight = dialogue
        self._dialogue_send_complete = False
        try:
            async with self._send_lock:
                await session.send_realtime_input(
                    text=self._heartbeat_text(dialogue)
                )
        except Exception:
            self._response_in_flight = False
            if self._dialogue_in_flight == dialogue:
                self._dialogue_in_flight = None
                self._dialogue_send_complete = False
            raise
        if self._dialogue_in_flight == dialogue:
            self._dialogue_send_complete = True
            self.dialogue_sent_count += 1
            self._last_model_activity_s = time.monotonic()

    def _heartbeat_text(self, dialogue: str) -> str:
        memory = ""
        if not self._memory_sent:
            if self.memory_store is not None:
                memory = self.memory_store.context()
        parts = []
        if self._bootstrap_pending:
            parts.append(f"[START] Situation: {self.situation}")
            if not dialogue and self._latest_user_request != self.situation:
                parts.append(f"Current request: {self._latest_user_request}")
        camera = (
            "fresh"
            if self._has_fresh_frame()
            else "stale"
        )
        speech = (
            "ready"
            if not self._speech_blocked
            else (
                "complete for the current dialogue; do not call speak again; "
                "choose move, turn, hover, or ack"
            )
        )
        action_state = self._action_state_text()
        parts.append(
            f"[STATE] camera={camera}; telemetry={_telemetry_text(self._telemetry)}; "
            f"action={action_state}; speech={speech}"
        )
        if dialogue:
            parts.append(f"[USER] {dialogue}")
        if memory:
            parts.append(
                "[MEMORY] Prior experience; verify it against the current image "
                f"and telemetry:\n{memory}"
            )
        if self._active_action is not None:
            heartbeat = (
                "A physical action is still running. Observe its progress, but do "
                "not call move or turn until its measured result is returned."
            )
        elif self._last_action_result:
            heartbeat = (
                "The last physical action finished. Inspect its result and the "
                "newest image, then choose the next tool."
            )
        else:
            heartbeat = "Inspect the newest image and state, then choose the next tool."
        parts.append(
            f"[HEARTBEAT] {heartbeat} Call one tool directly: move, turn, hover, "
            "speak, or ack. Use ack when no physical action is needed yet; if no "
            "safe action is clear, call hover."
        )
        return "\n".join(parts)

    def _has_fresh_frame(self) -> bool:
        return (
            self._latest_frame_at_s is not None
            and time.monotonic() - self._latest_frame_at_s <= MAX_FRAME_AGE_S
        )

    async def _receive(self, session, types, response_started_s):
        async for message in session.receive():
            if (
                message.server_content is not None
                or message.tool_call is not None
                or message.tool_call_cancellation is not None
            ):
                self._last_model_activity_s = time.monotonic()
            turn_complete = False
            update = message.session_resumption_update
            if update is not None and update.resumable and update.new_handle:
                self._session_handle = update.new_handle

            if message.go_away is not None:
                self._cancel_action("Gemini session reconnecting")
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
                            self.latest_thought = _model_text(self._response_thoughts)
                        else:
                            self._response_parts.append(part.text)
                            self.latest_response = _model_text(self._response_parts)
                else:
                    transcript = content.output_transcription
                    if transcript is not None and transcript.text:
                        self._response_parts.append(transcript.text)
                turn_complete = bool(
                    content.turn_complete or content.generation_complete
                )
                if content.interrupted:
                    # Keep completed tool effects, but let the next heartbeat
                    # start a clean decision cycle.
                    self._response_parts.clear()
                    self._response_thoughts.clear()
                    self._actions.clear()
                    self._response_in_flight = False
                    return
            tool_call = message.tool_call
            if tool_call is not None:
                responses = []
                for call in tool_call.function_calls:
                    args = call.args or {}
                    result = await self._execute(call.name, args)
                    if call.name in ("move", "turn") and result.get("status") in (
                        "completed",
                        "timed out before target",
                        "cancelled",
                    ):
                        fresh_frame = await self._wait_for_fresh_action_frame()
                        result["camera_frame"] = self._last_frame_sent_count
                        result["camera_observation"] = (
                            "a fresh camera frame captured after the action was "
                            "sent immediately before this result"
                            if fresh_frame
                            else "no fresh camera frame arrived before the result"
                        )
                        result["movement_tools"] = (
                            "available now"
                            if fresh_frame
                            else "unavailable until a fresh camera frame arrives"
                        )
                    responses.append(
                        types.FunctionResponse(
                            name=call.name,
                            response=result,
                            id=call.id,
                        )
                    )
                if responses:
                    async with self._send_lock:
                        await session.send_tool_response(
                            function_responses=responses
                        )
                    self._last_model_activity_s = time.monotonic()
            if message.tool_call_cancellation is not None:
                self._cancel_action("Gemini cancelled it")

            if self._reconnect_requested:
                self._response_parts.clear()
                self._response_thoughts.clear()
                self._actions.clear()
                return
            if turn_complete:
                self._acknowledge_dialogue()
                self._finish_turn(response_started_s)
                self._response_in_flight = False
                return

    def _acknowledge_dialogue(self):
        """Remove a user message only after Gemini completes its turn."""

        if not self._dialogue_send_complete or not self._dialogue_in_flight:
            return
        if self._dialogue and self._dialogue[0] == self._dialogue_in_flight:
            self._dialogue.popleft()
            self.dialogue_count += 1
        self._dialogue_in_flight = None
        self._dialogue_send_complete = False

    async def _execute(self, name: str, args: dict) -> dict:
        if name == "move":
            result = await self._move(args)
        elif name == "turn":
            result = await self._turn(args)
        elif name == "ack":
            self._record_action("ack")
            self.action_count += 1
            result = {
                "status": "acknowledged",
                "reason": "no physical action is needed yet; continue observing",
                "telemetry": _telemetry_text(self._telemetry),
            }
        elif name == "hover":
            if (
                self._active_action is None
                and not self._stop_requested
            ):
                result = {
                    "status": "already_hovering",
                    "reason": "the vehicle is already holding position",
                    "telemetry": _telemetry_text(self._telemetry),
                }
            elif self._active_action is not None and not self._stop_requested:
                result = {
                    "status": "unavailable",
                    "reason": (
                        "the physical action is still running; hover can "
                        "interrupt it only for an explicit stop request"
                    ),
                    "active_action": self._action_label(),
                    "movement_tools": "unavailable until the action completes",
                    "telemetry": _telemetry_text(self._telemetry),
                }
            else:
                self._stop_requested = False
                cancelled = self._cancel_action("hover")
                self._record_action("hover")
                result = {
                    "status": "hovering",
                    "cancelled_action": cancelled or "none",
                    "telemetry": _telemetry_text(self._telemetry),
                }
        elif name == "speak":
            message = str(args.get("message", "")).strip()
            if not message:
                result = {"status": "rejected", "reason": "message is required"}
            elif self._speech_blocked:
                result = {
                    "status": "already_spoken",
                    "reason": (
                        "a response was already spoken for this dialogue; do not "
                        "call speak again. Choose move, turn, hover, or ack. "
                        "Speech becomes available after new dialogue or a completed "
                        "physical action"
                    ),
                    "movement_tools": "available now",
                }
            else:
                self._record_action(f"speak: {message}")
                print(f"Companion: {message}", flush=True)
                self._speech_blocked = True
                result = {"status": "spoken"}
        else:
            result = {"status": "rejected", "reason": "unknown tool"}
        if result.get("status") in ("hovering", "spoken"):
            self.action_count += 1
        if result.get("status") in ("rejected", "unavailable", "already_spoken"):
            reason = str(result.get("reason", "")).strip()
            action = f"{name} {result['status']}"
            if reason:
                action += f": {reason}"
            self._record_action(action)
        return result

    async def _move(self, args: dict) -> dict:
        forward_m_s = _number_between(
            args,
            "forward_m_s",
            -MAX_FORWARD_SPEED_M_S,
            MAX_FORWARD_SPEED_M_S,
        )
        right_m_s = _number_between(
            args, "right_m_s", -MAX_RIGHT_SPEED_M_S, MAX_RIGHT_SPEED_M_S
        )
        duration_s = _number_between(args, "duration_s", MIN_MOVE_S, MAX_MOVE_S)
        yaw_rate_deg_s = 0.0
        if "yaw_rate_deg_s" in args:
            yaw_rate_deg_s = _number_between(
                args,
                "yaw_rate_deg_s",
                -TURN_RATE_DEG_S,
                TURN_RATE_DEG_S,
            )
        if (
            forward_m_s is None
            or right_m_s is None
            or duration_s is None
            or yaw_rate_deg_s is None
        ):
            return {
                "status": "rejected",
                "reason": (
                    "forward_m_s must be -"
                    f"{MAX_FORWARD_SPEED_M_S} to {MAX_FORWARD_SPEED_M_S}; "
                    "right_m_s must be "
                    f"-{MAX_RIGHT_SPEED_M_S} to {MAX_RIGHT_SPEED_M_S}; "
                    "yaw_rate_deg_s must be "
                    f"-{TURN_RATE_DEG_S} to {TURN_RATE_DEG_S}; "
                    f"duration_s must be {MIN_MOVE_S} to {MAX_MOVE_S}"
                ),
            }
        if forward_m_s == 0.0 and right_m_s == 0.0:
            return {
                "status": "rejected",
                "reason": "at least one body-frame velocity must be non-zero",
            }
        busy = self._busy_response()
        if busy is not None:
            return busy
        observation = self._observation_required_response()
        if observation is not None:
            return observation
        now = time.monotonic()
        action = ActiveAction(
            "move",
            _move_direction(forward_m_s, right_m_s),
            duration_s,
            now + duration_s,
            start_heading_rad=(
                self._telemetry.heading_rad
                if yaw_rate_deg_s and _finite(self._telemetry.heading_rad)
                else None
            ),
            forward_m_s=forward_m_s,
            right_m_s=right_m_s,
            yaw_rate_deg_s=yaw_rate_deg_s,
            last_update_s=now,
            last_sample_s=now,
            completion=asyncio.get_running_loop().create_future(),
        )
        self._active_action = action
        self._in_place_turn_deg = 0.0
        self._last_action_result = ""
        self.action_count += 1
        self._record_action(f"started {self._action_label(action)}")
        return await self._wait_for_action(action)

    async def _turn(self, args: dict) -> dict:
        direction = str(args.get("direction", "")).strip().lower()
        duration_s = args.get("duration_s")
        if direction not in ("left", "right"):
            return {
                "status": "rejected",
                "reason": "direction must be left or right",
            }
        if duration_s is None:
            duration_s = DEFAULT_TURN_S
        else:
            duration_s = _number_between(
                args, "duration_s", MIN_TURN_S, MAX_TURN_S
            )
        if duration_s is None:
            return {
                "status": "rejected",
                "reason": f"duration_s must be {MIN_TURN_S} to {MAX_TURN_S}",
            }
        busy = self._busy_response()
        if busy is not None:
            return busy
        observation = self._observation_required_response()
        if observation is not None:
            return observation
        now = time.monotonic()
        duration_s = float(duration_s)
        angle_deg = duration_s * TURN_RATE_DEG_S
        if self._in_place_turn_deg + angle_deg > MAX_IN_PLACE_TURN_DEG:
            remaining_deg = max(
                0.0,
                MAX_IN_PLACE_TURN_DEG - self._in_place_turn_deg,
            )
            return {
                "status": "unavailable",
                "reason": (
                    "the in-place visual sweep has reached its 180-degree limit; "
                    "translate before turning further"
                ),
                "movement_tools": "move and hover available; turn available after translation",
                "remaining_scan_deg": remaining_deg,
                "telemetry": _telemetry_text(self._telemetry),
            }
        action = ActiveAction(
            "turn",
            direction,
            duration_s,
            now + duration_s + ACTION_GRACE_S,
            start_heading_rad=(
                self._telemetry.heading_rad
                if _finite(self._telemetry.heading_rad)
                else None
            ),
            last_update_s=now,
            completion=asyncio.get_running_loop().create_future(),
        )
        self._active_action = action
        self._in_place_turn_deg += angle_deg
        self._last_action_result = ""
        self.action_count += 1
        self._record_action(f"started {self._action_label(action)}")
        return await self._wait_for_action(action)

    async def _wait_for_action(self, action: ActiveAction) -> dict:
        if action.completion is None:
            return {"status": "cancelled", "reason": "action had no completion handle"}
        return await asyncio.shield(action.completion)

    def _busy_response(self):
        self._refresh_action()
        action = self._active_action
        if action is None:
            if self._stop_requested:
                return {
                    "status": "unavailable",
                    "reason": "an explicit stop request is active; hover before moving again",
                    "movement_tools": "unavailable until hovering is acknowledged",
                    "telemetry": _telemetry_text(self._telemetry),
                }
            return None
        reason = (
            "safety telemetry is currently holding the action; it will resume "
            "when movement is permitted"
            if self._action_is_blocked(action)
            else "a physical movement action is still in progress"
        )
        return {
            "status": "unavailable",
            "reason": reason,
            "active_action": self._action_label(action),
            "phase": action.phase,
            "remaining_s": max(0.0, action.deadline_s - time.monotonic()),
            "movement_tools": "unavailable until the active action completes",
            "telemetry": _telemetry_text(self._telemetry),
        }

    def _observation_required_response(self):
        if (
            self._action_finished_at_s is not None
            and (
                self._latest_frame_at_s is None
                or self._latest_frame_at_s <= self._action_finished_at_s
            )
        ):
            return {
                "status": "unavailable",
                "reason": (
                    "wait for a fresh camera frame after the previous physical "
                    "action before choosing another movement"
                ),
                "movement_tools": "unavailable until a fresh frame arrives",
                "telemetry": _telemetry_text(self._telemetry),
            }
        return None

    def _translation_text(self, action: ActiveAction) -> str:
        return (
            "observed translation "
            f"forward={action.observed_forward_m:+.2f}m "
            f"right={action.observed_right_m:+.2f}m"
        )

    def _action_label(self, action: Optional[ActiveAction] = None) -> str:
        action = action or self._active_action
        if action is None:
            return "none"
        if action.kind == "move":
            label = (
                f"move forward={action.forward_m_s:+.2f}m/s "
                f"right={action.right_m_s:+.2f}m/s for {action.duration_s:.1f}s"
            )
            if action.yaw_rate_deg_s:
                label += f" yaw={action.yaw_rate_deg_s:+.1f}deg/s"
            return label
        return f"turn {action.direction} for {action.duration_s:.1f}s"

    def _action_state_text(self) -> str:
        self._refresh_action()
        action = self._active_action
        if action is None:
            if self._last_action_result:
                return (
                    f"{self._last_action_result}; movement tools available; "
                    f"in-place scan remaining="
                    f"{max(0.0, MAX_IN_PLACE_TURN_DEG - self._in_place_turn_deg):.0f} degrees"
                )
            return (
                "none; movement tools available; in-place scan remaining="
                f"{max(0.0, MAX_IN_PLACE_TURN_DEG - self._in_place_turn_deg):.0f} degrees"
            )
        details = [
            f"{self._action_label(action)}; {action.phase}",
            f"remaining={max(0.0, action.deadline_s - time.monotonic()):.1f}s",
        ]
        if action.kind == "move":
            details.append(self._translation_text(action))
        actual = self._heading_change_deg(action)
        if actual is not None:
            details.append(f"observed heading change={actual:+.1f} degrees")
        if self._action_is_blocked(action):
            details.append("paused until safety telemetry permits movement")
        details.append("move and turn tools unavailable until completion")
        details.append("hover may interrupt")
        return "; ".join(details)

    def _refresh_action(self):
        action = self._active_action
        if action is None:
            return
        now = time.monotonic()
        self._record_translation(action, now)
        if self._action_is_blocked(action):
            if action.blocked_since_s is None:
                action.blocked_since_s = now
            elif now - action.blocked_since_s >= ACTION_SAFETY_HOLD_S:
                self._cancel_action("safety hold persisted")
                return
        else:
            action.blocked_since_s = None
        self._pause_action_if_blocked(action, now)
        if action.kind == "turn":
            if action.start_heading_rad is None and _finite(self._telemetry.heading_rad):
                action.start_heading_rad = self._telemetry.heading_rad
            actual = self._heading_change_deg(action)
            if action.phase == "running" and actual is not None:
                requested_rad = math.radians(_turn_angle_deg(action))
                progress_rad = math.radians(actual)
                if progress_rad >= _turn_completion_rad(requested_rad):
                    action.phase = "settling"
                    action.stable_since_s = now
                    action.last_heading_rad = self._telemetry.heading_rad
                    action.deadline_s = max(
                        action.deadline_s,
                        now + ACTION_SETTLE_S,
                    )
                    return
            if action.phase == "settling":
                heading = self._telemetry.heading_rad
                if _finite(heading) and action.last_heading_rad is not None:
                    change_rad = abs(_angle_delta_rad(action.last_heading_rad, heading))
                    if change_rad > HEADING_STABILITY_RAD:
                        action.stable_since_s = now
                    action.last_heading_rad = heading
                    if (
                        action.stable_since_s is not None
                        and now - action.stable_since_s >= ACTION_STABLE_S
                    ):
                        self._complete_action("completed", actual)
                        return
        if (
            action.kind == "move"
            and action.phase == "running"
            and now >= action.deadline_s
        ):
            # Stop commanding motion, then let telemetry catch up before
            # reporting how far the vehicle actually moved.
            action.phase = "settling"
            action.deadline_s = now + MOVE_SETTLE_S
            return
        if now >= action.deadline_s:
            actual = self._heading_change_deg(action)
            if action.kind == "turn" and actual is not None:
                completion_rad = _turn_completion_rad(
                    math.radians(_turn_angle_deg(action))
                )
                status = (
                    "completed"
                    if math.radians(actual) >= completion_rad
                    else "timed out before target"
                )
                self._complete_action(status, actual)
            else:
                self._complete_action("completed")

    def _record_translation(self, action: ActiveAction, now: float):
        if action.kind != "move":
            return
        if action.last_sample_s is None:
            action.last_sample_s = now
            return
        elapsed = max(0.0, now - action.last_sample_s)
        if not self._action_is_blocked(action):
            if _finite(self._telemetry.forward_velocity_m_s):
                action.observed_forward_m += (
                    self._telemetry.forward_velocity_m_s * elapsed
                )
            if _finite(self._telemetry.right_velocity_m_s):
                action.observed_right_m += self._telemetry.right_velocity_m_s * elapsed
        action.last_sample_s = now

    def _pause_action_if_blocked(self, action: ActiveAction, now: float):
        """Do not spend an action's time while safety state holds it still."""

        if (
            action.last_update_s is not None
            and action.phase == "running"
            and self._action_is_blocked(action)
        ):
            action.deadline_s += max(0.0, now - action.last_update_s)
        action.last_update_s = now

    def _action_is_blocked(self, action: ActiveAction) -> bool:
        if not _obstacle_is_valid(self._telemetry.obstacle_distance_m):
            return True
        if action.kind != "turn" and not _obstacle_is_clear(
            self._telemetry.obstacle_distance_m
        ):
            return True
        if any(
            not _finite(value)
            for value in (
                self._telemetry.forward_velocity_m_s,
                self._telemetry.right_velocity_m_s,
                self._telemetry.down_velocity_m_s,
            )
        ):
            return True
        return action.kind == "turn" and not _finite(self._telemetry.heading_rad)

    def _heading_change_deg(self, action: ActiveAction) -> Optional[float]:
        if action.start_heading_rad is None or not _finite(self._telemetry.heading_rad):
            return None
        change_rad = _angle_delta_rad(
            action.start_heading_rad,
            self._telemetry.heading_rad,
        )
        if action.direction == "left":
            change_rad = -change_rad
        return math.degrees(change_rad)

    def _turn_rate(self, action: ActiveAction) -> float:
        """Slow the turn as measured heading approaches its requested angle."""

        actual = self._heading_change_deg(action)
        if actual is None:
            return TURN_RATE_DEG_S
        remaining = max(0.0, _turn_angle_deg(action) - actual)
        if remaining >= TURN_SLOW_THRESHOLD_DEG:
            return TURN_RATE_DEG_S
        return max(
            MIN_TURN_RATE_DEG_S,
            TURN_RATE_DEG_S * remaining / TURN_SLOW_THRESHOLD_DEG,
        )

    def _complete_action(self, status: str, actual_heading_deg: Optional[float] = None):
        action = self._active_action
        if action is None:
            return
        if actual_heading_deg is None and action.start_heading_rad is not None:
            actual_heading_deg = self._heading_change_deg(action)
        result = f"{self._action_label(action)} {status}"
        if actual_heading_deg is not None:
            result += f"; observed heading change {actual_heading_deg:+.1f} degrees"
        if action.kind == "move":
            result += f"; {self._translation_text(action)}"
        self._last_action_result = result
        self._action_finished_at_s = time.monotonic()
        if action.kind in ("move", "turn"):
            self._speech_blocked = False
        response = self._action_response(action, status, result, actual_heading_deg)
        self._record_action(result)
        self._remember_action(result)
        self._active_action = None
        if action.completion is not None and not action.completion.done():
            action.completion.set_result(response)

    def _cancel_action(self, reason: str) -> str:
        action = self._active_action
        if action is None:
            return ""
        result = f"{self._action_label(action)} cancelled by {reason}"
        actual = self._heading_change_deg(action)
        if actual is not None:
            result += f"; observed heading change {actual:+.1f} degrees"
        if action.kind == "move":
            result += f"; {self._translation_text(action)}"
        self._last_action_result = result
        self._action_finished_at_s = time.monotonic()
        self._speech_blocked = False
        response = self._action_response(action, "cancelled", result, actual)
        self._record_action(result)
        self._remember_action(result)
        self._active_action = None
        if action.completion is not None and not action.completion.done():
            action.completion.set_result(response)
        return self._action_label(action)

    def _action_response(
        self,
        action: ActiveAction,
        status: str,
        result: str,
        actual_heading_deg: Optional[float],
    ) -> dict:
        movement_tools = "available now"
        if (
            self._action_finished_at_s is not None
            and (
                self._latest_frame_at_s is None
                or self._latest_frame_at_s <= self._action_finished_at_s
            )
        ):
            movement_tools = "available after a fresh camera frame"
        response = {
            "status": status,
            "action": result,
            "heading_deg": _heading_value(self._telemetry.heading_rad),
            "telemetry": _telemetry_text(self._telemetry),
            "movement_tools": movement_tools,
        }
        if actual_heading_deg is not None:
            response["observed_heading_change_deg"] = actual_heading_deg
        if action.start_heading_rad is not None:
            response["heading_before_deg"] = _heading_value(action.start_heading_rad)
        if action.kind == "turn":
            response["requested_duration_s"] = action.duration_s
            response["expected_heading_change_deg"] = _turn_angle_deg(action)
            response["in_place_scan_remaining_deg"] = max(
                0.0,
                MAX_IN_PLACE_TURN_DEG - self._in_place_turn_deg,
            )
            response["visual_effect"] = (
                "the scene should have moved toward image-right after a left turn"
                if action.direction == "left"
                else "the scene should have moved toward image-left after a right turn"
            )
        if action.kind == "move":
            response["observed_translation_m"] = {
                "forward": action.observed_forward_m,
                "right": action.observed_right_m,
            }
            response["yaw_rate_deg_s"] = action.yaw_rate_deg_s
        return response

    def _fresh_frame_sent_after_action(self) -> bool:
        finished_at_s = self._action_finished_at_s
        return (
            finished_at_s is not None
            and self._latest_frame_at_s is not None
            and self._latest_frame_at_s > finished_at_s
            and self._last_frame_sent_at_s is not None
            and self._last_frame_sent_at_s > finished_at_s
        )

    async def _wait_for_fresh_action_frame(self) -> bool:
        deadline = time.monotonic() + 2.0 * VIDEO_PERIOD_S
        while (
            not self._fresh_frame_sent_after_action()
            and not self._closed.is_set()
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(0.05)
        return self._fresh_frame_sent_after_action()

    def _record_action(self, action: str):
        action = " ".join(str(action).split())
        if action:
            self._actions.append(action)
            self.latest_action = action

    def _remember_action(self, action: str):
        if self.memory_store is not None:
            self.memory_store.remember(
                f"{_telemetry_text(self._telemetry)}; action={action}"
            )

    def _finish_turn(self, response_started_s):
        thought = _model_text(self._response_thoughts)
        response = _model_text(self._response_parts)
        action = self._actions[-1] if self._actions else "none"
        self._response_thoughts.clear()
        self._response_parts.clear()
        self._actions.clear()
        self.latest_thought = thought
        self.latest_response = response
        self.latest_action = action
        if response and action == "none" and not self._last_action_result:
            self._last_action_result = (
                "No physical tool was called; the previous response did not move "
                "or speak. Call one available tool directly now."
            )
        summary = thought or response or action
        now = time.monotonic()
        self.latest_turn_duration_s = max(0.0, now - response_started_s)
        if thought:
            self.thought_count += 1
            print(f"Gemini thought: {thought}", flush=True)
        if response:
            print(f"Gemini response: {response}", flush=True)
        if (
            self.memory_store is not None
            and summary not in ("", "none")
            and action not in summary
        ):
            self.memory_store.remember(
                f"{_telemetry_text(self._telemetry)}; summary={summary}"
            )


def _tools():
    """Return the high-level actions exposed to Gemini."""

    return [{"function_declarations": [
        {
            "name": "move",
            "description": (
                "Move slowly in the body frame for a short duration. Forward is "
                "positive and right is positive. Use a clear path and valid range "
                "reading. When a visible subject is off-center and the path is clear, "
                "a small lateral velocity and yaw rate can make a smooth arc while "
                "keeping it in view. Inspect the next image and measured result after "
                "the move."
            ),
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "forward_m_s": {
                        "type": "NUMBER",
                        "description": (
                            "Forward body velocity; negative is backward, from "
                            f"-{MAX_FORWARD_SPEED_M_S} through "
                            f"{MAX_FORWARD_SPEED_M_S} m/s."
                        ),
                        "minimum": -MAX_FORWARD_SPEED_M_S,
                        "maximum": MAX_FORWARD_SPEED_M_S,
                    },
                    "right_m_s": {
                        "type": "NUMBER",
                        "description": (
                            "Right body velocity from "
                            f"-{MAX_RIGHT_SPEED_M_S} through "
                            f"{MAX_RIGHT_SPEED_M_S} m/s."
                        ),
                        "minimum": -MAX_RIGHT_SPEED_M_S,
                        "maximum": MAX_RIGHT_SPEED_M_S,
                    },
                    "duration_s": {
                        "type": "NUMBER",
                        "description": (
                            f"A duration from {MIN_MOVE_S} through {MAX_MOVE_S} "
                            "seconds."
                        ),
                        "minimum": MIN_MOVE_S,
                        "maximum": MAX_MOVE_S,
                    },
                    "yaw_rate_deg_s": {
                        "type": "NUMBER",
                        "description": (
                            "Optional body yaw rate while translating; positive is "
                            f"right, from -{TURN_RATE_DEG_S} through "
                            f"{TURN_RATE_DEG_S} degrees per second."
                        ),
                        "minimum": -TURN_RATE_DEG_S,
                        "maximum": TURN_RATE_DEG_S,
                    },
                },
                "required": ["forward_m_s", "right_m_s", "duration_s"],
            },
        },
        {
            "name": "turn",
            "description": (
                "Apply a short, slow in-place yaw pulse. Choose the direction from "
                "the newest image and heading: image-left means left and image-right "
                "means right. Use a short duration, then inspect the new image and "
                "measured heading before choosing another pulse. Use move with a yaw "
                "rate when translating and turning together would make a smoother arc."
            ),
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["left", "right"],
                        "description": (
                            "Choose left when the target is on image-left and "
                            "right when it is on image-right. This is relative "
                            "to the current nose, not the room."
                        ),
                    },
                    "duration_s": {
                        "type": "NUMBER",
                        "description": (
                            f"Optional short yaw pulse from {MIN_TURN_S:.2f} "
                            f"through {MAX_TURN_S:.1f} seconds. Omit it for the "
                            f"normal {DEFAULT_TURN_S:.2f}-second pulse."
                        ),
                        "minimum": MIN_TURN_S,
                        "maximum": MAX_TURN_S,
                    },
                },
                "required": ["direction"],
            },
        },
        {
            "name": "hover",
            "description": (
                "Stop horizontal motion and hold position when the task is complete, "
                "while waiting, or when the scene is unclear."
            ),
            "behavior": "BLOCKING",
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "ack",
            "description": (
                "Acknowledge the newest image and telemetry when no physical action "
                "is needed yet. Keep observing and wait for a useful change instead "
                "of inventing movement."
            ),
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "speak",
            "description": (
                "Say one short message when the user asks or a meaningful new event "
                "is worth sharing. Do not announce a planned movement instead of "
                "calling move or turn; speak after the observation or action is "
                "real. After speaking, do not call speak again until new dialogue "
                "or a completed physical action."
            ),
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {"message": {"type": "STRING"}},
                "required": ["message"],
            },
        },
    ]}]


def _system_instruction() -> str:
    """State the control contract in plain language."""

    return (
        "You are the high-level brain of an indoor DEXI 3 companion drone. Use the "
        "newest camera image, TOF distance, body velocity, heading, active action, "
        "dialogue, measured result, and calibration memory. Use the tools directly; "
        "for each decision choose `move`, `turn`, `hover`, or `speak`; use `ack` "
        "when no physical action is needed yet. Keep the "
        "situation or request active until it is complete or changed. Seeing a target "
        "is not completion: when safe, continue toward it, verify the new view and "
        "measured state, then report completion.\n\n"
        "The camera faces forward: image-left and image-right are vehicle-left and "
        "vehicle-right. Move only with a clear path and valid TOF range. Prefer short, "
        "slow translation. If a visible subject is offset and the path is clear, use a "
        "small lateral velocity and yaw rate for a smooth arc. Turn only when the view "
        "or path needs reorientation. A turn is a short yaw pulse: look again after "
        "each pulse and correct from the new image and heading instead of estimating "
        "a large angle in advance. If the path is clear and TOF is well beyond the "
        "stop limit, translate instead of repeatedly turning in place. Hover when no "
        "safe step is clear. When a visible target is ahead on a clear path, keep it "
        "in view and translate toward it instead of scanning again. Do not repeat the "
        "same in-place direction without a new visual reason. Trust the newest image "
        "and telemetry over memory.\n\n"
        "Move and turn are blocking physical actions. Wait for their measured result "
        "before choosing another movement. A requested duration is intent, "
        "not proof; use observed translation, heading, telemetry, and action state to "
        "choose the next correction. Do not ask the user for an exact turn amount or "
        "narrate instead of acting. Speak after a real observation, event, or user request. "
        "The CM5 limits every physical command."
    )


def _jpeg(frame) -> bytes:
    """Turn one Gazebo BGR frame into the bounded JPEG sent to Gemini."""

    image = Image.fromarray(frame[:, :, ::-1])
    if image.width > MAX_IMAGE_WIDTH:
        image.thumbnail((MAX_IMAGE_WIDTH, image.height))
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def _model_text(parts) -> str:
    """Return useful model text without empty structured-output placeholders."""

    value = " ".join("".join(parts).split())
    if (
        value.startswith("[START]")
        or value.startswith("[STATE]")
        or value.casefold().startswith("# no action chosen")
    ):
        return ""
    if value.casefold().rstrip(".!?") in {
        "none",
        "no action",
        "no tool call necessary",
        "no tool call is necessary",
    }:
        return ""
    return value


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
            f"heading_deg={_heading_number(telemetry.heading_rad)}",
        )
    )


def _number(value) -> str:
    """Format one finite telemetry value without pretending unknown is zero."""

    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.2f}"
    return "?"


def _number_between(args: dict, name: str, minimum: float, maximum: float):
    """Return one finite numeric argument inside its allowed range."""

    value = args.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        return None
    return float(value)


def _move_direction(forward_m_s: float, right_m_s: float) -> str:
    """Give a movement a compact label for trace output."""

    if forward_m_s and right_m_s:
        return "move"
    if forward_m_s:
        return "forward" if forward_m_s > 0.0 else "backward"
    return "left" if right_m_s < 0.0 else "right"


def _obstacle_is_clear(distance_m: Optional[float]) -> bool:
    """Return whether the forward range reading permits movement."""

    return _obstacle_is_valid(distance_m) and distance_m > OBSTACLE_STOP_M


def _obstacle_is_valid(distance_m: Optional[float]) -> bool:
    """Return whether a forward range reading is usable."""

    return _finite(distance_m) and distance_m >= 0.0


def _heading_number(value) -> str:
    """Format one heading in degrees without hiding missing telemetry."""

    if _finite(value):
        return f"{math.degrees(value):.1f}"
    return "?"


def _heading_value(value):
    """Return a numeric heading for structured tool feedback."""

    return math.degrees(value) if _finite(value) else None


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_explicit_stop(message: str) -> bool:
    """Recognize the short dialogue commands that may interrupt movement."""

    message = " ".join(message.casefold().split())
    if message.startswith("please "):
        message = message[7:]
    return message in {
        "stop",
        "stop moving",
        "hover",
        "hold position",
        "cancel",
        "cancel movement",
    }


def _resume_rejected(error: Exception) -> bool:
    text = str(error).casefold()
    return "1007" in text or "invalid frame payload" in text


def _angle_delta_rad(start: float, end: float) -> float:
    """Return the signed shortest heading change from start to end."""

    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _turn_completion_rad(requested_rad: float) -> float:
    """Require meaningful progress even for the smallest allowed turn."""

    return max(requested_rad * 0.5, requested_rad - HEADING_TOLERANCE_RAD)


def _turn_angle_deg(action: ActiveAction) -> float:
    """Return the heading change expected from one timed yaw pulse."""

    return action.duration_s * TURN_RATE_DEG_S
