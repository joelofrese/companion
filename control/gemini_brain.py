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
from control.mind import ConsciousDecision, Telemetry, VisualObservation
from control.safety_limits import OBSTACLE_STOP_M
from control.velocity import VelocityCommand


DEFAULT_MODEL = "gemini-robotics-er-2-streaming-preview"
DEFAULT_SITUATION = "Observe the indoor environment and decide what to do next."
# Give the streaming model a fresh view often enough for short closed-loop moves.
VIDEO_PERIOD_S = 1.0
# Favor timely closed-loop control over long internal deliberation.
THINKING_BUDGET = 1024
# Do not cancel a valid model turn just because it takes longer than one
# heartbeat.  The video and body control continue while Gemini thinks.
# A slow ER2 turn is safer to wait through than to discard and restart.
RESPONSE_TIMEOUT_S = 60.0
START_TIMEOUT_S = 20.0
INITIAL_CONNECT_RETRIES = 1
RECONNECT_DELAY_S = 1.0
MIN_MOVE_S = 0.2
MAX_MOVE_S = 1.0
MAX_FORWARD_SPEED_M_S = 0.25
MAX_RIGHT_SPEED_M_S = 0.20
MIN_TURN_DEG = 5.0
MAX_TURN_DEG = 90.0
# Keep the yaw rate low enough for PX4 to settle near the requested heading.
TURN_RATE_DEG_S = 8.0
MAX_IMAGE_WIDTH = 640
# PX4 may take longer than the commanded yaw rate to settle on a heading.
ACTION_GRACE_S = 3.0
ACTION_SETTLE_S = 1.0
ACTION_STABLE_S = 0.3
HEADING_STABILITY_RAD = math.radians(2.0)
HEADING_TOLERANCE_RAD = math.radians(2.0)
MAX_FRAME_AGE_S = 1.5


@dataclass
class ActiveAction:
    """One physical action that must finish before another can start."""

    kind: str
    direction: str
    amount: float
    deadline_s: float
    start_heading_rad: Optional[float] = None
    phase: str = "running"
    stable_since_s: Optional[float] = None
    last_heading_rad: Optional[float] = None
    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    last_update_s: Optional[float] = None
    last_sample_s: Optional[float] = None
    observed_forward_m: float = 0.0
    observed_right_m: float = 0.0
    completion: Optional[asyncio.Future] = None


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
        self._latest_frame_at_s: Optional[float] = None
        self._last_frame_sent_at_s: Optional[float] = None
        self._telemetry = Telemetry()
        self._dialogue = deque()
        self._active_action: Optional[ActiveAction] = None
        self._needs_post_action_frame = False
        self._stop_requested = False
        self._last_action_result = ""
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
        self.tool_call_count = 0
        self.dialogue_count = 0
        self.turn_count = 0
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
        if _is_explicit_stop(message):
            self._stop_requested = True
            self._cancel_action("explicit stop request")
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
            if self._needs_post_action_frame:
                self._needs_post_action_frame = False
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
                )
            return VelocityCommand()
        if (
            action.kind == "turn"
            and action.phase == "running"
            and _finite(telemetry.heading_rad)
            and _obstacle_is_clear(telemetry.obstacle_distance_m)
        ):
            yaw_rate = TURN_RATE_DEG_S * (
                1.0 if action.direction == "right" else -1.0
            )
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
                try:
                    config = types.LiveConnectConfig(
                        response_modalities=["TEXT"],
                        tools=_tools(),
                        system_instruction=_system_instruction(),
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=THINKING_BUDGET,
                            include_thoughts=True,
                        ),
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
                        self._session = session
                        self._last_frame_sent_at_s = None
                        connected = True
                        self._ready.set()
                        await self._frame_ready.wait()
                        if self._closed.is_set():
                            return
                        video_task = asyncio.create_task(
                            self._stream_video(session, types)
                        )
                        try:
                            receive_task = None
                            response_started_s = None
                            while (
                                not self._closed.is_set()
                                and not self._reconnect_requested
                            ):
                                if receive_task is None:
                                    await self._heartbeat(session, types)
                                    receive_task = asyncio.create_task(
                                        self._receive(session, types)
                                    )
                                    response_started_s = time.monotonic()
                                elif self._dialogue:
                                    await self._send_pending_dialogue(session, types)
                                sent_at_s = time.monotonic()
                                try:
                                    await asyncio.wait_for(
                                        asyncio.shield(receive_task),
                                        timeout=VIDEO_PERIOD_S,
                                    )
                                except asyncio.TimeoutError:
                                    pass
                                if receive_task.done():
                                    await receive_task
                                    receive_task = None
                                    response_started_s = None
                                    if self._reconnect_requested:
                                        break
                                elif (
                                    response_started_s is not None
                                    and time.monotonic() - response_started_s
                                    > RESPONSE_TIMEOUT_S
                                ):
                                    raise TimeoutError(
                                        "Gemini did not complete its response"
                                    )
                                if video_task.done():
                                    await video_task
                                remaining_s = VIDEO_PERIOD_S - (
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
                            tasks = [video_task]
                            if receive_task is not None:
                                tasks.append(receive_task)
                            for task in tasks:
                                task.cancel()
                            await asyncio.gather(
                                *tasks,
                                return_exceptions=True,
                            )
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
        self.video_frame_count += 1

    async def _heartbeat(self, session, types):
        """Send a fresh state prompt when Gemini is ready for a new turn."""

        await self._send_frame(session, types)
        self._last_heartbeat_at_s = time.monotonic()
        dialogue = self._dialogue[0] if self._dialogue else ""
        async with self._send_lock:
            await session.send_realtime_input(text=self._heartbeat_text(dialogue))
        if not self._memory_sent:
            self._memory_sent = True
        if self._bootstrap_pending:
            self._bootstrap_pending = False
        if dialogue:
            self._dialogue.popleft()
            self.dialogue_count += 1

    async def _send_pending_dialogue(self, session, types):
        """Interrupt the current turn with one new user message."""

        if not self._dialogue:
            return
        await self._send_frame(session, types)
        dialogue = self._dialogue.popleft()
        self.dialogue_count += 1
        async with self._send_lock:
            await session.send_realtime_input(
                text=self._heartbeat_text(dialogue)
            )

    def _heartbeat_text(self, dialogue: str) -> str:
        memory = ""
        if not self._memory_sent:
            if self.memory_store is not None:
                memory = self.memory_store.context()
        start = ""
        if self._bootstrap_pending:
            start = f"[START]\nSituation: {self.situation}\n"
        camera = "fresh" if self._has_fresh_frame() else "stale or missing"
        state = (
            f"{start}[STATE]\n"
            f"Camera: {camera}\n"
            f"Vehicle: {_telemetry_text(self._telemetry)}\n"
            f"Action: {self._action_state_text()}\n"
            "Inspect the latest image and current state. Decide for yourself whether "
            "to move, turn, hover, speak, or do nothing. Never infer that a move or "
            "turn is safe from the image alone. A heartbeat does not require a tool "
            "or a visible response."
        )
        if dialogue:
            state += f"\nUser: {dialogue}"
        if memory:
            state += (
                "\nMemory (prior experience; verify it against the current image "
                f"and telemetry):\n{memory}"
            )
        return state

    def _has_fresh_frame(self) -> bool:
        return (
            self._latest_frame_at_s is not None
            and time.monotonic() - self._latest_frame_at_s <= MAX_FRAME_AGE_S
        )

    async def _receive(self, session, types):
        async for message in session.receive():
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
                        else:
                            self._response_parts.append(part.text)
                else:
                    transcript = content.output_transcription
                    if transcript is not None and transcript.text:
                        self._response_parts.append(transcript.text)
                turn_complete = bool(
                    content.turn_complete or content.generation_complete
                )
            tool_call = message.tool_call
            if tool_call is not None:
                self.tool_call_count += len(tool_call.function_calls)
                responses = []
                for call in tool_call.function_calls:
                    result = await self._execute(
                        call.name,
                        call.args or {},
                    )
                    started = (
                        call.name in ("move", "turn")
                        and result.get("status") == "started"
                    )
                    if started and self._active_action is not None:
                        action = self._active_action
                        if action.completion is not None:
                            result = await asyncio.shield(action.completion)
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
            if message.tool_call_cancellation is not None:
                self._cancel_action("Gemini cancelled it")

            if self._reconnect_requested:
                self._response_parts.clear()
                self._response_thoughts.clear()
                self._actions.clear()
                return
            if turn_complete:
                self.turn_count += 1
                self._finish_turn()
                return

    async def _execute(self, name: str, args: dict) -> dict:
        if name == "move":
            return await self._move(args)
        if name == "turn":
            return await self._turn(args)
        if name == "hover":
            if self._active_action is not None and not self._stop_requested:
                return {
                    "status": "unavailable",
                    "reason": (
                        "the physical action is still running; hover can "
                        "interrupt it only for an explicit stop request"
                    ),
                    "active_action": self._action_label(),
                    "movement_tools": "unavailable until the action completes",
                    "telemetry": _telemetry_text(self._telemetry),
                }
            self._stop_requested = False
            cancelled = self._cancel_action("hover")
            self._record_action("hover")
            return {
                "status": "hovering",
                "cancelled_action": cancelled or "none",
                "telemetry": _telemetry_text(self._telemetry),
            }
        if name == "speak":
            message = str(args.get("message", "")).strip()
            if not message:
                return {"status": "rejected", "reason": "message is required"}
            self._record_action(f"speak: {message}")
            print(f"Companion: {message}", flush=True)
            return {"status": "spoken"}
        return {"status": "rejected", "reason": "unknown tool"}

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
        if forward_m_s is None or right_m_s is None or duration_s is None:
            return {
                "status": "rejected",
                "reason": (
                    "forward_m_s must be -"
                    f"{MAX_FORWARD_SPEED_M_S} to {MAX_FORWARD_SPEED_M_S}; "
                    "right_m_s must be "
                    f"-{MAX_RIGHT_SPEED_M_S} to {MAX_RIGHT_SPEED_M_S}; "
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
        now = time.monotonic()
        action = ActiveAction(
            "move",
            _move_direction(forward_m_s, right_m_s),
            duration_s,
            now + duration_s,
            forward_m_s=forward_m_s,
            right_m_s=right_m_s,
            last_update_s=now,
            last_sample_s=now,
            completion=asyncio.get_running_loop().create_future(),
        )
        self._active_action = action
        self._last_action_result = ""
        self._record_action(f"started {self._action_label(action)}")
        return {
            "status": "started",
            "action": self._action_label(action),
            "body_velocity": {
                "forward_m_s": forward_m_s,
                "right_m_s": right_m_s,
            },
            "movement_tools": "unavailable until this action completes",
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
        busy = self._busy_response()
        if busy is not None:
            return busy
        now = time.monotonic()
        angle_deg = float(angle_deg)
        action = ActiveAction(
            "turn",
            direction,
            angle_deg,
            now + angle_deg / TURN_RATE_DEG_S + ACTION_GRACE_S,
            self._telemetry.heading_rad
            if _finite(self._telemetry.heading_rad)
            else None,
            last_update_s=now,
            completion=asyncio.get_running_loop().create_future(),
        )
        self._active_action = action
        self._last_action_result = ""
        self._record_action(f"started {self._action_label(action)}")
        return {
            "status": "started",
            "action": self._action_label(action),
            "movement_tools": "unavailable until this action completes",
            "telemetry": _telemetry_text(self._telemetry),
        }

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
            if not self._needs_post_action_frame:
                return None
            return {
                "status": "unavailable",
                "reason": "the last physical action finished; waiting for a fresh camera frame",
                "movement_tools": "unavailable until a fresh camera frame arrives",
                "telemetry": _telemetry_text(self._telemetry),
            }
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
            return (
                f"move forward={action.forward_m_s:+.2f}m/s "
                f"right={action.right_m_s:+.2f}m/s for {action.amount:.1f}s"
            )
        return f"turn {action.direction} {action.amount:.0f} degrees"

    def _movement_label(self) -> str:
        action = self._active_action
        if (
            action is not None
            and action.kind == "move"
            and action.phase == "running"
        ):
            return action.direction
        return "stop"

    def _action_state_text(self) -> str:
        self._refresh_action()
        action = self._active_action
        if action is None:
            if self._needs_post_action_frame:
                result = self._last_action_result or "last action finished"
                return (
                    f"{result}; waiting for a fresh camera frame; "
                    "movement tools unavailable until then"
                )
            if self._last_action_result:
                return f"{self._last_action_result}; movement tools available"
            return "none; movement tools available"
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
        self._pause_action_if_blocked(action, now)
        if action.kind == "turn":
            if action.start_heading_rad is None and _finite(self._telemetry.heading_rad):
                action.start_heading_rad = self._telemetry.heading_rad
            actual = self._heading_change_deg(action)
            if action.phase == "running" and actual is not None:
                requested_rad = math.radians(action.amount)
                progress_rad = math.radians(actual)
                if progress_rad >= requested_rad - HEADING_TOLERANCE_RAD:
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
        if now >= action.deadline_s:
            actual = self._heading_change_deg(action)
            if action.kind == "turn" and actual is not None:
                requested = action.amount - math.degrees(HEADING_TOLERANCE_RAD)
                status = (
                    "completed"
                    if actual >= requested
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
        if not _obstacle_is_clear(self._telemetry.obstacle_distance_m):
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

    def _complete_action(self, status: str, actual_heading_deg: Optional[float] = None):
        action = self._active_action
        if action is None:
            return
        result = f"{self._action_label(action)} {status}"
        if actual_heading_deg is not None:
            result += f"; observed heading change {actual_heading_deg:+.1f} degrees"
        if action.kind == "move":
            result += f"; {self._translation_text(action)}"
        self._last_action_result = result
        self._needs_post_action_frame = True
        self._finish_action_waiter(
            action,
            self._action_response(action, status, result, actual_heading_deg),
        )
        self._record_action(result)
        self._active_action = None

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
        self._needs_post_action_frame = True
        self._finish_action_waiter(
            action,
            self._action_response(action, "cancelled", result, actual),
        )
        self._record_action(result)
        self._active_action = None
        return self._action_label(action)

    def _action_response(
        self,
        action: ActiveAction,
        status: str,
        result: str,
        actual_heading_deg: Optional[float],
    ) -> dict:
        response = {
            "status": status,
            "action": result,
            "telemetry": _telemetry_text(self._telemetry),
            "movement_tools": (
                "unavailable until a fresh camera frame arrives"
                if self._needs_post_action_frame
                else "available"
            ),
        }
        if actual_heading_deg is not None:
            response["observed_heading_change_deg"] = actual_heading_deg
        if action.kind == "move":
            response["observed_translation_m"] = {
                "forward": action.observed_forward_m,
                "right": action.observed_right_m,
            }
        return response

    def _finish_action_waiter(self, action: ActiveAction, response: dict):
        if action.completion is not None and not action.completion.done():
            action.completion.set_result(response)

    def _record_action(self, action: str):
        action = " ".join(str(action).split())
        if action:
            self._actions.append(action)

    def _finish_turn(self):
        thought = _model_text(self._response_thoughts)
        response = _model_text(self._response_parts)
        actions = tuple(self._actions)
        action = "; ".join(actions) or "none"
        self._response_thoughts.clear()
        self._response_parts.clear()
        self._actions.clear()
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
            movement=self._movement_label(),
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
            memory_action = "; ".join(
                event for event in actions if not event.startswith("started ")
            )
            action_outcome = any(
                marker in action
                for marker in (
                    " completed",
                    " timed out",
                    " cancelled ",
                    "speak:",
                    "hover",
                )
            )
            useful_summary = summary not in ("", "none") and action not in summary
            if action_outcome or useful_summary:
                experience = _telemetry_text(self._telemetry)
                if action_outcome and memory_action:
                    experience += f"; action={memory_action}"
                if useful_summary:
                    experience += f"; summary={summary}"
                self.memory_store.remember(experience)


def _tools():
    """Return the high-level actions exposed to Gemini."""

    return [{"function_declarations": [
        {
            "name": "move",
            "description": (
                "Move slowly in the body frame for a short time. "
                "Forward is positive and right is positive."
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
                        "description": f"A duration from {MIN_MOVE_S} through {MAX_MOVE_S} seconds.",
                        "minimum": MIN_MOVE_S,
                        "maximum": MAX_MOVE_S,
                    },
                },
                "required": ["forward_m_s", "right_m_s", "duration_s"],
            },
        },
        {
            "name": "turn",
            "description": (
                "Turn in place slowly by an angle relative to the current heading. "
                "Use small corrections when aligning with something."
            ),
            "behavior": "BLOCKING",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "direction": {
                        "type": "STRING",
                        "enum": ["left", "right"],
                    },
                    "angle_deg": {
                        "type": "NUMBER",
                        "description": (
                            f"A turn from {MIN_TURN_DEG:.0f} through "
                            f"{MAX_TURN_DEG:.0f} degrees."
                        ),
                    },
                },
                "required": ["direction", "angle_deg"],
            },
        },
        {
            "name": "hover",
            "description": (
                "Stop horizontal motion and hold position. Use this when the "
                "task is complete, while waiting, or when the scene is unclear. "
                "It can interrupt a movement only for an explicit stop."
            ),
            "behavior": "BLOCKING",
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "speak",
            "description": "Say one short message to the nearby user.",
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
        "newest camera image, telemetry, active action, conversation, and previous "
        "action results. Telemetry includes forward TOF distance, body velocity, "
        "and heading. Turn angles are relative to the current heading. You have no "
        "lidar or direct motor control; PX4 stabilizes "
        "the vehicle and holds altitude. "
        "Think broadly and choose whether to move, turn, hover, speak, or do nothing. "
        "Treat each user message as the active request and begin a physical action "
        "when the current state permits; do not only describe a plan. Estimate the "
        "smallest useful speed, direction, angle, and duration yourself. "
        "The camera image is not mirrored: an object on image-left needs a left turn, "
        "and an object on image-right needs a right turn. Use the current heading and "
        "measured motion as feedback. When an action reports completion, accept its "
        "observed heading or translation as fact and choose the next action from the "
        "new image and state; never repeat or extend an old action without new "
        "evidence. "
        "Movement is a closed loop: start only one move or turn at a time, wait for "
        "the state to report completion and a fresh image, then reassess. Movement "
        "tools are unavailable while an action is active. A heartbeat may arrive "
        "during an action; use it to observe progress, but wait for completion "
        "before choosing another movement. In an open-ended task, "
        "continue with another small purposeful action after each fresh view unless "
        "the task is complete, waiting, or the state is unclear. If you are aligning "
        "with something, make a small correction and reassess instead of repeating "
        "a turn without new visual evidence. When a requested thing is already "
        "visible, act on that view before scanning elsewhere. If the request says to "
        "wait, stop, or inspect when close, hover and remain there when that condition "
        "is met. "
        "Use hover when the task is complete or the image or telemetry is unclear. "
        "Do not use hover merely to wait for a response or end a turn early. "
        "Speak only when useful. Do not repeat the state block or telemetry. "
        "Never request altitude, motors, attitude, position, or a long translation. "
        "The CM5 checks every physical action and may stop it; its safety result and "
        "the reported action completion are authoritative."
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
    for prefix in ("```json", "```", "{}", "[]", "null", "---"):
        if value.startswith(prefix):
            value = value[len(prefix):].lstrip(" :")
            break
    while value.endswith("---"):
        value = value[:-3].rstrip()
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
    if value.replace("```json", "").replace("```", "").strip() in {
        "{}",
        "[]",
        "null",
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

    return _finite(distance_m) and distance_m > OBSTACLE_STOP_M


def _heading_number(value) -> str:
    """Format one heading in degrees without hiding missing telemetry."""

    if _finite(value):
        return f"{math.degrees(value):.1f}"
    return "?"


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
