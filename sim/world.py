"""A small fixed world for PX4 behavior checks.

It supplies repeatable targets and faults while the vehicle flies in Gazebo
through the real command path.
"""

import asyncio
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from mavsdk import System
from mavsdk.offboard import OffboardError

from control.mind import (
    CompanionMemory,
    ConsciousDecision,
    MacMind,
    Telemetry,
    VisualObservation,
)
from control.dialogue import DialogueInput
from control.mind_runtime import (
    MAX_MOVEMENT_AGE_S,
    MIN_MOVEMENT_CONFIDENCE,
    MindRuntime,
)
from control.safety_limits import OBSTACLE_STOP_M
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.safety import LatestDistanceSensor
from sim.flight import (
    RecordingForwarder,
    close_mavsdk,
    land,
    prepare,
    wait_for_offboard,
)
from sim.gazebo_camera import GazeboCamera
from sim.gazebo_depth import GazeboDepthRangefinder
from sim.mavsdk_forwarder import MavsdkVelocityForwarder
from sim.offboard_control import (
    DistanceMessage,
    PROFILE_DURATION_S,
    SECOND_FOLLOW_END_S,
    SECOND_FOLLOW_START_S,
    SETPOINT_PERIOD_S,
    THIRD_FOLLOW_END_S,
    THIRD_FOLLOW_START_S,
)
from voice.intent import parse_intent


TARGET_RIGHT_START_S = 1.0
TARGET_RIGHT_END_S = 2.0
CONTROL_PAUSE_START_S = 1.5
CONTROL_PAUSE_END_S = 1.75
TARGET_LOST_START_S = 2.0
TARGET_LOST_END_S = 2.6
OBSTACLE_START_S = 2.8
OBSTACLE_END_S = 3.7
RECOVERY_END_S = 4.2
INVALID_SENSOR_START_S = 4.2
INVALID_SENSOR_END_S = 4.6
DROPOUT_START_S = 4.6
DROPOUT_END_S = 5.2
LINK_RECOVERY_END_S = 5.35
INVALID_COMMAND_START_S = 5.35
INVALID_COMMAND_END_S = 5.75
LOW_CONFIDENCE_START_S = 12.4
LOW_CONFIDENCE_END_S = 12.8
STALE_SENSOR_START_S = 15.0
STALE_SENSOR_END_S = 15.5
HOVER_START_S = 5.8
MAX_EXPLORATORY_SPEED_M_S = 1.0
MAX_HEADING_CHANGE_DEG = 15.0
ATTITUDE_TELEMETRY_RATE_HZ = 5.0
DEFAULT_EXPLORATORY_INTENT = "explore the surroundings"


@dataclass(frozen=True)
class WorldStep:
    obstacle_distance_m: Optional[float] = 2.0
    distance_fresh: bool = True
    transmit: bool = True
    command_override: Optional[VelocityCommand] = None


class SyntheticWorld:
    """Provide fixed targets, sensors, and link faults."""

    def __init__(self, exploratory: bool = False):
        self.exploratory = exploratory

    def target_offset_east(self, elapsed_s: float) -> Optional[float]:
        if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
            return None
        return 0.8 if TARGET_RIGHT_START_S <= elapsed_s < TARGET_RIGHT_END_S else 0.0

    def vision_confidence(self, elapsed_s: float) -> float:
        if (
            not self.exploratory
            and LOW_CONFIDENCE_START_S <= elapsed_s < LOW_CONFIDENCE_END_S
        ):
            return 0.0
        return 1.0

    def step(
        self,
        elapsed_s: float,
    ) -> WorldStep:
        if self.exploratory:
            return WorldStep()
        if STALE_SENSOR_START_S <= elapsed_s < STALE_SENSOR_END_S:
            return WorldStep(distance_fresh=False)
        if OBSTACLE_START_S <= elapsed_s < OBSTACLE_END_S:
            return WorldStep(obstacle_distance_m=0.3)
        if OBSTACLE_END_S <= elapsed_s < RECOVERY_END_S:
            return WorldStep()
        if INVALID_SENSOR_START_S <= elapsed_s < INVALID_SENSOR_END_S:
            return WorldStep(obstacle_distance_m=math.nan)
        if DROPOUT_START_S <= elapsed_s < DROPOUT_END_S:
            return WorldStep(transmit=False)
        if DROPOUT_END_S <= elapsed_s < LINK_RECOVERY_END_S:
            return WorldStep()
        if INVALID_COMMAND_START_S <= elapsed_s < INVALID_COMMAND_END_S:
            return WorldStep(
                command_override=VelocityCommand(north_m_s=1.0, east_m_s=1.0)
            )
        return WorldStep()


class WorldVisualModel:
    """Provide fixed scene descriptions to the Mac VLM boundary."""

    def __init__(
        self,
        world: SyntheticWorld,
        started_at_s: float,
        synthetic_scene: bool = True,
    ):
        self.world = world
        self.started_at_s = started_at_s
        self.synthetic_scene = synthetic_scene

    def observe(
        self,
        _image,
        timestamp_s: float,
        focus: str,
        intent: str,
        previous_movement: str,
        previous_observation: str,
        telemetry: Telemetry,
    ) -> VisualObservation:
        if not self.synthetic_scene:
            return VisualObservation(
                timestamp_s=timestamp_s,
                description="camera frame received; no visual model configured",
                next_focus=focus,
            )
        elapsed_s = max(0.0, timestamp_s - self.started_at_s)
        target_offset_east_m = self.world.target_offset_east(elapsed_s)
        if target_offset_east_m is None:
            description = "no person is visible"
            movement = "stop"
        elif target_offset_east_m > 0.0:
            movement = "right"
            description = "the person is to the right"
        else:
            movement = "forward"
            description = "the person is ahead"
        if parse_intent(intent) != "following":
            movement = "stop"
        return VisualObservation(
            timestamp_s=timestamp_s,
            description=description,
            focused_answer=description if focus else "",
            movement=movement,
            next_focus=focus or "person",
            confidence=self.world.vision_confidence(elapsed_s),
        )


class WorldLanguageModel:
    """Choose simulated intent while exercising the conscious boundary."""

    def __init__(self, exploratory: bool):
        self.exploratory = exploratory
        self.started_at_s = 0.0
        self.intent = None

    def think(self, information) -> ConsciousDecision:
        if self.intent is None:
            self.intent = information.intent
        dialogue = ""
        if information.dialogue:
            intent = parse_intent(information.dialogue)
            if intent is None:
                dialogue = "I did not understand that request."
            else:
                self.intent = intent
                dialogue = f"Intent changed to {self.intent}."
        if not self.exploratory:
            elapsed_s = max(0.0, time.monotonic() - self.started_at_s)
            following = (
                elapsed_s < HOVER_START_S
                or SECOND_FOLLOW_START_S <= elapsed_s < SECOND_FOLLOW_END_S
                or THIRD_FOLLOW_START_S <= elapsed_s < THIRD_FOLLOW_END_S
            )
            self.intent = "following" if following else "hover"
        summary = information.summary
        if information.new_observations:
            summary = information.new_observations[-1].description
        return ConsciousDecision(
            intent=self.intent,
            focus="person" if parse_intent(self.intent) == "following" else "",
            dialogue=dialogue,
            summary=summary or "The simulated world is running.",
        )


async def run(
    exploratory: bool = False,
    camera: bool = False,
    world_name: str = "default",
    duration_s: float = PROFILE_DURATION_S,
    ollama: bool = False,
    vlm_model: str = "moondream",
    llm_model: str = "gemma3:4b",
    ollama_timeout: float = 60.0,
    initial_intent: str = DEFAULT_EXPLORATORY_INTENT,
    depth: bool = False,
    memory_path: Optional[Path] = None,
    dialogue_request: Optional[str] = None,
    trace: bool = False,
):
    """Run one complete Gazebo world simulation."""

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("simulation duration must be positive")
    if not exploratory and duration_s < PROFILE_DURATION_S:
        raise ValueError("deterministic simulation duration cannot be shorter than its profile")
    if not isinstance(initial_intent, str) or not initial_intent.strip():
        raise ValueError("initial intent must be a non-empty string")
    initial_intent = initial_intent.strip()
    if camera and depth:
        raise ValueError("camera and depth modes cannot run together")
    if depth and not exploratory:
        raise ValueError("Gazebo depth mode requires exploratory simulation")
    if ollama and not (exploratory and (camera or depth)):
        raise ValueError("Ollama simulation requires exploratory camera or depth mode")
    if memory_path is not None and not exploratory:
        raise ValueError("experience memory requires exploratory simulation")
    if dialogue_request is not None and not exploratory:
        raise ValueError("dialogue request requires exploratory simulation")
    if dialogue_request is not None and not dialogue_request.strip():
        raise ValueError("dialogue request must not be empty")
    requested_intent = (
        parse_intent(dialogue_request) if dialogue_request is not None else None
    )

    ollama_client = None
    if ollama:
        from control.ollama_brain import OllamaClient, OllamaLanguageModel, OllamaVisionModel

        ollama_client = OllamaClient(timeout_s=ollama_timeout)
        await asyncio.to_thread(ollama_client.check)
        await asyncio.to_thread(ollama_client.preload, vlm_model)
        await asyncio.to_thread(ollama_client.preload, llm_model)
    memory_store = CompanionMemory(memory_path) if memory_path is not None else None

    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    service_task = None
    mind_task = None
    service_stop = asyncio.Event()
    mind_stop = asyncio.Event()
    telemetry_task = None
    attitude_task = None
    offboard_task = None
    dialogue_input = DialogueInput(dialogue_request) if exploratory else None
    gazebo_camera = None
    gazebo_depth = None
    control = None
    applied_dialogue_intent = None
    offboard_started = False
    armed = False
    landed = False
    distance_sensor = LatestDistanceSensor()
    north_velocity_m_s = None
    east_velocity_m_s = None
    down_velocity_m_s = None
    velocity_telemetry_seen = False
    initial_yaw_deg = None
    max_heading_change_deg = 0.0
    try:
        world = SyntheticWorld(exploratory)
        if camera or depth:
            camera_sensor = "IMX214" if depth else "camera"
            camera_model = "x500_depth" if depth else "x500_mono_cam"
            gazebo_camera = GazeboCamera(
                f"/world/{world_name}/model/{camera_model}_0/"
                f"link/camera_link/sensor/{camera_sensor}/image"
            )
            gazebo_camera.start()
        if depth:
            gazebo_depth = GazeboDepthRangefinder("/depth_camera")
            gazebo_depth.start()

        await prepare(drone)
        await drone.telemetry.set_rate_attitude_euler(ATTITUDE_TELEMETRY_RATE_HZ)
        armed = True

        async def observe_heading():
            nonlocal initial_yaw_deg, max_heading_change_deg
            async for attitude in drone.telemetry.attitude_euler():
                yaw_deg = attitude.yaw_deg
                if not math.isfinite(yaw_deg):
                    continue
                if initial_yaw_deg is None:
                    initial_yaw_deg = yaw_deg
                change_deg = (yaw_deg - initial_yaw_deg + 180.0) % 360.0 - 180.0
                max_heading_change_deg = max(
                    max_heading_change_deg,
                    abs(change_deg),
                )

        attitude_task = asyncio.create_task(observe_heading())

        forwarder = MavsdkVelocityForwarder(drone)

        safe_commands = RecordingForwarder(forwarder)

        service = SafetyCommandService(
            receiver,
            safe_commands,
            tick_period_s=SETPOINT_PERIOD_S,
            obstacle_distance=distance_sensor.read,
            velocity_provider=lambda: (
                north_velocity_m_s,
                east_velocity_m_s,
                down_velocity_m_s,
            ),
        )
        service.start()
        service_task = asyncio.create_task(service.run(service_stop))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        started_at = time.monotonic()
        if ollama_client is None:
            visual_model = WorldVisualModel(
                world,
                started_at,
                synthetic_scene=not (camera or depth),
            )
            language_model = WorldLanguageModel(exploratory)
            language_model.started_at_s = started_at
        else:
            visual_model = OllamaVisionModel(ollama_client, vlm_model)
            language_model = OllamaLanguageModel(ollama_client, llm_model)
        control = MindRuntime(
            MacMind(visual_model, language_model, memory=memory_store)
        )

        camera_frames = 0
        depth_samples = 0
        valid_depth_samples = 0
        minimum_depth_distance = math.inf

        def brain_telemetry():
            nonlocal velocity_telemetry_seen
            telemetry = sender.telemetry()
            if any(
                value is not None
                for value in (
                    telemetry.north_velocity_m_s,
                    telemetry.east_velocity_m_s,
                    telemetry.down_velocity_m_s,
                )
            ):
                velocity_telemetry_seen = True
            return telemetry

        def read_step(elapsed_s: float):
            nonlocal depth_samples, valid_depth_samples, minimum_depth_distance
            step = world.step(elapsed_s)
            if gazebo_depth is not None:
                sample = gazebo_depth.latest()
                if sample is not None:
                    distance_sensor.update(sample)
                    depth_samples += 1
                    distance = distance_sensor.read()
                    if math.isfinite(distance):
                        valid_depth_samples += 1
                        minimum_depth_distance = min(minimum_depth_distance, distance)
                return replace(
                    step,
                    obstacle_distance_m=distance_sensor.read(),
                    distance_fresh=sample is not None,
                )
            if step.distance_fresh:
                distance_sensor.update(
                    DistanceMessage(
                        step.obstacle_distance_m
                        if step.obstacle_distance_m is not None
                        else math.nan
                    )
                )
            return replace(step, obstacle_distance_m=distance_sensor.read())

        def send_packet(timestamp_s: float, step, intent=None):
            nonlocal camera_frames
            frame = gazebo_camera.latest() if gazebo_camera else step
            if gazebo_camera is not None and frame is not None:
                camera_frames += 1
            telemetry = brain_telemetry()
            command = control.tick(
                frame=frame,
                timestamp_s=timestamp_s,
                intent=intent,
                telemetry=telemetry,
            )
            if step.transmit:
                sender.send(step.command_override or command)
            return command, frame is not None

        starting_intent = initial_intent if exploratory else "following"
        send_packet(
            time.monotonic(),
            read_step(0.0),
            intent=starting_intent,
        )
        deadline = time.monotonic() + 5.0
        while len(safe_commands.commands) < 3 and time.monotonic() < deadline:
            await asyncio.sleep(SETPOINT_PERIOD_S)
        if len(safe_commands.commands) < 3:
            raise RuntimeError("CM5 did not forward consecutive priming setpoints")
        print("CM5 priming setpoints=verified.")
        await asyncio.sleep(SETPOINT_PERIOD_S * 4)
        await drone.offboard.start()
        offboard_started = True
        offboard_task = asyncio.create_task(wait_for_offboard(drone))
        started_at = time.monotonic()
        if ollama_client is None:
            visual_model.started_at_s = started_at
            language_model.started_at_s = started_at
        dialogue_provider = None
        if dialogue_input is not None:
            def read_dialogue():
                nonlocal applied_dialogue_intent
                message = dialogue_input.next()
                if message:
                    applied_dialogue_intent = parse_intent(message)
                return message

            dialogue_provider = read_dialogue

        mind_task = asyncio.create_task(
            control.think_loop(
                mind_stop,
                telemetry_provider=brain_telemetry,
                dialogue_provider=dialogue_provider,
            )
        )
        if dialogue_input:
            dialogue_input.start()
            print("Exploratory mission: observe the brain, then land and disarm.")
        else:
            print("Mission: follow, recover, handle faults, hover, land, and disarm.")

        max_north_velocity = 0.0
        min_north_velocity = 0.0
        max_east_velocity = 0.0
        min_east_velocity = 0.0
        last_traced_observation = 0
        last_traced_decision = 0
        last_observation_signature = None
        last_decision_signature = None
        last_traced_command = None

        async def observe_velocity():
            nonlocal max_north_velocity, min_north_velocity
            nonlocal max_east_velocity, min_east_velocity
            nonlocal north_velocity_m_s, east_velocity_m_s, down_velocity_m_s
            async for velocity in drone.telemetry.velocity_ned():
                north_velocity_m_s = velocity.north_m_s
                east_velocity_m_s = velocity.east_m_s
                down_velocity_m_s = velocity.down_m_s
                max_north_velocity = max(max_north_velocity, velocity.north_m_s)
                min_north_velocity = min(min_north_velocity, velocity.north_m_s)
                max_east_velocity = max(max_east_velocity, velocity.east_m_s)
                min_east_velocity = min(min_east_velocity, velocity.east_m_s)

        telemetry_task = asyncio.create_task(observe_velocity())
        reported = set()

        def event_for(elapsed_s, step):
            if TARGET_RIGHT_START_S <= elapsed_s < TARGET_RIGHT_END_S:
                return "target moved right"
            if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
                return "target lost; holding"
            if (
                step.obstacle_distance_m is not None
                and step.obstacle_distance_m < OBSTACLE_STOP_M
            ):
                return "obstacle detected; backing off"
            if not step.distance_fresh:
                return "distance sensor dropout"
            if isinstance(step.obstacle_distance_m, float) and math.isnan(step.obstacle_distance_m):
                return "invalid obstacle reading"
            if not step.transmit:
                return "command link dropout"
            if step.command_override is not None:
                return "out-of-bounds command"
            if LOW_CONFIDENCE_START_S <= elapsed_s < LOW_CONFIDENCE_END_S:
                return "low-confidence vision; holding"
            if SECOND_FOLLOW_START_S <= elapsed_s < SECOND_FOLLOW_END_S:
                return "intent changed back to following"
            if THIRD_FOLLOW_START_S <= elapsed_s < THIRD_FOLLOW_END_S:
                return "intent changed to following again"
            if elapsed_s >= HOVER_START_S:
                return "intent changed to hover"
            return None

        def trace_brain(elapsed_s, step, mac_command, frame_available):
            nonlocal last_traced_observation, last_traced_decision
            nonlocal last_observation_signature, last_decision_signature
            nonlocal last_traced_command
            if not trace:
                return

            def clean(value):
                return " ".join(str(value).split()) or "none"

            if control.observation_count != last_traced_observation:
                observation = control.latest_observation
                if observation is not None:
                    signature = (
                        clean(observation.description),
                        clean(observation.next_focus),
                        observation.movement,
                        round(observation.confidence, 2),
                    )
                    if signature != last_observation_signature:
                        print(
                            f"[VLM {elapsed_s:5.1f}s] "
                            f"{signature[0]}; focus={signature[1]}; "
                            f"movement={signature[2]}; "
                            f"confidence={signature[3]:.2f}",
                            flush=True,
                        )
                        last_observation_signature = signature
                last_traced_observation = control.observation_count

            if control.decision_count != last_traced_decision:
                decision = control.latest_decision
                if decision is not None:
                    signature = (
                        clean(decision.intent),
                        clean(decision.focus),
                        clean(decision.summary),
                    )
                    if signature != last_decision_signature:
                        print(
                            f"[LLM {elapsed_s:5.1f}s] "
                            f"intent={signature[0]}; focus={signature[1]}; "
                            f"summary={signature[2]}",
                            flush=True,
                        )
                        last_decision_signature = signature
                last_traced_decision = control.decision_count

            distance = step.obstacle_distance_m
            if not step.transmit:
                reason = "command link dropout"
            elif not step.distance_fresh or distance is None or not math.isfinite(distance):
                reason = "CM5 has no fresh distance reading"
            elif distance < OBSTACLE_STOP_M:
                reason = f"CM5 obstacle protection at {distance:.2f}m"
            elif step.command_override is not None:
                reason = "CM5 rejected the injected invalid command"
            elif mac_command == VelocityCommand():
                decision = control.latest_decision
                observation = control.latest_observation
                if decision is not None and parse_intent(decision.intent) == "hover":
                    reason = "conscious intent is hover"
                elif not frame_available:
                    reason = "waiting for a fresh camera frame"
                elif observation is None:
                    reason = "waiting for the first VLM observation"
                elif time.monotonic() - observation.timestamp_s > MAX_MOVEMENT_AGE_S:
                    reason = "VLM observation is stale"
                elif observation.confidence < MIN_MOVEMENT_CONFIDENCE:
                    reason = "VLM confidence is below the movement threshold"
                elif observation.movement in ("stop", "hover"):
                    reason = f"VLM suggested {observation.movement}"
                else:
                    reason = "Mac timing or intent refresh held zero"
            else:
                reason = "Mac suggested movement"
            forwarded = safe_commands.commands[-1][1] if safe_commands.commands else None
            command_state = (mac_command, forwarded, reason)
            if command_state != last_traced_command:
                print(
                    f"[CMD {elapsed_s:5.1f}s] mac={mac_command}; "
                    f"cm5={forwarded or 'pending'}; reason={reason}",
                    flush=True,
                )
                last_traced_command = command_state

        try:
            while (elapsed := time.monotonic() - started_at) < duration_s:
                now = time.monotonic()
                step = read_step(elapsed)
                event = None if exploratory else event_for(elapsed, step)
                if event is not None and event not in reported:
                    print(event.capitalize() + ".")
                    reported.add(event)
                if not exploratory and CONTROL_PAUSE_START_S <= elapsed < CONTROL_PAUSE_END_S:
                    if "Mac control pause" not in reported:
                        print("Mac control pause; watchdog holds zero.")
                        reported.add("Mac control pause")
                    await asyncio.sleep(CONTROL_PAUSE_END_S - elapsed)
                    now = time.monotonic()
                    step = read_step(now - started_at)
                    command, frame_available = send_packet(now, step)
                    trace_brain(now - started_at, step, command, frame_available)
                    continue
                command, frame_available = send_packet(now, step)
                trace_brain(elapsed, step, command, frame_available)
                await asyncio.sleep(SETPOINT_PERIOD_S)
            await asyncio.wait_for(offboard_task, timeout=5.0)
            print("Offboard telemetry=verified through synthetic world and CM5 safety.")
        finally:
            sender.close()
            service_stop.set()
            await service_task
            mind_stop.set()
            if mind_task is not None:
                await mind_task
            control.close()
            telemetry_task.cancel()
            await asyncio.gather(telemetry_task, return_exceptions=True)
            attitude_task.cancel()
            await asyncio.gather(attitude_task, return_exceptions=True)
            if offboard_started:
                try:
                    await drone.offboard.stop()
                except OffboardError:
                    pass
                offboard_started = False

        commands = safe_commands.commands
        if not commands or commands[-1][1] != VelocityCommand():
            raise RuntimeError("SITL did not observe zero command on CM5 shutdown")
        if depth:
            minimum_forwarded_north = min(
                command.north_m_s for _, command in commands
            )
            print(
                "Minimum CM5-forwarded north command: "
                f"{minimum_forwarded_north:.2f}m/s"
            )

        decision = control.latest_decision
        if decision is None:
            raise RuntimeError("SITL did not observe a conscious Mac decision")
        if control.latest_observation is None:
            raise RuntimeError("SITL did not observe a Mac visual observation")
        if not decision.summary:
            raise RuntimeError("SITL did not retain a conscious visual summary")
        if control.observation_count == 0:
            raise RuntimeError("SITL did not complete a Mac VLM observation")
        if control.decision_count == 0:
            raise RuntimeError("SITL did not complete a conscious thought")
        if not velocity_telemetry_seen:
            raise RuntimeError("SITL did not feed CM5 velocity telemetry to the Mac brain")
        print("Mac velocity telemetry=verified.")
        if initial_yaw_deg is None:
            raise RuntimeError("SITL did not provide heading telemetry")
        if max_heading_change_deg > MAX_HEADING_CHANGE_DEG:
            raise RuntimeError(
                "SITL heading changed unexpectedly: "
                f"{max_heading_change_deg:.1f} degrees"
            )
        print(
            "PX4 heading hold=verified: "
            f"maximum change {max_heading_change_deg:.1f} degrees."
        )
        if requested_intent is not None:
            if applied_dialogue_intent != requested_intent:
                raise RuntimeError(
                    "SITL did not receive the scripted dialogue request: "
                    f"expected {requested_intent}, got {applied_dialogue_intent}"
                )
            print(f"Scripted dialogue=verified: intent={requested_intent}.")
            if requested_intent == "hover":
                if any(command != VelocityCommand() for _, command in commands):
                    raise RuntimeError("explicit hover dialogue commanded motion")
                print("Explicit hover dialogue stop=verified.")
        elif dialogue_request is not None:
            print("Scripted open-ended dialogue=delivered to the conscious mind.")
        print(
            "Conscious Mac decision=verified: "
            f"intent={decision.intent}, focus={decision.focus or 'none'}."
        )
        print("Conscious visual memory=verified.")
        print("Mac visual observation=verified.")
        print(f"Mac VLM observations=verified ({control.observation_count}).")
        print(f"Conscious thoughts=verified ({control.decision_count}).")
        if ollama:
            print(
                "Local Ollama brain=verified: "
                f"VLM={vlm_model}, LLM={llm_model}."
            )
        if memory_store is not None:
            if memory_store.context():
                print("Conscious experience memory=verified.")

        def forward_count(start_s, end_s):
            return sum(
                command.north_m_s > 0.0
                for timestamp_s, command in commands
                if start_s <= timestamp_s - started_at < end_s
            )

        if not exploratory:
            print(
                "Following commands by cycle: "
                f"{forward_count(0.0, 4.0)}, "
                f"{forward_count(SECOND_FOLLOW_START_S, SECOND_FOLLOW_END_S)}, "
                f"{forward_count(THIRD_FOLLOW_START_S, THIRD_FOLLOW_END_S)}."
            )

        def observed(start_s, end_s, predicate):
            return any(
                start_s <= timestamp_s - started_at < end_s and predicate(command)
                for timestamp_s, command in commands
            )

        checks = () if exploratory else (
            (0.0, 1.0, lambda command: command.north_m_s > 0.0, "forward following"),
            (
                TARGET_RIGHT_START_S,
                TARGET_RIGHT_END_S,
                lambda command: command.east_m_s > 0.0,
                "lateral target tracking",
            ),
            (
                CONTROL_PAUSE_START_S,
                CONTROL_PAUSE_END_S + 0.1,
                lambda command: command == VelocityCommand(),
                "Mac heartbeat pause fail-safe",
            ),
            (
                CONTROL_PAUSE_END_S + 0.1,
                TARGET_RIGHT_END_S,
                lambda command: command.east_m_s > 0.0,
                "Mac heartbeat recovery",
            ),
            (2.1, TARGET_LOST_END_S, lambda command: command == VelocityCommand(), "hold after target loss"),
            (OBSTACLE_START_S, OBSTACLE_END_S, lambda command: command.north_m_s < 0.0, "obstacle backoff"),
            (
                OBSTACLE_END_S,
                RECOVERY_END_S,
                lambda command: command.north_m_s > 0.0,
                "following recovery after obstacle",
            ),
            (
                INVALID_SENSOR_START_S,
                INVALID_SENSOR_END_S,
                lambda command: command == VelocityCommand(),
                "invalid obstacle fail-safe",
            ),
            (
                LOW_CONFIDENCE_START_S,
                LOW_CONFIDENCE_END_S + 0.1,
                lambda command: command == VelocityCommand(),
                "low-confidence visual stop",
            ),
            (
                LOW_CONFIDENCE_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                lambda command: command.north_m_s > 0.0,
                "following recovery after low-confidence vision",
            ),
            (
                STALE_SENSOR_START_S + 0.2,
                STALE_SENSOR_END_S,
                lambda command: command == VelocityCommand(),
                "stale obstacle fail-safe",
            ),
            (
                STALE_SENSOR_END_S + 0.1,
                SECOND_FOLLOW_END_S,
                lambda command: command.north_m_s > 0.0,
                "following recovery after stale obstacle sensor",
            ),
            (
                DROPOUT_START_S + 0.15,
                DROPOUT_END_S,
                lambda command: command == VelocityCommand(),
                "command-dropout expiry",
            ),
            (
                DROPOUT_END_S,
                LINK_RECOVERY_END_S,
                lambda command: command.north_m_s > 0.0,
                "command-link recovery",
            ),
            (
                INVALID_COMMAND_START_S,
                INVALID_COMMAND_END_S,
                lambda command: command == VelocityCommand(),
                "invalid-command rejection",
            ),
            (
                HOVER_START_S,
                SECOND_FOLLOW_START_S,
                lambda command: command == VelocityCommand(),
                "first hover",
            ),
            (
                SECOND_FOLLOW_START_S,
                SECOND_FOLLOW_END_S,
                lambda command: command.north_m_s > 0.0,
                "following after hover",
            ),
            (
                SECOND_FOLLOW_END_S,
                THIRD_FOLLOW_START_S,
                lambda command: command == VelocityCommand(),
                "second hover",
            ),
            (
                THIRD_FOLLOW_START_S,
                THIRD_FOLLOW_END_S,
                lambda command: command.north_m_s > 0.0,
                "following after second hover",
            ),
            (
                THIRD_FOLLOW_END_S,
                PROFILE_DURATION_S,
                lambda command: command == VelocityCommand(),
                "final hover",
            ),
        )
        for start_s, end_s, predicate, behavior in checks:
            if not observed(start_s, end_s, predicate):
                raise RuntimeError(f"SITL did not observe {behavior}")
            print(f"Mission objective passed: {behavior}.")
        if not exploratory:
            if max_north_velocity <= 0.02:
                raise RuntimeError(f"SITL did not observe forward following: {max_north_velocity:.2f}m/s")
            if min_north_velocity >= -0.05:
                raise RuntimeError(f"SITL did not observe obstacle backoff: {min_north_velocity:.2f}m/s")
            if max_east_velocity <= 0.02:
                raise RuntimeError(f"SITL did not observe lateral following: {max_east_velocity:.2f}m/s")
        if exploratory and max(
            abs(max_north_velocity),
            abs(min_north_velocity),
            abs(max_east_velocity),
            abs(min_east_velocity),
        ) > MAX_EXPLORATORY_SPEED_M_S:
            raise RuntimeError(
                "exploratory flight exceeded its telemetry speed envelope: "
                f"north={min_north_velocity:.2f}..{max_north_velocity:.2f}, "
                f"east={min_east_velocity:.2f}..{max_east_velocity:.2f}"
            )
        if camera or depth:
            if camera_frames == 0:
                raise RuntimeError("Gazebo did not provide a camera frame")
            print(f"Gazebo camera frames=verified ({camera_frames}).")
        if camera and not ollama:
            if any(command != VelocityCommand() for _, command in commands):
                raise RuntimeError(
                    "camera transport run commanded motion without a visual model"
                )
            print("Gazebo camera transport safe stop=verified.")
        if depth:
            if not valid_depth_samples:
                raise RuntimeError("Gazebo did not provide a valid depth reading")
            print(
                "Gazebo depth samples=verified: "
                f"{valid_depth_samples} valid of {depth_samples}."
            )
            print(f"Minimum Gazebo depth distance: {minimum_depth_distance:.2f}m")
        print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
        print(f"Max observed east velocity: {max_east_velocity:.2f}m/s")
        print(f"Min observed north velocity: {min_north_velocity:.2f}m/s")
        print(f"Observed east velocity range: {min_east_velocity:.2f}..{max_east_velocity:.2f}m/s")
        await land(drone)
        landed = True
    finally:
        receiver.close()
        if gazebo_camera is not None:
            gazebo_camera.close()
        if gazebo_depth is not None:
            gazebo_depth.close()
        if sender is not None:
            sender.close()
        if service_task is not None and not service_task.done():
            service_stop.set()
            await service_task
        mind_stop.set()
        if mind_task is not None and not mind_task.done():
            await mind_task
        if control is not None:
            control.close()
        if offboard_task is not None and not offboard_task.done():
            offboard_task.cancel()
            await asyncio.gather(offboard_task, return_exceptions=True)
        if offboard_started:
            try:
                await drone.offboard.stop()
            except Exception:
                pass
        if armed and not landed:
            try:
                await land(drone)
            except Exception:
                pass
        close_mavsdk(drone)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the synthetic companion world")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--vlm-model", default="moondream")
    parser.add_argument("--llm-model", default="gemma3:4b")
    parser.add_argument("--ollama-timeout", type=float, default=60.0)
    parser.add_argument("--memory", type=Path, help="persist conscious experience across runs")
    parser.add_argument(
        "--request",
        help="send one dialogue request automatically at the start of an exploratory run",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print meaningful VLM observations, conscious decisions, and command reasons",
    )
    parser.add_argument("--world", default="default")
    parser.add_argument("--duration", type=float, default=PROFILE_DURATION_S)
    parser.add_argument(
        "--intent",
        default=DEFAULT_EXPLORATORY_INTENT,
        help="initial high-level intent for an exploratory run",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            run(
                exploratory=args.explore,
                camera=args.camera,
                depth=args.depth,
                world_name=args.world,
                duration_s=args.duration,
                ollama=args.ollama,
                vlm_model=args.vlm_model,
                llm_model=args.llm_model,
                ollama_timeout=args.ollama_timeout,
                initial_intent=args.intent,
                memory_path=args.memory,
                dialogue_request=args.request,
                trace=args.trace,
            )
        )
    except (RuntimeError, ValueError) as error:
        print(f"simulation: {error}", file=sys.stderr)
        raise SystemExit(1)
