"""A small fixed world for PX4 behavior checks.

It supplies repeatable targets and faults while the vehicle flies in Gazebo
through the real command path.
"""

import asyncio
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from mavsdk import System
from mavsdk.offboard import OffboardError

from control.mind import ConsciousDecision, MacMind, Telemetry, VisualObservation
from control.mind_runtime import MindRuntime
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.ros2_bridge import LatestDistanceSensor
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from sim.flight import RecordingForwarder, close_mavsdk, land, prepare, wait_for_offboard
from sim.gazebo_camera import GazeboCamera
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
STALE_SENSOR_START_S = 15.0
STALE_SENSOR_END_S = 15.5
HOVER_START_S = 5.8
MAX_EXPLORATORY_SPEED_M_S = 1.0


@dataclass(frozen=True)
class WorldStep:
    obstacle_distance_m: Optional[float] = 2.0
    distance_fresh: bool = True
    transmit: bool = True
    command_override: Optional[VelocityCommand] = None


class DialogueInput:
    """Read live dialogue without blocking the flight loop."""

    def __init__(self):
        self._messages = queue.SimpleQueue()

    def start(self):
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        print("Dialogue is live. Type follow me, hover, or stop.", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if line:
                self._messages.put(line)

    def next(self) -> Optional[str]:
        try:
            return self._messages.get_nowait()
        except queue.Empty:
            return None


class SyntheticWorld:
    """Provide fixed targets, sensors, and link faults."""

    def __init__(self, exploratory: bool = False):
        self.exploratory = exploratory

    def target_offset_east(self, elapsed_s: float) -> Optional[float]:
        if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
            return None
        return 0.8 if TARGET_RIGHT_START_S <= elapsed_s < TARGET_RIGHT_END_S else 0.0

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
            return WorldStep(command_override=VelocityCommand(north_m_s=1.0))
        return WorldStep()


class WorldVisualModel:
    """Provide fixed scene descriptions to the Mac VLM boundary."""

    def __init__(self, world: SyntheticWorld, started_at_s: float):
        self.world = world
        self.started_at_s = started_at_s

    def observe(
        self,
        image,
        timestamp_s: float,
        focus: str,
        intent: str,
        previous_movement: str,
        previous_observation: str,
        telemetry: Telemetry,
    ) -> VisualObservation:
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
        if intent != "following":
            movement = "stop"
        return VisualObservation(
            timestamp_s=timestamp_s,
            description=description,
            focused_answer=description if focus else "",
            movement=movement,
            next_focus=focus or "person",
            confidence=1.0,
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
            print(dialogue, flush=True)
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
            focus="person" if self.intent == "following" else "",
            dialogue=dialogue,
            summary=summary or "The simulated world is running.",
        )


async def run(
    exploratory: bool = False,
    camera: bool = False,
    world_name: str = "default",
    duration_s: float = PROFILE_DURATION_S,
):
    """Run the complete synthetic mission."""

    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("simulation duration must be positive")
    if not exploratory and duration_s < PROFILE_DURATION_S:
        raise ValueError("deterministic simulation duration cannot be shorter than its profile")

    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    service_task = None
    mind_task = None
    service_stop = asyncio.Event()
    mind_stop = asyncio.Event()
    telemetry_task = None
    offboard_task = None
    dialogue_input = DialogueInput() if exploratory else None
    gazebo_camera = None
    control = None
    offboard_started = False
    armed = False
    landed = False
    distance_sensor = LatestDistanceSensor()
    try:
        await prepare(drone)
        armed = True

        forwarder = MavsdkVelocityForwarder(drone)

        safe_commands = RecordingForwarder(forwarder)

        service = SafetyCommandService(
            receiver,
            safe_commands,
            tick_period_s=SETPOINT_PERIOD_S,
            obstacle_distance=distance_sensor.read,
        )
        service.start()
        service_task = asyncio.create_task(service.run(service_stop))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        started_at = time.monotonic()
        world = SyntheticWorld(exploratory)
        if camera:
            gazebo_camera = GazeboCamera(
                f"/world/{world_name}/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
            )
            gazebo_camera.start()
        visual_model = WorldVisualModel(world, started_at)
        language_model = WorldLanguageModel(exploratory)
        language_model.started_at_s = started_at
        control = MindRuntime(MacMind(visual_model, language_model))

        camera_frames = 0

        def send_packet(elapsed_s: float, timestamp_s: float, intent=None):
            nonlocal camera_frames
            step = world.step(elapsed_s)
            frame = gazebo_camera.latest() if gazebo_camera else step
            if gazebo_camera is not None and frame is not None:
                camera_frames += 1
            command = control.tick(
                frame=frame,
                timestamp_s=timestamp_s,
                intent=intent,
                obstacle_distance_m=step.obstacle_distance_m,
            )
            if step.transmit:
                sender.send(step.command_override or command)

        send_packet(0.0, time.monotonic(), intent="following")
        deadline = time.monotonic() + 5.0
        while not safe_commands.commands and time.monotonic() < deadline:
            await asyncio.sleep(SETPOINT_PERIOD_S)
        if not safe_commands.commands:
            raise RuntimeError("CM5 did not forward a priming setpoint")
        print("CM5 priming setpoint=verified.")
        await asyncio.sleep(SETPOINT_PERIOD_S * 2)
        await drone.offboard.start()
        offboard_started = True
        offboard_task = asyncio.create_task(wait_for_offboard(drone))
        started_at = time.monotonic()
        visual_model.started_at_s = started_at
        language_model.started_at_s = started_at
        mind_task = asyncio.create_task(
            control.think_loop(
                mind_stop,
                telemetry_provider=lambda: Telemetry(
                    obstacle_distance_m=distance_sensor.read()
                ),
                dialogue_provider=dialogue_input.next if dialogue_input else None,
                period_s=1.0,
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

        async def observe_velocity():
            nonlocal max_north_velocity, min_north_velocity
            nonlocal max_east_velocity, min_east_velocity
            async for velocity in drone.telemetry.velocity_ned():
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
            if step.obstacle_distance_m is not None and step.obstacle_distance_m < 0.6:
                return "obstacle detected; backing off"
            if not step.distance_fresh:
                return "distance sensor dropout"
            if isinstance(step.obstacle_distance_m, float) and math.isnan(step.obstacle_distance_m):
                return "invalid obstacle reading"
            if not step.transmit:
                return "command link dropout"
            if step.command_override is not None:
                return "out-of-bounds command"
            if SECOND_FOLLOW_START_S <= elapsed_s < SECOND_FOLLOW_END_S:
                return "intent changed back to following"
            if THIRD_FOLLOW_START_S <= elapsed_s < THIRD_FOLLOW_END_S:
                return "intent changed to following again"
            if elapsed_s >= HOVER_START_S:
                return "intent changed to hover"
            return None

        try:
            while (elapsed := time.monotonic() - started_at) < duration_s:
                now = time.monotonic()
                step = world.step(elapsed)
                if step.distance_fresh:
                    distance_sensor.update(
                        DistanceMessage(
                            step.obstacle_distance_m
                            if step.obstacle_distance_m is not None
                            else math.nan
                        )
                    )
                event = None if exploratory else event_for(elapsed, step)
                if event is not None and event not in reported:
                    print(event.capitalize() + ".")
                    reported.add(event)
                if not exploratory and CONTROL_PAUSE_START_S <= elapsed < CONTROL_PAUSE_END_S:
                    if "Mac control pause" not in reported:
                        print("Mac control pause; watchdog holds zero.")
                        reported.add("Mac control pause")
                    await asyncio.sleep(CONTROL_PAUSE_END_S - elapsed)
                    continue
                send_packet(elapsed, now)
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
            if offboard_started:
                try:
                    await drone.offboard.stop()
                except OffboardError:
                    pass
                offboard_started = False

        commands = safe_commands.commands
        if not commands or commands[-1][1] != VelocityCommand():
            raise RuntimeError("SITL did not observe zero command on CM5 shutdown")

        decision = control.latest_decision
        if decision is None:
            raise RuntimeError("SITL did not observe a conscious Mac decision")
        if control.latest_observation is None:
            raise RuntimeError("SITL did not observe a Mac visual observation")
        if not decision.summary:
            raise RuntimeError("SITL did not retain a conscious visual summary")
        print(
            "Conscious Mac decision=verified: "
            f"intent={decision.intent}, focus={decision.focus or 'none'}."
        )
        print("Conscious visual memory=verified.")
        print("Mac visual observation=verified.")

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
        if camera:
            if camera_frames == 0:
                raise RuntimeError("Gazebo did not provide a camera frame")
            print(f"Gazebo camera frames=verified ({camera_frames}).")
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

    parser = argparse.ArgumentParser(description="Run the synthetic companion world")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--world", default="default")
    parser.add_argument("--duration", type=float, default=PROFILE_DURATION_S)
    args = parser.parse_args()
    asyncio.run(
        run(
            exploratory=args.explore,
            camera=args.camera,
            world_name=args.world,
            duration_s=args.duration,
        )
    )
