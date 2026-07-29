"""Small deterministic world for closed-loop PX4 behavior scenarios.

This is ground-truth perception for control testing, not a replacement for
camera or TOF hardware. It makes target motion and failures repeatable while
the vehicle still flies in Gazebo through the production command path.
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Optional

from mavsdk import System
from mavsdk.offboard import OffboardError

from control.runtime import CompanionRuntime
from control.state_machine import State
from control.tracking import TrackEstimate
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.ros2_bridge import LatestDistanceSensor
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from sim.flight import close_mavsdk, land, prepare, wait_for_offboard
from sim.offboard_control import (
    DistanceMessage,
    PROFILE_DURATION_S,
    SECOND_FOLLOW_END_S,
    SECOND_FOLLOW_START_S,
    SETPOINT_PERIOD_S,
    THIRD_FOLLOW_END_S,
    THIRD_FOLLOW_START_S,
    demo_state,
)


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
HOVER_START_S = 5.8


@dataclass(frozen=True)
class WorldStep:
    intent: State
    obstacle_distance_m: Optional[float] = 2.0
    transmit: bool = True
    command_override: Optional[VelocityCommand] = None


class SyntheticWorld:
    """Provide repeatable target, obstacle, and transport events for SITL."""

    def target(
        self,
        elapsed_s: float,
    ) -> Optional[TrackEstimate]:
        if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
            return None
        target_distance_m = 8.0
        target_offset_east_m = 0.8 if TARGET_RIGHT_START_S <= elapsed_s < TARGET_RIGHT_END_S else 0.0
        x_px = 320.0 + 320.0 * target_offset_east_m / target_distance_m
        return TrackEstimate(
            x_px=x_px,
            y_px=240.0,
            vx_px_s=0.0,
            vy_px_s=0.0,
            predicted_x_px=x_px,
            predicted_y_px=240.0,
            target_height_px=120.0 / target_distance_m,
        )

    def step(
        self,
        elapsed_s: float,
    ) -> WorldStep:
        intent = demo_state(elapsed_s)
        if OBSTACLE_START_S <= elapsed_s < OBSTACLE_END_S:
            return WorldStep(
                intent,
                obstacle_distance_m=0.3,
            )
        if OBSTACLE_END_S <= elapsed_s < RECOVERY_END_S:
            return WorldStep(State.FOLLOWING)
        if INVALID_SENSOR_START_S <= elapsed_s < INVALID_SENSOR_END_S:
            return WorldStep(State.FOLLOWING, obstacle_distance_m=math.nan)
        if DROPOUT_START_S <= elapsed_s < DROPOUT_END_S:
            return WorldStep(State.FOLLOWING, transmit=False)
        if DROPOUT_END_S <= elapsed_s < LINK_RECOVERY_END_S:
            return WorldStep(State.FOLLOWING)
        if INVALID_COMMAND_START_S <= elapsed_s < INVALID_COMMAND_END_S:
            return WorldStep(
                State.FOLLOWING,
                command_override=VelocityCommand(north_m_s=1.0),
            )
        return WorldStep(intent)


class WorldVision:
    """Expose synthetic-world target truth through the normal vision interface."""

    def __init__(self, world: SyntheticWorld, started_at_s: float):
        self.world = world
        self.started_at_s = started_at_s

    def process(self, frame, timestamp_s: float) -> Optional[TrackEstimate]:
        return self.world.target(max(0.0, timestamp_s - self.started_at_s))


async def run():
    """Run the companion recovery mission through the complete flight path."""

    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    service_task = None
    service_stop = asyncio.Event()
    telemetry_task = None
    offboard_task = None
    offboard_started = False
    armed = False
    landed = False
    distance_sensor = LatestDistanceSensor()
    safe_commands = []
    try:
        await prepare(drone)
        armed = True

        forwarder = MavsdkVelocityForwarder(drone)

        class ObservingForwarder:
            async def send(self, command):
                safe_commands.append((time.monotonic(), command))
                await forwarder.send(command)

        service = SafetyCommandService(
            receiver,
            ObservingForwarder(),
            tick_period_s=SETPOINT_PERIOD_S,
            obstacle_distance=distance_sensor.read,
        )
        service.start()
        service_task = asyncio.create_task(service.run(service_stop))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        started_at = time.monotonic()
        world = SyntheticWorld()
        world_vision = WorldVision(world, started_at)
        control = CompanionRuntime(
            world_vision
        )

        def send_packet(elapsed_s: float, timestamp_s: float):
            step = world.step(elapsed_s)
            command = control.tick(
                frame=None,
                timestamp_s=timestamp_s,
                intent=step.intent,
                obstacle_distance_m=step.obstacle_distance_m,
            )
            if step.transmit:
                sender.send(step.command_override or command)

        send_packet(0.0, time.monotonic())
        await asyncio.sleep(SETPOINT_PERIOD_S * 2)
        await drone.offboard.start()
        offboard_started = True
        offboard_task = asyncio.create_task(wait_for_offboard(drone))
        started_at = time.monotonic()
        world_vision.started_at_s = started_at
        print("Mission: follow, recover, handle faults, hover, land, and disarm.")

        max_north_velocity = 0.0
        min_north_velocity = 0.0
        max_east_velocity = 0.0

        async def observe_velocity():
            nonlocal max_north_velocity, min_north_velocity, max_east_velocity
            async for velocity in drone.telemetry.velocity_ned():
                max_north_velocity = max(max_north_velocity, velocity.north_m_s)
                min_north_velocity = min(min_north_velocity, velocity.north_m_s)
                max_east_velocity = max(max_east_velocity, velocity.east_m_s)

        telemetry_task = asyncio.create_task(observe_velocity())
        reported = set()
        try:
            while (elapsed := time.monotonic() - started_at) < PROFILE_DURATION_S:
                now = time.monotonic()
                step = world.step(elapsed)
                distance_sensor.update(
                    DistanceMessage(
                        step.obstacle_distance_m
                        if step.obstacle_distance_m is not None
                        else math.nan
                    )
                )
                event = (
                    "target moved right" if TARGET_RIGHT_START_S <= elapsed < TARGET_RIGHT_END_S else
                    "target lost; holding" if TARGET_LOST_START_S <= elapsed < TARGET_LOST_END_S else
                    "obstacle detected; backing off" if (
                        step.obstacle_distance_m is not None and step.obstacle_distance_m < 0.6
                    ) else
                    "invalid obstacle reading" if isinstance(step.obstacle_distance_m, float) and math.isnan(step.obstacle_distance_m) else
                    "command link dropout" if not step.transmit else
                    "out-of-bounds command" if step.command_override is not None else None
                )
                if (
                    SECOND_FOLLOW_START_S <= elapsed < SECOND_FOLLOW_END_S
                    and "intent changed back to following" not in reported
                ):
                    event = "intent changed back to following"
                elif (
                    THIRD_FOLLOW_START_S <= elapsed < THIRD_FOLLOW_END_S
                    and "intent changed to following again" not in reported
                ):
                    event = "intent changed to following again"
                elif elapsed >= HOVER_START_S and "intent changed to hover" not in reported:
                    event = "intent changed to hover"
                if event is not None and event not in reported:
                    print(event.capitalize() + ".")
                    reported.add(event)
                if CONTROL_PAUSE_START_S <= elapsed < CONTROL_PAUSE_END_S:
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
            telemetry_task.cancel()
            await asyncio.gather(telemetry_task, return_exceptions=True)
            if offboard_started:
                try:
                    await drone.offboard.stop()
                except OffboardError:
                    pass
                offboard_started = False

        commands = [(timestamp_s - started_at, command) for timestamp_s, command in safe_commands]
        if not safe_commands or safe_commands[-1][1] != VelocityCommand():
            raise RuntimeError("SITL did not observe zero command on CM5 shutdown")

        def forward_count(start_s, end_s):
            return sum(
                command.north_m_s > 0.0
                for elapsed_s, command in commands
                if start_s <= elapsed_s < end_s
            )

        print(
            "Following commands by cycle: "
            f"{forward_count(0.0, 4.0)}, "
            f"{forward_count(SECOND_FOLLOW_START_S, SECOND_FOLLOW_END_S)}, "
            f"{forward_count(THIRD_FOLLOW_START_S, THIRD_FOLLOW_END_S)}."
        )

        def observed(start_s, end_s, predicate):
            return any(start_s <= elapsed_s < end_s and predicate(command) for elapsed_s, command in commands)

        checks = (
            (0.0, 1.0, lambda command: command.north_m_s > 0.0, "forward following"),
            (TARGET_RIGHT_START_S, TARGET_RIGHT_END_S, lambda command: command.east_m_s > 0.0, "lateral target tracking"),
            (CONTROL_PAUSE_START_S, CONTROL_PAUSE_END_S + 0.1, lambda command: command == VelocityCommand(), "Mac heartbeat pause fail-safe"),
            (CONTROL_PAUSE_END_S + 0.1, TARGET_RIGHT_END_S, lambda command: command.east_m_s > 0.0, "Mac heartbeat recovery"),
            (2.1, TARGET_LOST_END_S, lambda command: command == VelocityCommand(), "hold after target loss"),
            (OBSTACLE_START_S, OBSTACLE_END_S, lambda command: command.north_m_s < 0.0, "obstacle backoff"),
            (OBSTACLE_END_S, RECOVERY_END_S, lambda command: command.north_m_s > 0.0, "following recovery after obstacle"),
            (INVALID_SENSOR_START_S, INVALID_SENSOR_END_S, lambda command: command == VelocityCommand(), "invalid obstacle fail-safe"),
            (DROPOUT_START_S + 0.15, DROPOUT_END_S, lambda command: command == VelocityCommand(), "command-dropout expiry"),
            (DROPOUT_END_S, LINK_RECOVERY_END_S, lambda command: command.north_m_s > 0.0, "command-link recovery"),
            (INVALID_COMMAND_START_S, INVALID_COMMAND_END_S, lambda command: command == VelocityCommand(), "invalid-command rejection"),
            (HOVER_START_S, SECOND_FOLLOW_START_S, lambda command: command == VelocityCommand(), "first hover"),
            (SECOND_FOLLOW_START_S, SECOND_FOLLOW_END_S, lambda command: command.north_m_s > 0.0, "following after hover"),
            (SECOND_FOLLOW_END_S, THIRD_FOLLOW_START_S, lambda command: command == VelocityCommand(), "second hover"),
            (THIRD_FOLLOW_START_S, THIRD_FOLLOW_END_S, lambda command: command.north_m_s > 0.0, "following after second hover"),
            (THIRD_FOLLOW_END_S, PROFILE_DURATION_S, lambda command: command == VelocityCommand(), "final hover"),
        )
        for start_s, end_s, predicate, behavior in checks:
            if not observed(start_s, end_s, predicate):
                raise RuntimeError(f"SITL did not observe {behavior}")
            print(f"Mission objective passed: {behavior}.")
        if max_north_velocity <= 0.02:
            raise RuntimeError(f"SITL did not observe forward following: {max_north_velocity:.2f}m/s")
        if min_north_velocity >= -0.05:
            raise RuntimeError(f"SITL did not observe obstacle backoff: {min_north_velocity:.2f}m/s")
        if max_east_velocity <= 0.02:
            raise RuntimeError(f"SITL did not observe lateral following: {max_east_velocity:.2f}m/s")
        print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
        print(f"Max observed east velocity: {max_east_velocity:.2f}m/s")
        print(f"Min observed north velocity: {min_north_velocity:.2f}m/s")
        await land(drone)
        landed = True
    finally:
        receiver.close()
        if sender is not None:
            sender.close()
        if service_task is not None and not service_task.done():
            service_stop.set()
            await service_task
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
    asyncio.run(run())
