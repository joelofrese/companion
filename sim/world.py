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
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from sim.flight import close_mavsdk, land, prepare
from sim.offboard_control import (
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
TARGET_LOST_START_S = 2.0
TARGET_LOST_END_S = 2.6
OBSTACLE_START_S = 2.8
OBSTACLE_END_S = 3.7
RECOVERY_END_S = 3.9
INVALID_SENSOR_START_S = 3.9
INVALID_SENSOR_END_S = 4.2
DROPOUT_START_S = 4.2
DROPOUT_END_S = 4.7
LINK_RECOVERY_END_S = 4.85
INVALID_COMMAND_START_S = 4.85
INVALID_COMMAND_END_S = 5.15
HOVER_START_S = 5.2


@dataclass(frozen=True)
class WorldStep:
    intent: State
    obstacle_distance_m: Optional[float] = None
    transmit: bool = True
    command_override: Optional[VelocityCommand] = None


class SyntheticWorld:
    """Provide repeatable target, obstacle, and transport events for SITL."""

    def target(
        self,
        elapsed_s: float,
        vehicle_position_m: Optional[tuple[float, float]],
    ) -> Optional[TrackEstimate]:
        if TARGET_LOST_START_S <= elapsed_s < TARGET_LOST_END_S:
            return None
        if vehicle_position_m is None:
            return None
        north_m, east_m = vehicle_position_m
        target_north_m = 8.0
        target_east_m = 0.8 if TARGET_RIGHT_START_S <= elapsed_s < TARGET_RIGHT_END_S else 0.0
        north_distance_m = target_north_m - north_m
        if north_distance_m <= 0.0:
            return None
        x_px = 320.0 + 320.0 * (target_east_m - east_m) / north_distance_m
        return TrackEstimate(
            x_px=x_px,
            y_px=240.0,
            vx_px_s=0.0,
            vy_px_s=0.0,
            predicted_x_px=x_px,
            predicted_y_px=240.0,
            target_height_px=120.0 / north_distance_m,
        )

    def step(
        self,
        elapsed_s: float,
        vehicle_position_m: Optional[tuple[float, float]] = None,
    ) -> WorldStep:
        intent = demo_state(elapsed_s)
        if OBSTACLE_START_S <= elapsed_s < OBSTACLE_END_S:
            north_m = vehicle_position_m[0] if vehicle_position_m is not None else 0.0
            return WorldStep(
                intent,
                obstacle_distance_m=max(0.2, 0.5 - north_m),
            )
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

    def __init__(self, world: SyntheticWorld, started_at_s: float, position):
        self.world = world
        self.started_at_s = started_at_s
        self.position = position

    def process(self, frame, timestamp_s: float) -> Optional[TrackEstimate]:
        return self.world.target(
            max(0.0, timestamp_s - self.started_at_s),
            self.position(),
        )


async def run():
    """Run the companion recovery mission through the complete flight path."""

    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    service_task = None
    service_stop = asyncio.Event()
    telemetry_task = None
    position_task = None
    offboard_started = False
    armed = False
    landed = False
    obstacle_distance_m = None
    safe_commands = []
    try:
        await prepare(drone)
        armed = True

        origin = None
        vehicle_position_m = None
        position_ready = asyncio.Event()

        async def observe_position():
            nonlocal origin, vehicle_position_m
            async for position_velocity in drone.telemetry.position_velocity_ned():
                position = position_velocity.position
                if origin is None:
                    origin = position.north_m, position.east_m
                vehicle_position_m = (
                    position.north_m - origin[0],
                    position.east_m - origin[1],
                )
                position_ready.set()

        position_task = asyncio.create_task(observe_position())
        await asyncio.wait_for(position_ready.wait(), timeout=5.0)

        forwarder = MavsdkVelocityForwarder(drone)

        class ObservingForwarder:
            async def send(self, command):
                safe_commands.append((time.monotonic(), command))
                await forwarder.send(command)

        service = SafetyCommandService(
            receiver,
            ObservingForwarder(),
            tick_period_s=SETPOINT_PERIOD_S,
            obstacle_distance=lambda: obstacle_distance_m,
        )
        service.start()
        service_task = asyncio.create_task(service.run(service_stop))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        started_at = time.monotonic()
        world = SyntheticWorld()
        control = CompanionRuntime(
            WorldVision(world, started_at, lambda: vehicle_position_m)
        )

        def send_packet(elapsed_s: float, timestamp_s: float):
            step = world.step(elapsed_s, vehicle_position_m)
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
        print("Mission: follow, recover, handle faults, hover, land, and disarm.")
        print("Offboard started through synthetic world and CM5 safety.")

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
                step = world.step(elapsed, vehicle_position_m)
                obstacle_distance_m = step.obstacle_distance_m
                event = (
                    "target moved right" if TARGET_RIGHT_START_S <= elapsed < TARGET_RIGHT_END_S else
                    "target lost; holding" if TARGET_LOST_START_S <= elapsed < TARGET_LOST_END_S else
                    "obstacle detected; backing off" if (
                        obstacle_distance_m is not None and obstacle_distance_m < 0.6
                    ) else
                    "invalid obstacle reading" if isinstance(obstacle_distance_m, float) and math.isnan(obstacle_distance_m) else
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
                send_packet(elapsed, now)
                await asyncio.sleep(SETPOINT_PERIOD_S)
        finally:
            sender.close()
            service_stop.set()
            await service_task
            telemetry_task.cancel()
            await asyncio.gather(telemetry_task, return_exceptions=True)
            position_task.cancel()
            await asyncio.gather(position_task, return_exceptions=True)
            if offboard_started:
                try:
                    await drone.offboard.stop()
                except OffboardError:
                    pass
                offboard_started = False

        commands = [(timestamp_s - started_at, command) for timestamp_s, command in safe_commands]
        if not safe_commands or safe_commands[-1][1] != VelocityCommand():
            raise RuntimeError("SITL did not observe zero command on CM5 shutdown")

        def observed(start_s, end_s, predicate):
            return any(start_s <= elapsed_s < end_s and predicate(command) for elapsed_s, command in commands)

        checks = (
            (0.0, 1.0, lambda command: command.north_m_s > 0.0, "forward following"),
            (TARGET_RIGHT_START_S, TARGET_RIGHT_END_S, lambda command: command.east_m_s > 0.0, "lateral target tracking"),
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
        if position_task is not None:
            position_task.cancel()
            await asyncio.gather(position_task, return_exceptions=True)
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
