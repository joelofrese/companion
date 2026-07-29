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
from sim.offboard import PROFILE_DURATION_S, SETPOINT_PERIOD_S


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
        if 2.0 <= elapsed_s < 2.6:
            return None
        if vehicle_position_m is None:
            return None
        north_m, east_m = vehicle_position_m
        target_north_m = 1.6
        target_east_m = 0.8 if 1.0 <= elapsed_s < 2.0 else 0.0
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
        if 2.8 <= elapsed_s < 3.8:
            north_m = vehicle_position_m[0] if vehicle_position_m is not None else 0.0
            return WorldStep(
                State.FOLLOWING,
                obstacle_distance_m=max(0.2, 0.5 - north_m),
            )
        if 3.9 <= elapsed_s < 4.0:
            return WorldStep(State.FOLLOWING, obstacle_distance_m=math.nan)
        if 4.0 <= elapsed_s < 4.5:
            return WorldStep(State.FOLLOWING, transmit=False)
        if 4.65 <= elapsed_s < 4.95:
            return WorldStep(
                State.FOLLOWING,
                command_override=VelocityCommand(north_m_s=1.0),
            )
        return WorldStep(State.FOLLOWING if elapsed_s < 5.0 else State.HOVERING)


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
    """Fly the synthetic world through Mac control, CM5 safety, and PX4 SITL."""

    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    service_task = None
    service_stop = asyncio.Event()
    telemetry_task = None
    position_task = None
    obstacle_distance_m = None
    safe_commands = []
    try:
        await prepare(drone)

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
        print("Offboard started through synthetic world and CM5 safety.")

        max_north_velocity = 0.0
        min_north_velocity = 0.0

        async def observe_velocity():
            nonlocal max_north_velocity, min_north_velocity
            async for velocity in drone.telemetry.velocity_ned():
                max_north_velocity = max(max_north_velocity, velocity.north_m_s)
                min_north_velocity = min(min_north_velocity, velocity.north_m_s)

        telemetry_task = asyncio.create_task(observe_velocity())
        reported = set()
        try:
            while (elapsed := time.monotonic() - started_at) < PROFILE_DURATION_S:
                now = time.monotonic()
                step = world.step(elapsed, vehicle_position_m)
                obstacle_distance_m = step.obstacle_distance_m
                event = (
                    "target moved right" if 1.0 <= elapsed < 2.0 else
                    "target lost; holding" if 2.0 <= elapsed < 2.6 else
                    "obstacle detected; backing off" if (
                        obstacle_distance_m is not None and obstacle_distance_m < 0.6
                    ) else
                    "invalid obstacle reading" if isinstance(obstacle_distance_m, float) and math.isnan(obstacle_distance_m) else
                    "command link dropout" if not step.transmit else
                    "out-of-bounds command" if step.command_override is not None else None
                )
                if elapsed >= 5.0 and "intent changed to hover" not in reported:
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
            try:
                await drone.offboard.stop()
            except OffboardError:
                pass

        commands = [(timestamp_s - started_at, command) for timestamp_s, command in safe_commands]

        def observed(start_s, end_s, predicate):
            return any(start_s <= elapsed_s < end_s and predicate(command) for elapsed_s, command in commands)

        checks = (
            (0.0, 1.0, lambda command: command.north_m_s > 0.0, "forward following"),
            (1.0, 2.0, lambda command: command.east_m_s > 0.0, "lateral target tracking"),
            (2.1, 2.6, lambda command: command == VelocityCommand(), "hold after target loss"),
            (2.8, 3.8, lambda command: command.north_m_s < 0.0, "obstacle backoff"),
            (3.8, 4.0, lambda command: command.north_m_s > 0.0, "following recovery after obstacle"),
            (3.9, 4.0, lambda command: command == VelocityCommand(), "invalid obstacle fail-safe"),
            (4.15, 4.5, lambda command: command == VelocityCommand(), "command-dropout expiry"),
            (4.5, 4.65, lambda command: command.north_m_s > 0.0, "command-link recovery"),
            (4.65, 4.95, lambda command: command == VelocityCommand(), "invalid-command rejection"),
            (5.0, 5.8, lambda command: command == VelocityCommand(), "hover recovery"),
        )
        for start_s, end_s, predicate, behavior in checks:
            if not observed(start_s, end_s, predicate):
                raise RuntimeError(f"SITL did not observe {behavior}")
        if min_north_velocity >= -0.05:
            raise RuntimeError(f"SITL did not observe obstacle backoff: {min_north_velocity:.2f}m/s")
        print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
        print(f"Min observed north velocity: {min_north_velocity:.2f}m/s")
        await land(drone)
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
        close_mavsdk(drone)


if __name__ == "__main__":
    asyncio.run(run())
