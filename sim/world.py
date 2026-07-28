"""Small deterministic world for closed-loop PX4 behavior scenarios.

This is ground-truth perception for control testing, not a replacement for
camera or TOF hardware. It makes target motion and failures repeatable while
the vehicle still flies in Gazebo through the production command path.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from mavsdk import System
from mavsdk.offboard import OffboardError
from mavsdk.telemetry import LandedState

from control.runtime import CompanionRuntime
from control.state_machine import State
from control.tracking import TrackEstimate
from control.udp_sender import UdpCommandSender
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from sim.offboard import PROFILE_DURATION_S, SETPOINT_PERIOD_S, TAKEOFF_ALTITUDE, _wait_until_in_air


@dataclass(frozen=True)
class WorldStep:
    intent: State
    obstacle_distance_m: Optional[float] = None
    transmit: bool = True
    command_override: Optional[VelocityCommand] = None


def _target(x_px: float = 320.0) -> TrackEstimate:
    return TrackEstimate(
        x_px=x_px,
        y_px=240.0,
        vx_px_s=0.0,
        vy_px_s=0.0,
        predicted_x_px=x_px,
        predicted_y_px=240.0,
        target_height_px=60.0,
    )


class SyntheticWorld:
    """Provide repeatable target, obstacle, and transport events for SITL."""

    def target(self, elapsed_s: float) -> Optional[TrackEstimate]:
        if 2.0 <= elapsed_s < 2.6:
            return None
        if 1.0 <= elapsed_s < 2.0:
            return _target(520.0)
        return _target()

    def step(self, elapsed_s: float) -> WorldStep:
        if 2.6 <= elapsed_s < 3.6:
            return WorldStep(State.FOLLOWING, obstacle_distance_m=0.5)
        if 3.6 <= elapsed_s < 4.1:
            return WorldStep(State.FOLLOWING, transmit=False)
        if 4.1 <= elapsed_s < 4.4:
            return WorldStep(
                State.FOLLOWING,
                command_override=VelocityCommand(north_m_s=1.0),
            )
        return WorldStep(State.FOLLOWING if elapsed_s < 5.0 else State.HOVERING)


class WorldVision:
    """Expose synthetic-world target truth through the normal vision interface."""

    def __init__(self, world: SyntheticWorld, started_at_s: float):
        self.world = world
        self.started_at_s = started_at_s

    def process(self, frame, timestamp_s: float) -> Optional[TrackEstimate]:
        return self.world.target(max(0.0, timestamp_s - self.started_at_s))


async def run():
    """Fly the synthetic world through Mac control, CM5 safety, and PX4 SITL."""

    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = None
    drone = System()
    service_task = None
    service_stop = asyncio.Event()
    telemetry_task = None
    obstacle_distance_m = None
    safe_commands = []
    try:
        print("Waiting for drone connection...")
        await drone.connect()
        async for state in drone.core.connection_state():
            if state.is_connected:
                print("Connected.")
                break

        print("Waiting for vehicle to be ready to arm...")
        async for health in drone.telemetry.health():
            if (health.is_global_position_ok
                    and health.is_home_position_ok
                    and health.is_magnetometer_calibration_ok):
                print("Ready.")
                break

        print("Arming...")
        await drone.action.arm()
        await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
        await drone.action.takeoff()
        await _wait_until_in_air(drone)

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
        control = CompanionRuntime(WorldVision(world, started_at))

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
                step = world.step(elapsed)
                obstacle_distance_m = step.obstacle_distance_m
                event = (
                    "target moved right" if 1.0 <= elapsed < 2.0 else
                    "target lost; holding" if 2.0 <= elapsed < 2.6 else
                    "obstacle detected; backing off" if obstacle_distance_m == 0.5 else
                    "command link dropout" if not step.transmit else
                    "out-of-bounds command" if step.command_override is not None else None
                )
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
            (2.6, 3.6, lambda command: command.north_m_s < 0.0, "obstacle backoff"),
            (3.8, 4.1, lambda command: command == VelocityCommand(), "command-dropout expiry"),
            (4.1, 4.4, lambda command: command == VelocityCommand(), "invalid-command rejection"),
        )
        for start_s, end_s, predicate, behavior in checks:
            if not observed(start_s, end_s, predicate):
                raise RuntimeError(f"SITL did not observe {behavior}")
        if min_north_velocity >= -0.05:
            raise RuntimeError(f"SITL did not observe obstacle backoff: {min_north_velocity:.2f}m/s")
        print(f"Max observed north velocity: {max_north_velocity:.2f}m/s")
        print(f"Min observed north velocity: {min_north_velocity:.2f}m/s")
        print("Landing...")
        await drone.action.land()
        async for state in drone.telemetry.landed_state():
            if state == LandedState.ON_GROUND:
                print("Landed.")
                break
    finally:
        receiver.close()
        if sender is not None:
            sender.close()
        if service_task is not None and not service_task.done():
            service_stop.set()
            await service_task


if __name__ == "__main__":
    asyncio.run(run())
