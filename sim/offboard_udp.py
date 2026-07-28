"""Verify Mac command packets through a CM5 safety receiver into PX4 SITL."""

import asyncio
import socket
import time

from mavsdk import System
from mavsdk.offboard import OffboardError
from mavsdk.telemetry import LandedState

from control.command_packet import CommandPacket
from control.loop import CompanionControlLoop
from control.step import CompanionControlStep
from onboard.command_receiver import UdpSafetyReceiver
from onboard.velocity_forwarder import MavsdkVelocityForwarder
from control.velocity import VelocityCommand
from sim.offboard import (
    PROFILE_DURATION_S,
    SETPOINT_PERIOD_S,
    TAKEOFF_ALTITUDE,
    _wait_until_in_air,
)
from sim.offboard_control import DemoVision, demo_obstacle_distance_m, demo_state


COMMAND_DROP_START_S = 1.0
COMMAND_DROP_END_S = 1.5


async def run():
    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    drone = System()
    receiver.start()
    forwarder = None
    telemetry_task = None
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
        control_loop = CompanionControlLoop(CompanionControlStep(DemoVision()))
        sequence = 0

        def send_packet(elapsed_s: float, timestamp_s: float, transmit: bool = True):
            nonlocal sequence
            command = control_loop.tick(
                frame=None,
                timestamp_s=timestamp_s,
                intent=demo_state(elapsed_s),
                obstacle_distance_m=demo_obstacle_distance_m(elapsed_s),
            )
            if transmit:
                sender.sendto(
                    CommandPacket(sequence, command).encode(),
                    ("127.0.0.1", receiver.port),
                )
            sequence += 1

        send_packet(0.0, time.monotonic())
        safe_command = receiver.poll(obstacle_distance_m=demo_obstacle_distance_m(0.0))
        await forwarder.send(safe_command)
        await drone.offboard.start()
        print("Offboard started through UDP safety receiver.")

        max_north_velocity = 0.0
        min_north_velocity = 0.0

        async def observe_velocity():
            nonlocal max_north_velocity, min_north_velocity
            async for velocity in drone.telemetry.velocity_ned():
                max_north_velocity = max(max_north_velocity, velocity.north_m_s)
                min_north_velocity = min(min_north_velocity, velocity.north_m_s)

        telemetry_task = asyncio.create_task(observe_velocity())
        started_at = time.monotonic()
        obstacle_active = False
        dropout_reported = False
        dropout_safe = False
        try:
            while (elapsed := time.monotonic() - started_at) < PROFILE_DURATION_S:
                now = time.monotonic()
                distance = demo_obstacle_distance_m(elapsed)
                dropping_commands = COMMAND_DROP_START_S <= elapsed < COMMAND_DROP_END_S
                if dropping_commands and not dropout_reported:
                    print("Command link dropout; waiting for CM5 heartbeat expiry.")
                    dropout_reported = True
                if distance < 0.6 and not obstacle_active:
                    print("Obstacle detected; CM5 envelope backing off.")
                    obstacle_active = True
                elif distance >= 0.6:
                    obstacle_active = False
                send_packet(elapsed, now, transmit=not dropping_commands)
                safe_command = receiver.poll(obstacle_distance_m=distance)
                if dropping_commands and elapsed >= COMMAND_DROP_START_S + receiver.safety.command_timeout_s:
                    if safe_command != VelocityCommand():
                        raise RuntimeError(f"CM5 did not expire the dropped command: {safe_command}")
                    dropout_safe = True
                await forwarder.send(safe_command)
                await asyncio.sleep(SETPOINT_PERIOD_S)
        finally:
            sender.close()
            await forwarder.send(receiver.poll())
            if telemetry_task is not None:
                telemetry_task.cancel()
                await asyncio.gather(telemetry_task, return_exceptions=True)
            try:
                await drone.offboard.stop()
            except OffboardError:
                pass

        if not dropout_safe:
            raise RuntimeError("SITL did not observe CM5 zero output during command link dropout")
        if min_north_velocity >= -0.05:
            raise RuntimeError(f"SITL did not observe CM5 obstacle backoff: {min_north_velocity:.2f}m/s")
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
        sender.close()


if __name__ == "__main__":
    asyncio.run(run())
