"""Shared PX4 flight steps for the simulations."""

import asyncio
import time

from mavsdk.telemetry import FlightMode
from mavsdk.telemetry import LandedState


TAKEOFF_ALTITUDE = 2.0
PREPARE_TIMEOUT_S = 30.0


class RecordingForwarder:
    """Forward commands and keep them for later checks."""

    def __init__(self, forwarder):
        self.forwarder = forwarder
        self.commands = []

    async def send(self, command):
        await self.forwarder.send(command)
        self.commands.append((time.monotonic(), command))


async def _wait_for_connection(drone):
    async for state in drone.core.connection_state():
        if state.is_connected:
            return


async def _wait_until_ready(drone):
    async for health in drone.telemetry.health():
        if (
            health.is_armable
            and health.is_local_position_ok
            and health.is_magnetometer_calibration_ok
        ):
            return


async def prepare(drone):
    """Connect, wait for arming health, arm, and reach the air."""

    print("Waiting for drone connection...")
    await drone.connect()
    try:
        await asyncio.wait_for(_wait_for_connection(drone), PREPARE_TIMEOUT_S)
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            f"drone did not connect within {PREPARE_TIMEOUT_S:.0f}s"
        ) from error
    print("Connected.")

    print("Waiting for vehicle to be ready to arm...")
    try:
        await asyncio.wait_for(_wait_until_ready(drone), PREPARE_TIMEOUT_S)
    except asyncio.TimeoutError as error:
        raise RuntimeError(
            f"vehicle did not become ready to arm within {PREPARE_TIMEOUT_S:.0f}s"
        ) from error
    print("Ready.")

    print("Arming...")
    try:
        await drone.action.arm()
        print(f"Taking off to {TAKEOFF_ALTITUDE}m...")
        await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
        await drone.action.takeoff()
        async for state in drone.telemetry.landed_state():
            if state == LandedState.IN_AIR:
                return
    except Exception:
        try:
            await drone.action.land()
        except Exception:
            pass
        raise


async def land(drone):
    """Land and wait for PX4 to report the vehicle on the ground."""

    print("Landing...")
    await drone.action.land()
    async for state in drone.telemetry.landed_state():
        if state == LandedState.ON_GROUND:
            break
    print("Landed.")
    async for armed in drone.telemetry.armed():
        if not armed:
            print("Disarmed.")
            return


async def wait_for_offboard(drone):
    """Wait for PX4 telemetry to confirm offboard mode is active."""

    async for mode in drone.telemetry.flight_mode():
        if mode is FlightMode.OFFBOARD:
            return


def close_mavsdk(drone):
    """Stop the MAVSDK helper started by the Python binding."""

    drone._stop_mavsdk_server()
