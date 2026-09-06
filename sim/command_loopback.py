"""Verify the brain-to-CM5 command link over local UDP."""

import asyncio
import math
import time

from control.udp_sender import UdpCommandSender
from control.udp_control import UdpControlService
from control.safety_limits import OBSTACLE_STOP_M
from control.telemetry import Telemetry
from control.velocity import VelocityCommand
from onboard.command_receiver import UdpSafetyReceiver
from onboard.command_service import SafetyCommandService


class FixedCommandControl:
    """Provide one normal brain command for the transport loopback."""

    def __init__(self):
        self.obstacle_distances = []
        self.last_commands = []

    def tick(self, frame, timestamp_s, intent=None, telemetry=Telemetry()):
        self.obstacle_distances.append(telemetry.obstacle_distance_m)
        self.last_commands.append(telemetry.last_command)
        return VelocityCommand(forward_m_s=0.25)


class RecordingForwarder:
    def __init__(self):
        self.commands = asyncio.Queue()

    async def send(self, command):
        await self.commands.put(command)


async def _next_command(forwarder):
    return await asyncio.wait_for(forwarder.commands.get(), timeout=1.0)


async def run():
    receiver = UdpSafetyReceiver(bind_host="127.0.0.1", port=0)
    forwarder = RecordingForwarder()
    cm5_stop_event = asyncio.Event()
    brain_stop_event = asyncio.Event()
    frame_count = 0
    cm5_started_at = None
    obstacle_cleared = False
    turn_obstacle = False

    def cm5_obstacle_distance():
        if turn_obstacle:
            return 0.5
        if obstacle_cleared:
            return 2.0
        elapsed_s = time.monotonic() - cm5_started_at
        if elapsed_s < 0.08:
            return None
        return 2.0 if elapsed_s < 0.12 else 0.5

    def cm5_velocity():
        return (0.12, -0.04, 0.03, 0.75)

    async def frame_reader():
        nonlocal frame_count
        frame_count += 1
        await asyncio.sleep(0.05)
        if frame_count >= 3:
            brain_stop_event.set()
        return time.monotonic(), None

    cm5_service = SafetyCommandService(
        receiver,
        forwarder,
        obstacle_distance=cm5_obstacle_distance,
        velocity_provider=cm5_velocity,
    )
    cm5_task = None
    brain_task = None
    sender = None
    reconnect_sender = None
    control = FixedCommandControl()
    try:
        cm5_service.start()
        cm5_started_at = time.monotonic()
        cm5_task = asyncio.create_task(cm5_service.run(cm5_stop_event))
        sender = UdpCommandSender("127.0.0.1", receiver.port)
        brain_task = asyncio.create_task(
            UdpControlService(
                control,
                sender,
                frame_reader,
                telemetry_provider=sender.telemetry,
                tick_period_s=0.01,
            ).run(brain_stop_event)
        )
        await brain_task
        await asyncio.sleep(0.05)
        commands = []
        while not forwarder.commands.empty():
            commands.append(forwarder.commands.get_nowait())
        if VelocityCommand(forward_m_s=0.25) not in commands:
            raise RuntimeError(
                "Brain control service did not produce a fresh follow command: "
                f"{commands}"
            )
        if VelocityCommand() not in commands:
            raise RuntimeError("CM5 did not stop when obstacle data was missing")
        if VelocityCommand(forward_m_s=-0.2) not in commands:
            raise RuntimeError(f"Brain control service did not produce obstacle backoff: {commands}")
        if VelocityCommand(forward_m_s=0.25) not in control.last_commands:
            raise RuntimeError(
                "Brain did not receive its previous command through telemetry: "
                f"{control.last_commands}"
            )
        if not any(
            distance is not None
            and not (isinstance(distance, float) and math.isnan(distance))
            and distance > 1.0
            for distance in control.obstacle_distances
        ):
            raise RuntimeError(
                f"Brain did not receive a clear CM5 distance: {control.obstacle_distances}"
            )
        if not any(
            distance is not None
            and not (isinstance(distance, float) and math.isnan(distance))
            and distance < OBSTACLE_STOP_M
            for distance in control.obstacle_distances
        ):
            raise RuntimeError(
                f"Brain did not receive the CM5 obstacle distance: {control.obstacle_distances}"
            )
        telemetry = sender.telemetry()
        if (
            telemetry.forward_velocity_m_s != 0.12
            or telemetry.right_velocity_m_s != -0.04
            or telemetry.down_velocity_m_s != 0.03
            or telemetry.heading_rad != 0.75
        ):
            raise RuntimeError(f"Brain did not receive CM5 velocity telemetry: {telemetry}")
        obstacle_cleared = True
        await asyncio.sleep(0.2)
        while not forwarder.commands.empty():
            forwarder.commands.get_nowait()
        reconnect_sender = UdpCommandSender("127.0.0.1", receiver.port)
        reconnect_sender.send(VelocityCommand(forward_m_s=0.25))
        await asyncio.sleep(0.1)
        reconnect_commands = []
        while not forwarder.commands.empty():
            reconnect_commands.append(forwarder.commands.get_nowait())
        if VelocityCommand(forward_m_s=0.25) not in reconnect_commands:
            raise RuntimeError(
                "a restarted brain sender did not reconnect through CM5 safety: "
                f"{reconnect_commands}"
            )
        print("Brain sender restart=verified.")
        while not forwarder.commands.empty():
            forwarder.commands.get_nowait()
        cm5_stop_event.set()
        await cm5_task
        cm5_stop_event = asyncio.Event()
        cm5_service.start()
        cm5_task = asyncio.create_task(cm5_service.run(cm5_stop_event))
        reconnect_sender.send(VelocityCommand(forward_m_s=0.25))
        restarted_telemetry = None
        for _ in range(20):
            await asyncio.sleep(0.02)
            restarted_telemetry = reconnect_sender.telemetry()
            if restarted_telemetry.obstacle_distance_m == 2.0:
                break
        if (
            restarted_telemetry is None
            or restarted_telemetry.obstacle_distance_m != 2.0
            or restarted_telemetry.forward_velocity_m_s != 0.12
            or restarted_telemetry.right_velocity_m_s != -0.04
            or restarted_telemetry.down_velocity_m_s != 0.03
            or restarted_telemetry.heading_rad != 0.75
        ):
            raise RuntimeError(
                "Brain did not receive fresh telemetry after CM5 restart: "
                f"{restarted_telemetry}"
            )
        print("CM5 restart=verified.")
        while not forwarder.commands.empty():
            forwarder.commands.get_nowait()
        turn_obstacle = True
        reconnect_sender.send(VelocityCommand(yaw_rate_deg_s=8.0))
        await asyncio.sleep(0.1)
        turn_commands = []
        while not forwarder.commands.empty():
            turn_commands.append(forwarder.commands.get_nowait())
        if VelocityCommand(yaw_rate_deg_s=8.0) not in turn_commands:
            raise RuntimeError(
                "CM5 did not preserve an in-place turn near an obstacle: "
                f"{turn_commands}"
            )
        print("CM5 in-place turn around obstacle=verified.")
        turn_obstacle = False
        while not forwarder.commands.empty():
            forwarder.commands.get_nowait()
        cm5_stop_event.set()
        await cm5_task
        shutdown_command = await _next_command(forwarder)
        if shutdown_command != VelocityCommand():
            raise RuntimeError("command service did not send zero on shutdown")
        print(
            "Brain control service=verified; CM5 telemetry=verified; "
            "velocity telemetry=verified; missing sensor=zero; "
            "obstacle command=verified; shutdown=zero"
        )
    finally:
        if sender is not None:
            sender.close()
        if reconnect_sender is not None:
            reconnect_sender.close()
        if brain_task is not None and not brain_task.done():
            brain_stop_event.set()
            await brain_task
        if cm5_task is not None and not cm5_task.done():
            cm5_stop_event.set()
            await cm5_task


if __name__ == "__main__":
    asyncio.run(run())
