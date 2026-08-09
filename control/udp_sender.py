"""Send commands and receive CM5 distance telemetry."""

import math
import socket
import time

from control.command_packet import MAX_PACKET_BYTES, CommandPacket, TelemetryPacket
from control.mind import Telemetry
from control.velocity import VelocityCommand


TELEMETRY_TIMEOUT_S = 0.2


class UdpCommandSender:
    """Send numbered commands to the CM5 safety receiver."""

    def __init__(
        self,
        destination_host: str,
        destination_port: int = 5001,
    ):
        if not destination_host.strip():
            raise ValueError("destination host must not be empty")
        if (
            isinstance(destination_port, bool)
            or not isinstance(destination_port, int)
            or destination_port < 1
            or destination_port > 65535
        ):
            raise ValueError("destination port must be between 1 and 65535")
        self.destination = (destination_host, destination_port)
        self._socket = None
        # A restarted Mac must be newer than packets the CM5 already accepted.
        self._sequence = time.time_ns()
        self._telemetry_sequence = None
        self._obstacle_distance_m = None
        self._forward_velocity_m_s = None
        self._right_velocity_m_s = None
        self._down_velocity_m_s = None
        self._telemetry_received_at_s = None

    def start(self):
        """Open the sender socket once; repeated starts are harmless."""

        if self._socket is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setblocking(False)

    def send(self, command: VelocityCommand):
        """Send one packet and advance its sequence only after a successful send."""

        self.start()
        payload = CommandPacket(self._sequence, command).encode()
        self._socket.sendto(payload, self.destination)
        self._sequence += 1

    def telemetry(self) -> Telemetry:
        """Return fresh CM5 sensor and PX4 velocity telemetry."""

        self._read_telemetry()
        if (
            self._telemetry_received_at_s is None
            or time.monotonic() - self._telemetry_received_at_s > TELEMETRY_TIMEOUT_S
        ):
            return Telemetry(obstacle_distance_m=math.nan)
        return Telemetry(
            obstacle_distance_m=(
                self._obstacle_distance_m
                if self._obstacle_distance_m is not None
                else math.nan
            ),
            forward_velocity_m_s=self._forward_velocity_m_s,
            right_velocity_m_s=self._right_velocity_m_s,
            down_velocity_m_s=self._down_velocity_m_s,
        )

    def _read_telemetry(self):
        if self._socket is None:
            return
        while True:
            try:
                payload, _ = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                packet = TelemetryPacket.decode(payload)
            except ValueError:
                continue
            if (
                self._telemetry_sequence is not None
                and packet.sequence <= self._telemetry_sequence
            ):
                continue
            self._telemetry_sequence = packet.sequence
            self._obstacle_distance_m = packet.obstacle_distance_m
            self._forward_velocity_m_s = packet.forward_velocity_m_s
            self._right_velocity_m_s = packet.right_velocity_m_s
            self._down_velocity_m_s = packet.down_velocity_m_s
            self._telemetry_received_at_s = time.monotonic()

    def close(self):
        """Release the UDP socket."""

        if self._socket is None:
            return
        self._socket.close()
        self._socket = None
