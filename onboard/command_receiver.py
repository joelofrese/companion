"""Receive Mac commands and apply CM5 safety checks."""

import math
import socket
import time
from numbers import Real
from typing import Optional

from control.command_packet import MAX_PACKET_BYTES, TelemetryPacket
from control.velocity import VelocityCommand
from onboard.safety import OnboardSafetyEnvelope


class UdpSafetyReceiver:
    """Read Mac packets and return only safe commands."""

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 5001,
        safety: Optional[OnboardSafetyEnvelope] = None,
    ):
        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or port < 0
            or port > 65535
        ):
            raise ValueError("UDP port must be between 0 and 65535")
        self.bind_host = bind_host
        self.port = port
        self.safety = safety or OnboardSafetyEnvelope()
        self._socket = None
        self._client_address = None
        self._telemetry_sequence = 0

    def start(self):
        """Bind the UDP socket."""

        if self._socket is not None:
            return
        receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            receiver_socket.bind((self.bind_host, self.port))
            receiver_socket.setblocking(False)
        except Exception:
            receiver_socket.close()
            raise
        self._socket = receiver_socket
        self.port = receiver_socket.getsockname()[1]
        self._client_address = None
        self._telemetry_sequence = 0

    def poll(self, obstacle_distance_m: Optional[float] = None) -> VelocityCommand:
        """Read available packets and return the safe command."""

        if self._socket is None:
            raise RuntimeError("receiver must be started before polling")
        while True:
            try:
                payload, address = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                accepted = self.safety.receive_packet(time.monotonic(), payload)
            except ValueError:
                continue
            if accepted:
                self._client_address = address
        return self.safety.tick(time.monotonic(), obstacle_distance_m=obstacle_distance_m)

    def send_telemetry(self, obstacle_distance_m: Optional[float] = None):
        """Return the latest sensor value to the Mac command sender."""

        if self._socket is None or self._client_address is None:
            return
        if (
            obstacle_distance_m is not None
            and (
                isinstance(obstacle_distance_m, bool)
                or not isinstance(obstacle_distance_m, Real)
                or not math.isfinite(obstacle_distance_m)
                or obstacle_distance_m < 0.0
            )
        ):
            obstacle_distance_m = None
        payload = TelemetryPacket(
            self._telemetry_sequence,
            obstacle_distance_m,
        ).encode()
        self._socket.sendto(payload, self._client_address)
        self._telemetry_sequence += 1

    def close(self):
        """Close the UDP socket."""

        if self._socket is None:
            return
        self._socket.close()
        self._socket = None
        self._client_address = None
