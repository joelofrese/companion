"""Receive Mac commands and apply CM5 safety checks."""

import socket
import time
from typing import Optional

from control.command_packet import MAX_PACKET_BYTES
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

    def poll(self, obstacle_distance_m: Optional[float] = None) -> VelocityCommand:
        """Read available packets and return the safe command."""

        if self._socket is None:
            raise RuntimeError("receiver must be started before polling")
        while True:
            try:
                payload, _ = self._socket.recvfrom(MAX_PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                self.safety.receive_packet(time.monotonic(), payload)
            except ValueError:
                continue
        return self.safety.tick(time.monotonic(), obstacle_distance_m=obstacle_distance_m)

    def close(self):
        """Close the UDP socket."""

        if self._socket is None:
            return
        self._socket.close()
        self._socket = None
