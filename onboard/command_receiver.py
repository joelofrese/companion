"""Non-blocking CM5 UDP receiver for Mac velocity command packets."""

import socket
import time
from typing import Callable, Optional

from control.command_packet import MAX_PACKET_BYTES
from control.velocity import VelocityCommand
from onboard.safety import OnboardSafetyEnvelope


class UdpSafetyReceiver:
    """Poll Mac packets and expose only commands approved by the CM5 envelope."""

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 5001,
        safety: Optional[OnboardSafetyEnvelope] = None,
        socket_factory: Optional[Callable[..., object]] = None,
    ):
        if port < 0 or port > 65535:
            raise ValueError("UDP port must be between 0 and 65535")
        self.bind_host = bind_host
        self.port = port
        self.safety = safety or OnboardSafetyEnvelope()
        self._socket_factory = socket_factory or socket.socket
        self._socket = None

    def start(self):
        """Bind the receiver; port zero requests an available ephemeral port."""

        if self._socket is not None:
            return
        self._socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self.bind_host, self.port))
        self._socket.setblocking(False)
        self.port = self._socket.getsockname()[1]

    def poll(self, obstacle_distance_m: Optional[float] = None) -> VelocityCommand:
        """Drain available packets and return the safe command for this local tick."""

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
        """Release the UDP socket."""

        if self._socket is None:
            return
        self._socket.close()
        self._socket = None
