"""Mac-side UDP sender for the versioned CM5 command stream."""

import socket
from typing import Callable, Optional

from control.command_packet import CommandPacket
from control.velocity import VelocityCommand


class UdpCommandSender:
    """Encode and send sequential velocity packets to the CM5 safety receiver."""

    def __init__(
        self,
        destination_host: str,
        destination_port: int = 5001,
        socket_factory: Optional[Callable[..., object]] = None,
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
        self._socket_factory = socket_factory or socket.socket
        self._socket = None
        self._sequence = 0

    def start(self):
        """Open the sender socket once; repeated starts are harmless."""

        if self._socket is None:
            self._socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: VelocityCommand):
        """Send one packet and advance its sequence only after a successful send."""

        self.start()
        payload = CommandPacket(self._sequence, command).encode()
        self._socket.sendto(payload, self.destination)
        self._sequence += 1

    def close(self):
        """Release the UDP socket."""

        if self._socket is None:
            return
        self._socket.close()
        self._socket = None
