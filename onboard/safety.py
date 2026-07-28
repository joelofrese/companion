"""CM5-side safety envelope for commands received from the Mac."""

import math
from numbers import Real
from typing import Optional

from control.command_packet import CommandPacket
from control.state_machine import BACKOFF_SPEED_M_S, OBSTACLE_STOP_M
from control.velocity import VelocityCommand


MAX_HORIZONTAL_SPEED_M_S = 0.5
MAX_VERTICAL_SPEED_M_S = 0.3


class OnboardSafetyEnvelope:
    """Expire stale Mac commands and apply the local forward obstacle override."""

    def __init__(self, command_timeout_s: float = 0.15):
        if command_timeout_s <= 0.0:
            raise ValueError("command timeout must be positive")
        self.command_timeout_s = command_timeout_s
        self._command: Optional[VelocityCommand] = None
        self._received_at_s: Optional[float] = None
        self._last_tick_s: Optional[float] = None
        self._last_sequence: Optional[int] = None

    def receive(self, timestamp_s: float, command: VelocityCommand):
        """Record one command using the CM5 receive timestamp."""

        if self._received_at_s is not None and timestamp_s <= self._received_at_s:
            raise ValueError("received command timestamps must increase")
        self._command = command
        self._received_at_s = timestamp_s

    def receive_packet(self, timestamp_s: float, payload: bytes) -> bool:
        """Accept a newer wire packet and ignore duplicates or reordered packets."""

        packet = CommandPacket.decode(payload)
        if self._last_sequence is not None and packet.sequence <= self._last_sequence:
            return False
        self.receive(timestamp_s, packet.command)
        self._last_sequence = packet.sequence
        return True

    def tick(
        self,
        timestamp_s: float,
        obstacle_distance_m: Optional[float] = None,
    ) -> VelocityCommand:
        """Return the command safe to forward to PX4 at this local tick."""

        if self._last_tick_s is not None and timestamp_s <= self._last_tick_s:
            raise ValueError("safety tick timestamps must increase")
        self._last_tick_s = timestamp_s

        if self._command is None or self._received_at_s is None:
            return VelocityCommand()
        if timestamp_s - self._received_at_s > self.command_timeout_s:
            return VelocityCommand()
        if obstacle_distance_m is not None:
            try:
                if not math.isfinite(obstacle_distance_m):
                    return VelocityCommand()
            except TypeError:
                return VelocityCommand()
            if obstacle_distance_m < OBSTACLE_STOP_M:
                return VelocityCommand(north_m_s=-BACKOFF_SPEED_M_S)
        if not self._command_is_safe(self._command):
            return VelocityCommand()
        return self._command

    @staticmethod
    def _command_is_safe(command: VelocityCommand) -> bool:
        values = (
            command.north_m_s,
            command.east_m_s,
            command.down_m_s,
            command.yaw_deg,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
            return False
        return (
            abs(command.north_m_s) <= MAX_HORIZONTAL_SPEED_M_S
            and abs(command.east_m_s) <= MAX_HORIZONTAL_SPEED_M_S
            and abs(command.down_m_s) <= MAX_VERTICAL_SPEED_M_S
        )
