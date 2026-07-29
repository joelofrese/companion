"""Encode velocity commands for the Mac-to-CM5 link."""

import json
import math
from dataclasses import dataclass
from numbers import Real

from control.velocity import VelocityCommand


PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 512


@dataclass(frozen=True)
class CommandPacket:
    """One numbered velocity command."""

    sequence: int
    command: VelocityCommand

    def encode(self) -> bytes:
        values = (
            self.command.north_m_s,
            self.command.east_m_s,
            self.command.down_m_s,
            self.command.yaw_deg,
        )
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
            raise ValueError("command velocity must be finite")
        payload = {
            "version": PROTOCOL_VERSION,
            "sequence": self.sequence,
            "velocity": {
                "north_m_s": self.command.north_m_s,
                "east_m_s": self.command.east_m_s,
                "down_m_s": self.command.down_m_s,
                "yaw_deg": self.command.yaw_deg,
            },
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("command packet is too large")
        return encoded

    @classmethod
    def decode(cls, payload: bytes) -> "CommandPacket":
        if len(payload) > MAX_PACKET_BYTES:
            raise ValueError("command packet is too large")
        try:
            data = json.loads(payload.decode("utf-8"))
            version = data["version"]
            sequence = data["sequence"]
            velocity = data["velocity"]
            values = tuple(velocity[name] for name in ("north_m_s", "east_m_s", "down_m_s", "yaw_deg"))
        except (AttributeError, KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid command packet") from error

        if version != PROTOCOL_VERSION or isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("invalid command packet header")
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
            raise ValueError("invalid command velocity")
        return cls(sequence, VelocityCommand(*values))
