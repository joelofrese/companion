"""Save small camera-and-action episodes for future policy learning."""

import json
import math
from pathlib import Path
from typing import Optional

from PIL import Image

from control.mind import Telemetry
from control.velocity import VelocityCommand


SAMPLE_PERIOD_S = 0.2
FRAME_WIDTH = 320


class TrajectoryRecorder:
    """Write one readable JSONL episode with optional JPEG frames."""

    def __init__(
        self,
        path: Path,
        metadata: dict,
        sample_period_s: float = SAMPLE_PERIOD_S,
    ):
        if sample_period_s <= 0.0 or not math.isfinite(sample_period_s):
            raise ValueError("trajectory sample period must be positive")
        self.path = Path(path).expanduser()
        if self.path.exists():
            if not self.path.is_dir() or any(self.path.iterdir()):
                raise ValueError("trajectory path must be a new or empty directory")
        self.path.mkdir(parents=True, exist_ok=True)
        self.frames_path = self.path / "frames"
        self.frames_path.mkdir()
        self._file = (self.path / "trajectory.jsonl").open(
            "w",
            encoding="utf-8",
        )
        self._sample_period_s = sample_period_s
        self._last_sample_s: Optional[float] = None
        self.sample_count = 0
        self._closed = False
        self._write({"type": "episode", "format": 1, **metadata})

    def record(
        self,
        elapsed_s: float,
        frame,
        telemetry: Telemetry,
        brain_command: VelocityCommand,
        forwarded_command: Optional[VelocityCommand],
        model_action: str,
        state: dict,
    ):
        """Save one sample when the recording interval has elapsed."""

        if self._closed:
            return
        if (
            self._last_sample_s is not None
            and elapsed_s - self._last_sample_s < self._sample_period_s
        ):
            return
        self._last_sample_s = elapsed_s
        self.sample_count += 1
        frame_name = None
        if frame is not None:
            frame_name = f"frames/{self.sample_count:06d}.jpg"
            image = Image.fromarray(frame[:, :, ::-1])
            if image.width > FRAME_WIDTH:
                image.thumbnail((FRAME_WIDTH, image.height))
            image.save(self.path / frame_name, format="JPEG", quality=80)
        self._write(
            {
                "type": "sample",
                "elapsed_s": round(elapsed_s, 3),
                "frame": frame_name,
                "telemetry": _telemetry(telemetry),
                "brain_command": _command(brain_command),
                "forwarded_command": _command(forwarded_command),
                "model_action": " ".join(str(model_action).split()),
                "state": state,
            }
        )

    def close(self):
        """Finish the episode and close its JSONL file."""

        if self._closed:
            return
        self._write({"type": "end", "samples": self.sample_count})
        self._file.close()
        self._closed = True

    def _write(self, value: dict):
        self._file.write(json.dumps(value, separators=(",", ":")) + "\n")
        self._file.flush()


def _telemetry(telemetry: Telemetry) -> dict:
    return {
        "obstacle_distance_m": _number(telemetry.obstacle_distance_m),
        "forward_velocity_m_s": _number(telemetry.forward_velocity_m_s),
        "right_velocity_m_s": _number(telemetry.right_velocity_m_s),
        "down_velocity_m_s": _number(telemetry.down_velocity_m_s),
        "heading_rad": _number(telemetry.heading_rad),
    }


def _command(command: Optional[VelocityCommand]) -> Optional[dict]:
    if command is None:
        return None
    return {
        "forward_m_s": _number(command.forward_m_s),
        "right_m_s": _number(command.right_m_s),
        "down_m_s": _number(command.down_m_s),
        "yaw_rate_deg_s": _number(command.yaw_rate_deg_s),
    }


def _number(value):
    return value if isinstance(value, (int, float)) and math.isfinite(value) else None
