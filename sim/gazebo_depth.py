"""Decode a Gazebo depth image as one forward distance."""

import base64
import math

import numpy as np

from sim.gazebo_topic import GazeboTopicReader
from sim.offboard_control import DistanceMessage


MIN_DEPTH_M = 0.2
MAX_DEPTH_M = 19.1


class GazeboDepthRangefinder(GazeboTopicReader):
    """Keep the nearest valid distance in the center of a depth image."""

    def __init__(self, topic: str):
        super().__init__(topic, _decode_depth, "depth")


def _decode_depth(message):
    width = int(message["width"])
    height = int(message["height"])
    step = int(message["step"])
    if step % 4:
        raise ValueError("depth step is not a multiple of four")
    values = np.frombuffer(
        base64.b64decode(message["data"]),
        dtype="<f4",
        count=(step // 4) * height,
    ).reshape(height, step // 4)[:, :width]
    center = values[
        height // 4 : 3 * height // 4,
        width // 4 : 3 * width // 4,
    ]
    valid = center[np.isfinite(center)]
    valid = valid[(valid >= MIN_DEPTH_M) & (valid <= MAX_DEPTH_M)]
    distance = float(np.min(valid)) if valid.size else math.nan
    return DistanceMessage(distance, MIN_DEPTH_M, MAX_DEPTH_M)
