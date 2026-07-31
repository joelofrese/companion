"""Decode camera frames from a running Gazebo world."""

import base64

import numpy as np

from sim.gazebo_topic import GazeboTopicReader


def _decode_frame(message):
    frame = base64.b64decode(message["data"])
    width = int(message["width"])
    height = int(message["height"])
    step = int(message["step"])
    image = np.frombuffer(frame, dtype=np.uint8).reshape(height, step)
    image = image[:, : width * 3]
    return image.reshape(height, width, 3)[:, :, ::-1]


class GazeboCamera(GazeboTopicReader):
    """Keep the newest Gazebo camera frame available to the brain."""

    def __init__(self, topic: str):
        super().__init__(topic, _decode_frame, "camera")
