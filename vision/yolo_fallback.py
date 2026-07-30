"""Temporary person-based visual fallback for the Mac brain."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Optional

from control.mind import Telemetry, VisualObservation


PERSON_CLASS_ID = 0
LATERAL_DEAD_ZONE = 0.05


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _box_coordinates(value: Any) -> list[float]:
    coordinates = value[0] if getattr(value, "ndim", 1) > 1 else value
    if hasattr(coordinates, "tolist"):
        coordinates = coordinates.tolist()
    return [float(coordinate) for coordinate in coordinates]


@dataclass(frozen=True)
class Person:
    """The position of one detected person in an image."""

    center_x_px: float
    height_px: float
    confidence: float


class YoloPersonDetector:
    """Return the largest valid person in one image."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        if not _finite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: Any) -> Optional[Person]:
        """Return one person detection, or none."""

        candidates = []
        for result in self._model(frame, verbose=False):
            for box in result.boxes:
                try:
                    class_id = int(_scalar(box.cls[0]))
                    confidence = _scalar(box.conf[0])
                    coordinates = _box_coordinates(box.xyxy)
                except (IndexError, TypeError, ValueError):
                    continue
                if (
                    class_id != PERSON_CLASS_ID
                    or not _finite(confidence)
                    or confidence < self.confidence_threshold
                    or len(coordinates) != 4
                ):
                    continue
                x1, y1, x2, y2 = coordinates
                if (
                    any(not _finite(coordinate) for coordinate in coordinates)
                    or x2 <= x1
                    or y2 <= y1
                ):
                    continue
                candidates.append(
                    (
                        (x2 - x1) * (y2 - y1),
                        Person((x1 + x2) / 2.0, y2 - y1, confidence),
                    )
                )
        return max(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


class YoloVisualModel:
    """Adapt one-frame person detection to the Mac subconscious interface."""

    def __init__(self, model_path: str, frame_width_px: float, target_height_px: float):
        if not _finite(frame_width_px) or frame_width_px <= 0.0:
            raise ValueError("frame width must be positive")
        if not _finite(target_height_px) or target_height_px <= 0.0:
            raise ValueError("target height must be positive")
        self._detector = YoloPersonDetector(model_path=model_path)
        self._frame_width_px = frame_width_px
        self._target_height_px = target_height_px

    def observe(
        self,
        image: Any,
        timestamp_s: float,
        focus: str,
        intent: str,
        previous_movement: str,
        previous_observation: str,
        telemetry: Telemetry,
    ) -> VisualObservation:
        person = self._detector.detect(image) if image is not None else None
        if intent != "following" or person is None:
            description = "no person is available to follow"
            movement = "stop"
        else:
            horizontal_error = (
                person.center_x_px - self._frame_width_px / 2.0
            ) / (self._frame_width_px / 2.0)
            if person.height_px < self._target_height_px:
                description = "the person is ahead"
                movement = "forward"
            elif horizontal_error > LATERAL_DEAD_ZONE:
                description = "the person is to the right"
                movement = "right"
            elif horizontal_error < -LATERAL_DEAD_ZONE:
                description = "the person is to the left"
                movement = "left"
            else:
                description = "the person is centered"
                movement = "stop"
        return VisualObservation(
            timestamp_s=timestamp_s,
            description=description,
            focused_answer=description if focus else "",
            movement=movement,
            next_focus=focus or "person",
            confidence=person.confidence if person is not None else 0.0,
        )
