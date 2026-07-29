"""Find people with YOLO and return one detection."""

import math
from numbers import Real
from typing import Any, Optional

from control.tracking import Detection


PERSON_CLASS_ID = 0
ASSOCIATION_MAX_GAP_S = 0.5


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)


def _scalar(value: Any) -> float:
    """Turn a tensor value into a Python number."""

    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _box_coordinates(value: Any):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        coordinates = value[0]
    elif getattr(value, "ndim", 1) > 1:
        coordinates = value[0]
    else:
        coordinates = value
    if hasattr(coordinates, "tolist"):
        coordinates = coordinates.tolist()
    return [float(coordinate) for coordinate in coordinates]


class YoloPersonDetector:
    """Return the largest valid person in each image."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        if not _finite(confidence_threshold) or not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self._last_center = None
        self._last_detection_timestamp_s = None

    def detect(self, frame: Any, timestamp_s: float) -> Optional[Detection]:
        """Return one person detection, or none."""

        if not _finite(timestamp_s):
            raise ValueError("detection timestamp must be finite")
        if (
            self._last_detection_timestamp_s is not None
            and timestamp_s - self._last_detection_timestamp_s > ASSOCIATION_MAX_GAP_S
        ):
            self._last_center = None

        results = self._model(frame, verbose=False)
        candidates = []
        for result in results:
            for box in result.boxes:
                try:
                    class_id = int(_scalar(box.cls[0]))
                    confidence = _scalar(box.conf[0])
                    coordinates = _box_coordinates(box.xyxy)
                except (IndexError, TypeError, ValueError):
                    continue
                if (
                    not _finite(confidence)
                    or not 0.0 <= confidence <= 1.0
                    or len(coordinates) != 4
                    or any(not _finite(coordinate) for coordinate in coordinates)
                ):
                    continue
                x1, y1, x2, y2 = coordinates
                if x2 <= x1 or y2 <= y1:
                    continue
                if class_id != PERSON_CLASS_ID or confidence < self.confidence_threshold:
                    continue
                area = (x2 - x1) * (y2 - y1)
                candidates.append(
                    (area, confidence, coordinates, (x1 + x2) / 2.0, (y1 + y2) / 2.0)
                )

        if not candidates:
            return None
        if self._last_center is None:
            best = max(candidates, key=lambda candidate: candidate[0])
        else:
            best = min(
                candidates,
                key=lambda candidate: (
                    (candidate[3] - self._last_center[0]) ** 2
                    + (candidate[4] - self._last_center[1]) ** 2,
                    -candidate[0],
                ),
            )
        _, confidence, coordinates, center_x, center_y = best
        self._last_center = (center_x, center_y)
        self._last_detection_timestamp_s = timestamp_s
        x1, y1, x2, y2 = coordinates
        return Detection(
            x_px=center_x,
            y_px=(y1 + y2) / 2.0,
            timestamp_s=timestamp_s,
            confidence=confidence,
            width_px=x2 - x1,
            height_px=y2 - y1,
        )
