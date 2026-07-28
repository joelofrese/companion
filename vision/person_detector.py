"""YOLOv8n adapter that emits the tracker’s narrow person-observation type."""

from typing import Any, Optional

from control.tracking import Detection


PERSON_CLASS_ID = 0


def _scalar(value: Any) -> float:
    """Convert a tensor scalar or ordinary numeric value without importing NumPy."""

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
    """Return the highest-confidence person center from each image."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5, model=None):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
        if model is None:
            from ultralytics import YOLO

            model = YOLO(model_path)
        self._model = model
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: Any, timestamp_s: float) -> Optional[Detection]:
        """Run YOLO and return one person observation, or None when no person is found."""

        results = self._model(frame, verbose=False)
        best = None
        for result in results:
            for box in result.boxes:
                class_id = int(_scalar(box.cls[0]))
                confidence = _scalar(box.conf[0])
                if class_id != PERSON_CLASS_ID or confidence < self.confidence_threshold:
                    continue
                if best is None or confidence > best[0]:
                    best = (confidence, _box_coordinates(box.xyxy))

        if best is None:
            return None
        confidence, coordinates = best
        x1, y1, x2, y2 = coordinates
        return Detection(
            x_px=(x1 + x2) / 2.0,
            y_px=(y1 + y2) / 2.0,
            timestamp_s=timestamp_s,
            confidence=confidence,
            width_px=x2 - x1,
            height_px=y2 - y1,
        )
