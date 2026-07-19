from __future__ import annotations

import threading
from pathlib import Path

import cv2
from ultralytics import YOLO

from backend.app.schemas import PoseDetection

COCO_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class PoseService:
    """Lazy-loaded YOLO-Pose inference, safe for concurrent requests."""

    def __init__(self, model_name: str, confidence: float) -> None:
        self.model_name = model_name
        self.confidence = confidence
        self._model: YOLO | None = None
        self._lock = threading.Lock()

    def _get_model(self) -> YOLO:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = YOLO(self.model_name)
        return self._model

    def analyze(self, image_path: Path, output_path: Path) -> list[PoseDetection]:
        model = self._get_model()
        with self._lock:
            result = model(str(image_path), conf=self.confidence, verbose=False)[0]

        rendered = result.plot()  # BGR image with boxes, skeletons, and keypoints
        if not cv2.imwrite(str(output_path), rendered):
            raise RuntimeError("无法写入关键点可视化图片")

        if result.keypoints is None or result.boxes is None:
            return []

        points = result.keypoints.data.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        detections: list[PoseDetection] = []
        for person_index, (person_points, confidence) in enumerate(zip(points, confidences)):
            keypoints: dict[str, list[float] | None] = {}
            for name, point in zip(COCO_KEYPOINT_NAMES, person_points):
                x, y, visibility = (float(value) for value in point[:3])
                keypoints[name] = [round(x, 1), round(y, 1), round(visibility, 3)] if visibility > 0 else None
            detections.append(PoseDetection(
                person_index=person_index,
                confidence=round(float(confidence), 3),
                keypoints=keypoints,
            ))
        return detections

