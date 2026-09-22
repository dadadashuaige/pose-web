"""YOLO11n-Pose：Ultralytics 通用权重，零配置可用。"""

from __future__ import annotations

import numpy as np

from ...config import settings
from .base import PersonPose, PoseEngine


class Yolo11nPoseEngine(PoseEngine):
    key = "yolo11n"
    label = "YOLO11n-Pose"

    def __init__(self) -> None:
        self._model = None

    @property
    def available(self) -> bool:
        # 权重缺失时 ultralytics 会自动下载，所以这里始终认为可用；
        # 真正加载失败会在推理时报错，并由 pipeline 兜住。
        return True

    @property
    def unavailable_reason(self) -> str:
        if not settings.yolo11n_weights.exists():
            return "权重文件当前不存在，首次推理时会自动下载（需要联网）"
        return ""

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(str(settings.yolo11n_weights))

    def detect(self, frame: np.ndarray) -> list[PersonPose]:
        self._load()
        import torch

        device = (
            0 if settings.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        result = self._model.predict(
            frame,
            device=device,
            verbose=False,
            conf=settings.pose_confidence,
            max_det=settings.max_people,
        )[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []

        xy = result.keypoints.xy.cpu().numpy()
        kconf = result.keypoints.conf
        kconf = (
            kconf.cpu().numpy()
            if kconf is not None
            else np.ones(xy.shape[:2], dtype=np.float32)
        )
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        people: list[PersonPose] = []
        for index in range(len(xy)):
            kpts = np.concatenate(
                [xy[index], kconf[index][:, None]], axis=1
            ).astype(np.float32)
            people.append(
                PersonPose(
                    box=boxes[index].astype(np.float32),
                    confidence=float(scores[index]),
                    keypoints=kpts,
                )
            )
        return people
