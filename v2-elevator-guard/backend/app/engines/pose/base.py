"""姿态引擎接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class PersonPose:
    """一个人的检测结果，坐标是原图像素。"""

    box: np.ndarray        # (4,) xyxy
    confidence: float
    keypoints: np.ndarray  # (17, 3) -> x, y, confidence

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


class PoseEngine(ABC):
    key: str = "base"
    label: str = "base"

    @property
    @abstractmethod
    def available(self) -> bool:
        """权重是否就绪。不可用时前端会禁用该选项。"""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[PersonPose]:
        """对单帧做人体检测与关键点提取。"""


def device_str() -> str:
    import torch

    from ...config import settings

    if settings.device.startswith("cuda") and torch.cuda.is_available():
        return settings.device
    return "cpu"
