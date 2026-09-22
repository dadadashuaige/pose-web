"""姿态估计引擎注册表。

新增模型：写一个继承 PoseEngine 的类，在 POSE_ENGINES 里加一行即可。
"""

from __future__ import annotations

from ...config import settings
from .base import PersonPose, PoseEngine
from .mec_pose import MecPoseEngine
from .yolo11n_pose import Yolo11nPoseEngine

POSE_ENGINES: dict[str, type[PoseEngine]] = {
    "mec": MecPoseEngine,
    "yolo11n": Yolo11nPoseEngine,
}

_instances: dict[str, PoseEngine] = {}


def get_pose_engine(key: str | None = None) -> PoseEngine:
    name = (key or settings.pose_engine or "mec").lower()
    if name not in POSE_ENGINES:
        raise KeyError(f"未知的姿态引擎：{name}（可选 {sorted(POSE_ENGINES)}）")
    if name not in _instances:
        _instances[name] = POSE_ENGINES[name]()
    return _instances[name]


__all__ = ["POSE_ENGINES", "get_pose_engine", "PersonPose", "PoseEngine"]
