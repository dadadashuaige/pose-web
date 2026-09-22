"""行为识别引擎接口与行为分类体系。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..pose.base import PersonPose

# 电梯场景关心的行为类别（对应最初需求里的六种）。key 是对外统一标识，改动要同步前端与报告。
# 其中「踢门 / 扒门 / 推门 / 靠门」四类都与轿门有关，规则里需要门区域（ROI）作为参照。
BEHAVIOR_LABELS: dict[str, str] = {
    "fall": "摔倒",
    "fight": "打架",
    "kick_door": "踢门",
    "pry_door": "扒门",
    "push_door": "推门",
    "lean_door": "靠门",
    "normal": "正常",
    "unanalyzed": "未分析",
}

# 危险行为 -> 风险先验。RiskAgent 用它做基线。
DANGER_PRIOR: dict[str, float] = {
    "fall": 0.75,
    "fight": 0.65,
    "pry_door": 0.70,
    "kick_door": 0.60,
    "push_door": 0.55,
    "lean_door": 0.35,
    "normal": 0.0,
    "unanalyzed": 0.0,
}


@dataclass
class SequenceContext:
    """交给行为引擎的输入：采样帧上的人体姿态。"""

    times: np.ndarray                       # (T,)
    people: list[list[PersonPose]]          # 每个采样帧一组人体
    fps: float
    frame_stride: int
    width: int
    height: int
    sample_count: int
    # 门区域（归一化 x1,y1,x2,y2）。电梯机位固定，标定一次即可。
    # 不含门区域的场景（比如只看摔倒/打架）可以传全零，此时门相关规则不会命中。
    door_roi: tuple = (0.0, 0.0, 0.0, 0.0)

    @property
    def door_rect(self) -> tuple[float, float, float, float]:
        """门区域换算成像素坐标。"""
        x1, y1, x2, y2 = self.door_roi
        return (x1 * self.width, y1 * self.height, x2 * self.width, y2 * self.height)

    @property
    def has_door(self) -> bool:
        x1, y1, x2, y2 = self.door_roi
        return (x2 - x1) > 0 and (y2 - y1) > 0

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0

    @property
    def people_max(self) -> int:
        return max((len(items) for items in self.people), default=0)


@dataclass
class BehaviorResult:
    behavior: str = "unanalyzed"
    confidence: float = 0.0
    raw: str = ""
    evidence: dict = field(default_factory=dict)
    engine: str = "none"
    note: str = ""
    reasoning: str = ""

    @property
    def label(self) -> str:
        return BEHAVIOR_LABELS.get(self.behavior, self.behavior)

    def to_dict(self) -> dict:
        return {
            "behavior": self.behavior,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "raw": self.raw,
            "evidence": {str(k): _plain(v) for k, v in (self.evidence or {}).items()},
            "engine": self.engine,
            "note": self.note,
            "reasoning": self.reasoning,
        }


def _plain(value):
    """numpy 标量不能直接 json 序列化，统一转成 Python 原生类型。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    return value


class BehaviorEngine(ABC):
    key: str = "base"
    label: str = "base"

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str:
        return ""

    @abstractmethod
    def predict(self, context: SequenceContext) -> BehaviorResult:
        """关键点序列 -> 行为结论。"""
