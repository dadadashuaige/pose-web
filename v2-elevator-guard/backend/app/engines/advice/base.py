"""建议引擎接口。

三种实现对应你要做的对比测试：手写模板 / 大模型 / 大模型+知识库。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..behavior.base import BehaviorResult


@dataclass
class AdviceResult:
    priority: str = "持续"
    title: str = ""
    actions: list[str] = field(default_factory=list)
    disclaimer: str = ""
    engine: str = "template"
    references: list[dict] = field(default_factory=list)
    reasoning: str = ""
    latency_ms: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "priority": self.priority,
            "title": self.title,
            "actions": list(self.actions),
            "disclaimer": self.disclaimer,
            "engine": self.engine,
            "references": list(self.references),
            "reasoning": self.reasoning,
            "latency_ms": int(self.latency_ms),
            "note": self.note,
        }


class AdviceEngine(ABC):
    key: str = "base"
    label: str = "base"

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str:
        return ""

    @abstractmethod
    def generate(
        self,
        behavior: BehaviorResult,
        risk: dict,
        context: dict,
        hits: list | None = None,
    ) -> AdviceResult:
        """依据行为与风险生成处置建议。"""


def priority_of(risk_level: str) -> str:
    if risk_level in {"高风险", "high"}:
        return "立即"
    if risk_level in {"中风险", "medium"}:
        return "尽快"
    return "持续"


COMMON_ACTIONS = [
    "保留原始视频、时间戳和模型证据，供人工复核。",
    "勿仅凭算法结论处分人员；由值守人员确认现场。",
]

DISCLAIMER = "系统输出仅作为安全管理辅助依据，高风险结论须经人工确认后执行处置。"
