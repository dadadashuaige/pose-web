"""行为识别引擎注册表。

新增模型：实现 BehaviorEngine 并在 BEHAVIOR_ENGINES 里注册一行。
"""

from __future__ import annotations

from ...config import settings
from .base import (
    BEHAVIOR_LABELS,
    DANGER_PRIOR,
    BehaviorEngine,
    BehaviorResult,
    SequenceContext,
)
from .llm_engine import LLMEngine
from .none_engine import NoneEngine
from .rule_engine import RuleEngine, load_rule_config
from .stgcn_engine import STGCNEngine

BEHAVIOR_ENGINES: dict[str, type[BehaviorEngine]] = {
    "none": NoneEngine,
    "rule": RuleEngine,
    "llm": LLMEngine,
    "stgcn": STGCNEngine,
}

_instances: dict[str, BehaviorEngine] = {}


def get_behavior_engine(key: str | None = None) -> BehaviorEngine:
    name = (key or settings.behavior_engine or "none").lower()
    if name not in BEHAVIOR_ENGINES:
        raise KeyError(f"未知的行为引擎：{name}（可选 {sorted(BEHAVIOR_ENGINES)}）")
    if name not in _instances:
        _instances[name] = BEHAVIOR_ENGINES[name]()
    return _instances[name]


__all__ = [
    "BEHAVIOR_ENGINES",
    "BEHAVIOR_LABELS",
    "DANGER_PRIOR",
    "BehaviorEngine",
    "BehaviorResult",
    "SequenceContext",
    "get_behavior_engine",
    "load_rule_config",
]
