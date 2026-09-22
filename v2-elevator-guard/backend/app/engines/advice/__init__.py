"""建议引擎注册表。

新增实现：继承 AdviceEngine 并在这里注册一行。
"""

from __future__ import annotations

from ...config import settings
from .base import AdviceEngine, AdviceResult
from .llm_engine import LLMAdviceEngine
from .rag_engine import RagAdviceEngine
from .template_engine import TemplateAdviceEngine

ADVICE_ENGINES: dict[str, type[AdviceEngine]] = {
    "template": TemplateAdviceEngine,
    "llm": LLMAdviceEngine,
    "rag": RagAdviceEngine,
}

_instances: dict[str, AdviceEngine] = {}


def get_advice_engine(key: str | None = None) -> AdviceEngine:
    name = (key or settings.advice_engine or "template").lower()
    if name not in ADVICE_ENGINES:
        raise KeyError(f"未知的建议引擎：{name}（可选 {sorted(ADVICE_ENGINES)}）")
    if name not in _instances:
        _instances[name] = ADVICE_ENGINES[name]()
    return _instances[name]


__all__ = ["ADVICE_ENGINES", "AdviceEngine", "AdviceResult", "get_advice_engine"]
