"""大模型 + 知识库建议：先检索电梯安全语料，再基于条目生成建议并标注引用。"""

from __future__ import annotations

from ...services.knowledge import get_knowledge_base
from ...services.llm_client import extract_json, get_llm_client
from ..behavior.base import BEHAVIOR_LABELS, BehaviorResult
from .base import AdviceEngine, AdviceResult, priority_of
from .template_engine import TemplateAdviceEngine

SYSTEM = (
    "你是电梯应急处置建议助手。你只能依据给定的知识库条目提出处置建议，"
    "不得引用知识库以外的规定，不得编造条款编号。"
    "每条建议都要标注它依据的条目 id。若给定条目不足以支撑建议，"
    "必须把 grounded 设为 false 并说明缺少什么信息。输出必须是 JSON。"
)


def build_query(behavior: BehaviorResult) -> str:
    label = BEHAVIOR_LABELS.get(behavior.behavior, behavior.behavior)
    clues = " ".join(str(value) for value in (behavior.evidence or {}).values())
    return f"电梯轿厢 {label} 处置 {behavior.behavior} {clues}"


def build_prompt(behavior: BehaviorResult, risk: dict, hits: list) -> str:
    entries = "\n".join(
        f"[{hit.entry_id}] {hit.title}（来源：{hit.source}，声明风险：{hit.risk_level}）\n{hit.text}"
        for hit in hits
    ) or "（知识库没有返回任何条目）"
    return f"""## 已确认行为
{behavior.label}（{behavior.behavior}），置信度 {behavior.confidence:.2f}

## 风险评估
等级 {risk.get('level') or risk.get('risk_level')}，分数 {float(risk.get('score') or 0):.2f}

## 知识库条目（唯一可用依据）
{entries}

## 输出格式
{{"title": "一句话结论",
  "actions": ["处置动作，须在 source 里标注依据的条目 id"],
  "grounded": true,
  "citations": ["entry_id"],
  "reasoning": "说明引用了哪条、为什么适用",
  "immediate": ["必须立即执行的动作"]}}
"""


class RagAdviceEngine(AdviceEngine):
    key = "rag"
    label = "大模型 + 知识库建议"

    @property
    def available(self) -> bool:
        return get_knowledge_base().count() > 0

    @property
    def unavailable_reason(self) -> str:
        return "知识库为空（check backend/knowledge 目录）"

    def generate(self, behavior: BehaviorResult, risk: dict, context: dict, hits=None) -> AdviceResult:
        kb = get_knowledge_base()
        hits = kb.search(build_query(behavior)) if kb.count() else []
        references = [
            {
                "entry_id": hit.entry_id,
                "title": hit.title,
                "source": hit.source,
                "snippet": hit.snippet(200),
                "score": hit.score,
                "risk_level": hit.risk_level,
            }
            for hit in hits
        ]

        client = get_llm_client()
        if not client.available:
            result = TemplateAdviceEngine().generate(behavior, risk, context)
            result.engine = self.key
            result.references = references
            result.note = (
                "未配置 LLM_API_KEY，本次仅返回检索到的知识库条目 + 模板建议。"
            )
            return result

        level = risk.get("level") or risk.get("risk_level") or "低风险"
        try:
            raw, latency = client.chat(SYSTEM, build_prompt(behavior, risk, hits))
            payload = extract_json(raw)
            if not payload:
                raise ValueError("模型未返回可解析的 JSON")
        except Exception as exc:  # noqa: BLE001
            result = TemplateAdviceEngine().generate(behavior, risk, context)
            result.engine = self.key
            result.references = references
            result.note = f"大模型调用失败（{type(exc).__name__}），已回退为模板建议 + 检索条目。"
            return result

        known = {hit.entry_id for hit in hits}
        citations = [str(item) for item in (payload.get("citations") or [])]
        valid = [item for item in citations if item in known]
        grounded = bool(payload.get("grounded", True)) and bool(valid)
        actions = [str(item) for item in (payload.get("actions") or [])]
        immediate = [str(item) for item in (payload.get("immediate") or [])]

        note = f"由 {client.model} 依据知识库生成，耗时 {latency}ms。"
        if not grounded:
            note += " 知识库证据不足，本建议不构成风险认定，需人工复核。"

        return AdviceResult(
            priority=priority_of(level),
            title=str(
                payload.get("title")
                or f"{level}：{BEHAVIOR_LABELS.get(behavior.behavior, behavior.behavior)}"
            ),
            actions=immediate + actions,
            reasoning=str(payload.get("reasoning") or ""),
            disclaimer="建议基于通用电梯安全语料生成，正式引用前请核对现行规程。",
            engine=self.key,
            references=references,
            latency_ms=latency,
            note=note,
        )
