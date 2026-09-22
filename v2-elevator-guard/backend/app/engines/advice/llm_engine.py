"""大模型建议：依据行为与风险生成处置话术。"""

from __future__ import annotations

from ...services.llm_client import extract_json, get_llm_client
from ..behavior.base import BEHAVIOR_LABELS, BehaviorResult
from .base import AdviceEngine, AdviceResult, priority_of
from .template_engine import TemplateAdviceEngine

SYSTEM = (
    "你是电梯应急处置建议助手。依据已确认的行为与风险等级，给出可立即执行、"
    "分步骤的现场处置建议，明确执行主体。不要复述风险描述，不要编造标准条款。"
    "输出必须是 JSON。"
)


def build_prompt(behavior: BehaviorResult, risk: dict, context: dict) -> str:
    return f"""## 已确认行为
{behavior.label}（{behavior.behavior}），置信度 {behavior.confidence:.2f}
判定依据：{behavior.reasoning or '；'.join(f'{k}={v}' for k, v in behavior.evidence.items())}

## 风险评估
等级 {risk.get('level') or risk.get('risk_level')}，分数 {float(risk.get('score') or 0):.2f}
风险因子：{risk.get('factors')}

## 场景信息
时长 {context.get('duration', 0):.1f} 秒，采样 {context.get('sample_count', 0)} 帧，
最大同时人数 {context.get('people_max', 0)}，分析模型 {context.get('pose_engine', '')}。

## 输出格式
{{"title": "一句话结论",
  "actions": ["按时间顺序排列、含执行主体的处置动作"],
  "reasoning": "简述建议依据"}}
"""


class LLMAdviceEngine(AdviceEngine):
    key = "llm"
    label = "大模型建议"

    @property
    def available(self) -> bool:
        return get_llm_client().available

    @property
    def unavailable_reason(self) -> str:
        return "未配置 LLM_API_KEY（可在 backend/.env 中填写）"

    def generate(self, behavior: BehaviorResult, risk: dict, context: dict, hits=None) -> AdviceResult:
        client = get_llm_client()
        if not client.available:
            result = TemplateAdviceEngine().generate(behavior, risk, context)
            result.engine = self.key
            result.note = self.unavailable_reason + "，本次回退为模板建议。"
            return result

        level = risk.get("level") or risk.get("risk_level") or "低风险"
        try:
            raw, latency = client.chat(SYSTEM, build_prompt(behavior, risk, context))
            payload = extract_json(raw)
            if not payload:
                raise ValueError("模型未返回可解析的 JSON")
        except Exception as exc:  # noqa: BLE001
            result = TemplateAdviceEngine().generate(behavior, risk, context)
            result.engine = self.key
            result.note = f"大模型调用失败（{type(exc).__name__}），本次回退为模板建议。"
            return result

        actions = [str(item) for item in (payload.get("actions") or [])]
        return AdviceResult(
            priority=priority_of(level),
            title=str(payload.get("title") or f"{level}：{BEHAVIOR_LABELS.get(behavior.behavior, behavior.behavior)}"),
            actions=actions,
            reasoning=str(payload.get("reasoning") or ""),
            disclaimer="由大模型生成，仅供参考，须经人工确认。",
            engine=self.key,
            latency_ms=latency,
            note=f"由 {client.model} 生成，耗时 {latency}ms。",
        )
