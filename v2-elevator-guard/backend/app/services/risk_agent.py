"""风险评分 Agent。

确定性算术，不是大模型：危险先验 × 行为置信度 + 拥挤修正 + 关键点质量修正。
每一项因子都会写进报告，方便人工核对“这个分数怎么来的”。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engines.behavior.base import BEHAVIOR_LABELS, DANGER_PRIOR


@dataclass
class RiskResult:
    score: float
    level: str
    factors: list = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "score": round(float(self.score), 4),
            "level": self.level,
            "factors": list(self.factors),
            "reason": self.reason,
        }


class RiskAgent:
    """可审计的规则型风险 Agent。"""

    def evaluate(
        self,
        behavior: str,
        confidence: float,
        people_max: int,
        posture_quality: float,
    ) -> RiskResult:
        prior = DANGER_PRIOR.get(behavior, 0.0)
        confidence = max(0.0, min(1.0, float(confidence or 0.0)))
        base = prior * confidence
        crowd = min(0.18, max(0, people_max - 1) * 0.06)
        low_quality = 0.08 if posture_quality < 0.40 else 0.0
        score = min(1.0, base + crowd + low_quality)

        if behavior in {"normal", "unanalyzed"}:
            # 没有行为结论时不制造风险，但也不假装“已验证安全”
            score = min(score, 0.20)

        if score >= 0.70:
            level = "高风险"
        elif score >= 0.40:
            level = "中风险"
        else:
            level = "低风险"

        factors = [
            {"name": "识别行为", "value": BEHAVIOR_LABELS.get(behavior, behavior)},
            {"name": "行为危险先验", "value": round(prior, 3)},
            {"name": "行为模型置信度", "value": round(confidence, 3)},
            {"name": "行为基础风险", "value": round(base, 3)},
            {"name": "最大在场人数", "value": people_max},
            {"name": "拥挤修正", "value": round(crowd, 3)},
            {"name": "关键点质量", "value": round(float(posture_quality or 0.0), 3)},
            {"name": "低质量修正", "value": round(low_quality, 3)},
        ]
        reason = (
            f"{BEHAVIOR_LABELS.get(behavior, behavior)}："
            f"先验 {prior:.2f} × 置信度 {confidence:.2f} = {base:.3f}，"
            f"加拥挤修正 {crowd:.3f}、低质量修正 {low_quality:.3f}，合计 {score:.3f}。"
        )
        return RiskResult(round(score, 3), level, factors, reason)
