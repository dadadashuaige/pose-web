"""不判定行为：只保留关键点与可视化。

用于“先跑通流程、行为识别以后再接”的场景。它输出 unanalyzed，
RiskAgent 因此给出低风险，但报告里会明确写清“未做行为判定”。
"""

from __future__ import annotations

from .base import BehaviorEngine, BehaviorResult, SequenceContext
from .features import build_features


class NoneEngine(BehaviorEngine):
    key = "none"
    label = "不判定行为"

    def predict(self, context: SequenceContext) -> BehaviorResult:
        features = build_features(context)
        return BehaviorResult(
            behavior="unanalyzed",
            confidence=0.0,
            raw="",
            engine=self.key,
            note="当前未启用行为识别引擎，仅输出关键点、轨迹与可视化，不做危险行为判定。",
            evidence={
                "采样帧数": context.sample_count,
                "最大同时人数": context.people_max,
                "稳定轨迹数": len(features.tracks),
            },
        )
