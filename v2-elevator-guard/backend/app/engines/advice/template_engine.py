"""手写模板建议：零成本、零延迟、结果稳定，作为对照基线。"""

from __future__ import annotations

from ..behavior.base import BEHAVIOR_LABELS, BehaviorResult
from .base import COMMON_ACTIONS, DISCLAIMER, AdviceEngine, AdviceResult, priority_of

BY_BEHAVIOR: dict[str, list[str]] = {
    "fall": [
        "通过对讲确认乘客意识与伤情，必要时拨打急救电话。",
        "确认轿厢是否平层，不要强行开启轿门，等待维保单位协助。",
    ],
    "fight": [
        "通过对讲制止冲突，通知安保到就近楼层等候，避免单人强行介入。",
        "保留完整时段录像并按现场安全流程升级处理。",
    ],
    "kick_door": [
        "广播制止踢踹行为，要求乘客远离轿门。",
        "事后通知维保单位检查门锁、门机与光幕状态。",
    ],
    "pry_door": [
        "立即通过对讲劝阻扒门行为，要求乘客松手并远离轿门。",
        "通知物业与维保单位；如轿厢停在层间，直接启动困人救援流程。",
    ],
    "push_door": [
        "广播劝阻持续推压轿门的行为，避免门机堵转。",
        "若伴随轿厢运行，应按高风险处置并暂停该梯使用。",
    ],
    "lean_door": [
        "广播提醒乘客与轿门保持安全距离。",
        "事后检查门板是否变形、门锁触点是否偏移。",
    ],
}


class TemplateAdviceEngine(AdviceEngine):
    key = "template"
    label = "手写模板建议"

    def generate(self, behavior: BehaviorResult, risk: dict, context: dict, hits=None) -> AdviceResult:
        level = risk.get("level") or risk.get("risk_level") or "低风险"
        behavior_key = behavior.behavior
        label = BEHAVIOR_LABELS.get(behavior_key, behavior_key)

        if behavior_key in {"normal", "unanalyzed"}:
            actions = ["继续观察监控画面，无需立即处置。"]
            title = f"{level}：{behavior.label}"
        else:
            actions = BY_BEHAVIOR.get(behavior_key, ["由值守人员复核视频与现场状况。"])
            title = f"{level}：{label}"

        if level == "高风险":
            actions = actions + ["启动现场应急处置流程，同步上报安全管理部门。"]

        return AdviceResult(
            priority=priority_of(level),
            title=title,
            actions=actions + list(COMMON_ACTIONS),
            disclaimer=f"风险分数 {float(risk.get('score') or 0):.0%} 是辅助预警，不替代人工安全判断。",
            engine=self.key,
            note="由确定性模板生成，结果可复现。",
        )
