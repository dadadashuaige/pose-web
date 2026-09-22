"""大模型行为判定。

只把「结构化统计量 + 明确的类别清单」交给模型，不让它凭空看画面猜。
没配 key 时自动降级为不判定，并写清原因，流程不中断。
"""

from __future__ import annotations

from ...services.llm_client import extract_json, get_llm_client
from .base import BEHAVIOR_LABELS, BehaviorEngine, BehaviorResult, SequenceContext
from .features import build_features

SYSTEM = (
    "你是电梯轿厢安全监控的行为分析助手。你只能依据给定的姿态统计量做判断，"
    "不得引入画面之外的假设。证据不足时必须选择 normal 或 unanalyzed，并说明原因。"
    "输出必须是 JSON，不要包含解释性文字。"
)

CANDIDATES = ["fall", "fight", "kick_door", "pry_door", "push_door", "lean_door", "normal"]


def build_prompt(context: SequenceContext, features, extra: str = "") -> str:
    taxonomy = "\n".join(
        f"- {key}（{BEHAVIOR_LABELS.get(key, key)}）" for key in CANDIDATES
    )
    tracks = "\n".join(features.summary_lines()) or "未形成稳定轨迹。"
    pair = (
        f"{float(features.nearest_pair.min()):.0f} 像素"
        if len(features.nearest_pair)
        else "不适用"
    )
    return f"""## 输入概况
共采样 {context.sample_count} 帧，时长 {context.duration:.1f} 秒，
原视频 {context.width}x{context.height} @ {context.fps:.1f}fps，
采样间隔为每 {context.frame_stride} 帧取 1 帧。
最大同时在场人数 {context.people_max}，稳定轨迹 {len(features.tracks)} 条，最近两人最小距离 {pair}。
{extra}

## 人员运动统计（距离与速度均以躯干长度归一化）
{tracks}

## 可选行为类别
{taxonomy}

## 判断要求
1. 摔倒（fall）：躯干明显偏离竖直且重心骤降，单人即可成立；
2. 打架（fight）：需要≥2 人靠近且上肢动作剧烈，单人的剧烈动作不足以判定；
3. 踢门（kick_door）/ 扒门（pry_door）/ 推门（push_door）/ 靠门（lean_door）：
   这四类都要求「手脚或重心确实贴近门区域」，判断时必须引用距门数值。
   如果数据里没有门区域信息（距门字段缺失或恒为 0），不要给出这四类结论；
4. 若统计量只能说明"有人动作较大"，应选择 normal 并在 reasoning 里说明证据不足。

## 输出格式
{{"behavior": "{'|'.join(CANDIDATES)}",
  "confidence": 0.0,
  "reasoning": "用中文说明你依据了哪些数值",
  "evidence": ["支撑结论的实测数值"]}}
"""


class LLMEngine(BehaviorEngine):
    key = "llm"
    label = "大模型判定"

    @property
    def available(self) -> bool:
        return get_llm_client().available

    @property
    def unavailable_reason(self) -> str:
        return "未配置 LLM_API_KEY（可在 backend/.env 中填写）"

    def predict(self, context: SequenceContext) -> BehaviorResult:
        features = build_features(context)
        client = get_llm_client()
        if not client.available:
            return BehaviorResult(
                behavior="unanalyzed",
                confidence=0.0,
                engine=self.key,
                note=self.unavailable_reason + "，本次未做行为判定。",
            )

        prompt = build_prompt(context, features)
        try:
            raw, latency = client.chat(SYSTEM, prompt)
            payload = extract_json(raw)
            if not payload:
                raise ValueError("模型未返回可解析的 JSON")
        except Exception as exc:  # noqa: BLE001
            return BehaviorResult(
                behavior="unanalyzed",
                confidence=0.0,
                engine=self.key,
                note=f"大模型调用失败（{type(exc).__name__}: {exc}），本次未做行为判定。",
            )

        behavior = str(payload.get("behavior") or "normal").strip()
        if behavior not in BEHAVIOR_LABELS:
            behavior = "normal"
        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        return BehaviorResult(
            behavior=behavior,
            confidence=max(0.0, min(1.0, confidence)),
            raw=f"llm:{behavior}",
            engine=self.key,
            note=f"由 {client.model} 依据姿态统计量判定，耗时 {latency}ms。",
            reasoning=str(payload.get("reasoning") or ""),
            evidence={
                "模型": client.model,
                "耗时(ms)": latency,
                "依据": payload.get("evidence") or [],
            },
        )
