from __future__ import annotations

import json
from typing import Literal

from backend.app.schemas import AnalysisReport, PoseDetection


class ActionAnalysisAgent:
    """LLM analysis with an explainable local fallback."""

    def __init__(self, api_key: str | None, base_url: str | None, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def analyze(self, detections: list[PoseDetection]) -> tuple[AnalysisReport, Literal["openai", "local-rules"]]:
        if self.api_key:
            try:
                return self._analyze_with_llm(detections), "openai"
            except Exception:
                # Provider errors should not break the visual-inspection workflow.
                pass
        return self._analyze_locally(detections), "local-rules"

    def _analyze_with_llm(self, detections: list[PoseDetection]) -> AnalysisReport:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        evidence = [item.model_dump() for item in detections]
        prompt = (
            "你是人体姿态分析助手。只根据 YOLO-Pose 的二维关键点证据给出谨慎的中文分析；"
            "不要诊断疾病、不要假装看到了关键点以外的信息。单张图片无法确认连续动作时必须说明不确定性。\n\n"
            f"检测结果：{json.dumps(evidence, ensure_ascii=False)}\n\n"
            "请严格返回 JSON，字段为 title、summary、posture、observations（字符串数组）、"
            "recommendations（字符串数组）、limitations。"
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你输出符合要求的 JSON，不使用 Markdown。"},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        if not raw:
            raise ValueError("LLM 未返回内容")
        return AnalysisReport.model_validate_json(raw)

    def _analyze_locally(self, detections: list[PoseDetection]) -> AnalysisReport:
        if not detections:
            return AnalysisReport(
                title="未检测到清晰人体姿态",
                summary="图片中未发现置信度足够的人体关键点。",
                posture="无法判断",
                observations=["请使用光线充足、全身或上半身清晰可见的图片后重试。"],
                recommendations=["确保人物未被遮挡", "让相机与人物保持适当距离"],
                limitations="本结果基于单张图片的二维姿态检测，不构成医学、运动或安全评估。",
            )
        count = len(detections)
        primary = detections[0]
        visible = sum(value is not None for value in primary.keypoints.values())
        return AnalysisReport(
            title="人体姿态检测完成",
            summary=f"检测到 {count} 人；主要人物可见 {visible}/17 个关键点，检测置信度 {primary.confidence:.0%}。",
            posture="已提取二维骨架，可结合 LLM 对关键点关系进行动作描述。",
            observations=[
                f"主要人物的关键点可见度：{visible}/17。",
                "可在右侧可视化图中核对关键点与肢体连接是否贴合人体。",
            ],
            recommendations=[
                "配置 OPENAI_API_KEY 后可启用基于关键点证据的自然语言动作分析。",
                "若需判断动作过程，请上传连续视频帧或接入视频时序模型。",
            ],
            limitations="单张图片只能估计二维姿态，不能可靠判定动作过程、三维角度或健康状况。",
        )

