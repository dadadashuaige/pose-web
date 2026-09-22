from pydantic import BaseModel, Field, field_validator


def _as_text(value) -> str:
    """把模型返回的字符串/数组统一收敛成字符串。

    大模型对 "limitations" 这类字段时而无、时而返回数组，
    严格 schema 会让整份结果校验失败并静默退回本地规则。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "；".join(str(item) for item in value if str(item).strip())
    return str(value)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


class PoseDetection(BaseModel):
    person_index: int
    confidence: float = Field(ge=0, le=1)
    keypoints: dict[str, list[float] | None]


class AnalysisReport(BaseModel):
    title: str
    summary: str
    posture: str
    observations: list[str]
    recommendations: list[str]
    limitations: str

    @field_validator("title", "summary", "posture", "limitations", mode="before")
    @classmethod
    def _coerce_text(cls, value):
        return _as_text(value)

    @field_validator("observations", "recommendations", mode="before")
    @classmethod
    def _coerce_list(cls, value):
        return _as_list(value)


class AnalyzeResponse(BaseModel):
    analysis_id: str
    original_url: str
    visualization_url: str
    model: str
    llm_provider: str
    detections: list[PoseDetection]
    report: AnalysisReport
