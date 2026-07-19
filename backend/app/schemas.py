from pydantic import BaseModel, Field


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


class AnalyzeResponse(BaseModel):
    analysis_id: str
    original_url: str
    visualization_url: str
    model: str
    llm_provider: str
    detections: list[PoseDetection]
    report: AnalysisReport

