"""接口出入参模型。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AnalysisItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    stage: str = ""
    progress: int = 0
    media_type: str = "video"
    created_at: datetime
    completed_at: datetime | None = None

    pose_engine: str | None = None
    behavior_engine: str | None = None
    advice_engine: str | None = None
    door_roi: list[float] = Field(default_factory=list)

    risk_level: str | None = None
    risk_score: float | None = None
    behavior: str | None = None
    behavior_label: str | None = None
    behavior_score: float | None = None
    summary: str | None = None
    advice: dict | None = None
    error: str | None = None
    result: dict | None = None

    overlay_path: str | None = None
    overlay_codec: str | None = None
    report_path: str | None = None

    duration_s: float | None = None
    frames_total: int | None = None
    frames_sampled: int | None = None
    people_max: int | None = None
    posture_quality: float | None = None

    archived: bool = False
    raw_video_deleted: bool = False

    has_report: bool = False
    has_video: bool = False


class AnalysisList(BaseModel):
    items: list[AnalysisItem]
    total: int
    page: int
    page_size: int
    stats: dict = Field(default_factory=dict)


class EngineInfo(BaseModel):
    key: str
    label: str
    note: str = ""
    kind: str = "behavior"
    available: bool = True
    weights: str = ""


class HealthResponse(BaseModel):
    status: str
    app_name: str
    device: dict
    engines: dict[str, list[EngineInfo]]
    defaults: dict[str, str]
    llm_configured: bool
    llm_model: str
    knowledge_entries: int
    knowledge_backend: str
    knowledge_detail: str = ""
    overlay_codecs: list[str]
    door_roi: list[float] = Field(default_factory=list)


class KnowledgeRebuild(BaseModel):
    entries: int
    backend: str
    detail: str = ""


class CleanupResult(BaseModel):
    deleted: int = 0
    detail: str = ""


class AnalysisOptions(BaseModel):
    pose_engine: str | None = None
    behavior_engine: str | None = None
    advice_engine: str | None = None
    compare_advice: bool = False
