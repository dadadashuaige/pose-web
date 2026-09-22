"""集中配置。

优先级：真实环境变量 > backend/.env 文件 > 代码里的默认值。

设计取舍：
  * 默认姿态引擎选 `yolo11n`，因为它的权重可以自动下载，clone 下来零配置就能跑。
    自训练的 MEC-Pose 需要你自己提供权重与源码路径，通过 .env 指定。
  * 门区域（DOOR_ROI）默认留空。此时「踢门/推门/扒门/靠门」四条规则不会参与判定，
    结果里会明确写出原因。这样不会因为一个瞎猜的默认区域而产生满屏误报。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    """把 .env 读进 os.environ；已存在的环境变量优先，不被覆盖。"""
    if not path.exists():
        return
    try:
        content = path.read_text(encoding="utf-8-sig")
    except OSError:
        return
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(BACKEND_ROOT / ".env")


def _path(key: str, default: str) -> Path:
    value = os.getenv(key)
    return Path(value).expanduser() if value else Path(default)


def _opt_path(key: str) -> Path | None:
    """可选路径：未配置时返回 None，而不是 '.'——Path("") 会等于当前目录，
    导致"文件存在"的判断永远为真。"""
    value = (os.getenv(key) or "").strip()
    return Path(value).expanduser() if value else None


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _roi(raw: str) -> tuple[float, float, float, float]:
    """解析归一化门区域；非法或未配置时返回全零（表示"未标定"）。"""
    try:
        values = [float(v) for v in raw.split(",")]
        if len(values) != 4:
            return (0.0, 0.0, 0.0, 0.0)
        x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in values)
        if x2 <= x1 or y2 <= y1:
            return (0.0, 0.0, 0.0, 0.0)
        return (x1, y1, x2, y2)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class EngineSpec:
    """前端下拉框用的一份引擎说明。可用性由引擎自身判断，这里只放展示信息。"""

    key: str
    label: str
    note: str = ""
    weights: Path | None = None
    repo: Path | None = None
    kind: str = "behavior"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "note": self.note,
            "kind": self.kind,
            "weights": str(self.weights) if self.weights else "",
        }


@dataclass(frozen=True)
class Settings:
    app_name: str = "电梯危险行为智能预警系统"

    # ---- 目录 ----------------------------------------------------------
    # 相对路径按启动时的工作目录解析（run.ps1 会先切到 backend/）。
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")).resolve()
    )
    knowledge_dir: Path = field(
        default_factory=lambda: _path("KNOWLEDGE_DIR", str(BACKEND_ROOT / "knowledge"))
    )
    chroma_dir: Path = field(default_factory=lambda: _path("CHROMA_DIR", "./data/chroma"))

    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///./data/elevator_risk.db"
        )
    )

    # ---- 推理设备 ------------------------------------------------------
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cuda"))
    frame_stride: int = field(default_factory=lambda: _int("FRAME_STRIDE", 3))
    max_sample_frames: int = field(default_factory=lambda: _int("MAX_SAMPLE_FRAMES", 600))
    pose_confidence: float = field(default_factory=lambda: _float("POSE_CONFIDENCE", 0.35))
    max_people: int = field(default_factory=lambda: _int("MAX_PEOPLE", 8))

    # ---- 引擎选择 ------------------------------------------------------
    pose_engine: str = field(default_factory=lambda: os.getenv("POSE_ENGINE", "yolo11n"))
    behavior_engine: str = field(default_factory=lambda: os.getenv("BEHAVIOR_ENGINE", "none"))
    advice_engine: str = field(default_factory=lambda: os.getenv("ADVICE_ENGINE", "template"))

    # ---- 门区域 --------------------------------------------------------
    # 归一化 (x1,y1,x2,y2)。留空表示未标定，门相关规则不参与判定。
    door_roi: tuple = field(
        default_factory=lambda: _roi(os.getenv("DOOR_ROI", ""))
    )

    # ---- 姿态模型路径 --------------------------------------------------
    # 自训练 MEC-Pose（可选，需自行提供权重与源码目录）
    mec_weights: Path | None = field(default_factory=lambda: _opt_path("MEC_WEIGHTS"))
    mec_repo: Path | None = field(default_factory=lambda: _opt_path("MEC_REPO"))
    # 通用 YOLO11n-Pose：只给文件名时 ultralytics 会自动下载到工作目录
    yolo11n_weights: Path = field(
        default_factory=lambda: _path("YOLO11N_WEIGHTS", "yolo11n-pose.pt")
    )

    # ---- 行为模型路径（PCG-GCN，保留但不默认启用）----------------------
    stgcn_weights: Path | None = field(default_factory=lambda: _opt_path("STGCN_WEIGHTS"))
    stgcn_repo: Path | None = field(default_factory=lambda: _opt_path("STGCN_REPO"))

    # ---- 可视化视频 ----------------------------------------------------
    overlay_enabled: bool = field(
        default_factory=lambda: os.getenv("OVERLAY_ENABLED", "1") not in {"0", "false", "False"}
    )
    overlay_max_width: int = field(default_factory=lambda: _int("OVERLAY_MAX_WIDTH", 1280))
    # 可视化视频的输出帧率。0 = 跟随源视频帧率（最流畅，但编码帧数最多）。
    # 编码是整条链路里最慢的一环：1080p 源、14 秒、每 3 帧取样，
    # 30fps 输出要编码 420 帧约 20 秒；改成 10fps 只要 140 帧约 7 秒，时长不变。
    overlay_fps: int = field(default_factory=lambda: _int("OVERLAY_FPS", 0))

    # ---- 大模型 --------------------------------------------------------
    llm_api_key: str | None = field(
        default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL",
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    )
    llm_timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 120.0))
    llm_temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.2))

    # ---- 知识库检索 ----------------------------------------------------
    embedding_api_key: str | None = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    )
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    rag_top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 5))

    # ---- 数据生命周期 --------------------------------------------------
    # 0 表示不自动清理。原始视频最占空间，报告与关键帧序列保留。
    raw_video_retain_days: int = field(
        default_factory=lambda: _int("RAW_VIDEO_RETAIN_DAYS", 0)
    )

    # ---- 引擎清单（前端下拉框用；可用性以引擎自身判断为准）-------------
    @property
    def pose_engines(self) -> dict[str, EngineSpec]:
        return {
            "yolo11n": EngineSpec(
                key="yolo11n",
                label="YOLO11n-Pose（通用 COCO）",
                note="零配置可用，首次使用会自动下载权重（需联网）",
                weights=self.yolo11n_weights,
                kind="pose",
            ),
            "mec": EngineSpec(
                key="mec",
                label="MEC-Pose（自训练 YOLOv5-Pose）",
                note="需在 .env 中配置 MEC_WEIGHTS 与 MEC_REPO",
                weights=self.mec_weights,
                repo=self.mec_repo,
                kind="pose",
            ),
        }

    @property
    def behavior_engines(self) -> dict[str, EngineSpec]:
        return {
            "none": EngineSpec(
                key="none",
                label="不判定行为（仅输出关键点）",
                note="跳过行为识别，只留姿态与可视化",
                kind="behavior",
            ),
            "rule": EngineSpec(
                key="rule",
                label="规则引擎（电梯六种行为）",
                note="确定性阈值规则，可解释、零成本",
                kind="behavior",
            ),
            "llm": EngineSpec(
                key="llm",
                label="大模型判定",
                note="把关键点统计量交给 LLM 判断，需要 LLM_API_KEY",
                kind="behavior",
            ),
            "stgcn": EngineSpec(
                key="stgcn",
                label="PCG-GCN（NTU-60 通用权重）",
                note="未用电梯数据训练，类别无法对齐，保留待后续微调",
                weights=self.stgcn_weights,
                repo=self.stgcn_repo,
                kind="behavior",
            ),
        }

    @property
    def advice_engines(self) -> dict[str, EngineSpec]:
        return {
            "template": EngineSpec(
                key="template",
                label="手写模板建议",
                note="确定性话术，零成本、零延迟，作为基线对照",
                kind="advice",
            ),
            "llm": EngineSpec(
                key="llm",
                label="大模型建议",
                note="由 LLM 依据行为与风险生成处置建议",
                kind="advice",
            ),
            "rag": EngineSpec(
                key="rag",
                label="大模型 + 知识库建议",
                note="先检索电梯安全知识库，再让 LLM 基于条目生成建议",
                kind="advice",
            ),
        }

    def defaults(self) -> dict:
        return {
            "pose": self.pose_engine,
            "behavior": self.behavior_engine,
            "advice": self.advice_engine,
        }

    def ensure_directories(self) -> None:
        for folder in (
            self.data_dir,
            self.data_dir / "uploads",
            self.data_dir / "reports",
            self.data_dir / "videos",
            self.chroma_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)


settings = Settings()
