"""数据模型与轻量迁移。

沿用原有的 analyses 表，只做加法：新增列用 ALTER TABLE 补，
不动已有数据，所以你之前那几条记录会原样保留。
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

_IS_SQLITE = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False, "timeout": 30} if _IS_SQLITE else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


if _IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        try:
            # WAL 让后台分析线程写入的同时，页面仍能读取。
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(24), default="queued")
    stage: Mapped[str] = mapped_column(String(40), default="排队中")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    media_type: Mapped[str] = mapped_column(String(16), default="video")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # 三个引擎各是谁，报告里要写清楚，方便复现
    pose_engine: Mapped[str | None] = mapped_column(String(32), nullable=True)
    behavior_engine: Mapped[str | None] = mapped_column(String(32), nullable=True)
    advice_engine: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # 归一化门区域 (x1,y1,x2,y2)。全零表示未标定。
    door_roi: Mapped[list] = mapped_column(JSON, default=list)

    risk_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    behavior: Mapped[str | None] = mapped_column(String(80), nullable=True)
    behavior_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    advice: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # 产物
    overlay_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    overlay_codec: Mapped[str | None] = mapped_column(String(16), nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sequence_path: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # 统计
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frames_sampled: Mapped[int | None] = mapped_column(Integer, nullable=True)
    people_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    posture_quality: Mapped[float | None] = mapped_column(Float, nullable=True)

    archived: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    raw_video_deleted: Mapped[bool] = mapped_column(Boolean, default=False)


Index("ix_analyses_status_created", Analysis.status, Analysis.created_at)

# 旧库补列：SQLite 的 ALTER TABLE ADD COLUMN 只做加法，不影响已有行。
_NEW_COLUMNS: dict[str, str] = {
    "stage": "VARCHAR(40) DEFAULT '排队中'",
    "progress": "INTEGER DEFAULT 0",
    "media_type": "VARCHAR(16) DEFAULT 'video'",
    "pose_engine": "VARCHAR(32)",
    "behavior_engine": "VARCHAR(32)",
    "advice_engine": "VARCHAR(32)",
    "door_roi": "JSON",
    "overlay_path": "VARCHAR(255)",
    "overlay_codec": "VARCHAR(16)",
    "report_path": "VARCHAR(255)",
    "sequence_path": "VARCHAR(255)",
    "duration_s": "FLOAT",
    "frames_total": "INTEGER",
    "frames_sampled": "INTEGER",
    "people_max": "INTEGER",
    "posture_quality": "FLOAT",
    "archived": "BOOLEAN DEFAULT 0",
    "raw_video_deleted": "BOOLEAN DEFAULT 0",
}


def migrate() -> list[str]:
    """把旧表补齐到当前模型，返回实际新增的列名。"""
    if not _IS_SQLITE:
        return []
    added: list[str] = []
    inspector = inspect(engine)
    if "analyses" not in inspector.get_table_names():
        return added
    existing = {column["name"] for column in inspector.get_columns("analyses")}
    with engine.begin() as connection:
        for name, definition in _NEW_COLUMNS.items():
            if name in existing:
                continue
            connection.execute(text(f"ALTER TABLE analyses ADD COLUMN {name} {definition}"))
            added.append(name)
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_analyses_created_at ON analyses (created_at)")
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS ix_analyses_status ON analyses (status)")
        )
    if added:
        logger.info("数据库已补充列：%s", ", ".join(added))
    return added


def init_db() -> None:
    settings.ensure_directories()
    Base.metadata.create_all(bind=engine)
    migrate()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
