"""FastAPI 入口：路由、上传、记录管理、文件下载。"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from .config import BACKEND_ROOT, settings
from .database import Analysis, SessionLocal, engine, init_db
from .engines.advice import ADVICE_ENGINES
from .engines.behavior import BEHAVIOR_ENGINES, BEHAVIOR_LABELS
from .engines.pose import POSE_ENGINES
from .schemas import (
    AnalysisItem,
    AnalysisList,
    AnalysisOptions,
    CleanupResult,
    EngineInfo,
    HealthResponse,
    KnowledgeRebuild,
)
from .services.knowledge import get_knowledge_base
from .services.llm_client import get_llm_client
from .services.visualizer import CODEC_CANDIDATES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

app = FastAPI(title=settings.app_name, version="0.2.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """前端资源禁止缓存。

    FastAPI 的 StaticFiles 不发送 Cache-Control，浏览器会按 Last-Modified
    推算一个“启发式新鲜期”；老文件因此可能长时间不重新校验，改完前端后
    用户看到的仍是旧页面（表现为新功能全部不生效、列表按旧格式解析而显示为空）。
    """
    response = await call_next(request)
    path = request.url.path
    if path in {"/", "/index.html"} or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup() -> None:
    settings.ensure_directories()
    init_db()
    knowledge = get_knowledge_base()
    knowledge.ensure_ready()
    logger.info(
        "知识库就绪：%s（%d 条）", knowledge.backend, knowledge.count()
    )
    if settings.raw_video_retain_days > 0:
        removed = _purge_expired_raw_videos()
        if removed:
            logger.info("已清理 %d 个过期原始视频", removed)


# ---------------------------------------------------------------- 序列化
def _serialize(item: Analysis) -> AnalysisItem:
    payload = AnalysisItem.model_validate(item)
    payload.behavior_label = BEHAVIOR_LABELS.get(item.behavior or "", item.behavior or "")
    payload.has_report = bool(
        item.report_path
        and (settings.data_dir / "reports" / item.report_path).exists()
    )
    payload.has_video = bool(
        item.overlay_path and (settings.data_dir / "videos" / item.overlay_path).exists()
    )
    return payload


def _engine_infos() -> dict[str, list[EngineInfo]]:
    from .engines.advice import ADVICE_ENGINES
    from .engines.behavior import BEHAVIOR_ENGINES
    from .engines.pose import POSE_ENGINES

    def build(table: dict, specs: dict) -> list[EngineInfo]:
        """可用性以引擎自身判断为准——例如 LLM 引擎取决于有没有配 Key，
        光看配置里的权重路径是判断不出来的。"""
        out = []
        for key, engine_class in table.items():
            spec = specs.get(key)
            try:
                engine = engine_class()
                available = bool(engine.available)
                reason = engine.unavailable_reason if not available else ""
            except Exception as exc:  # noqa: BLE001
                available, reason = False, f"{type(exc).__name__}: {exc}"
            out.append(
                EngineInfo(
                    key=key,
                    label=spec.label if spec else getattr(engine_class, "label", key),
                    note=reason or (spec.note if spec else ""),
                    kind=spec.kind if spec else "behavior",
                    available=available,
                    weights=str(spec.weights) if spec and spec.weights else "",
                )
            )
        return out

    return {
        "pose": build(POSE_ENGINES, settings.pose_engines),
        "behavior": build(BEHAVIOR_ENGINES, settings.behavior_engines),
        "advice": build(ADVICE_ENGINES, settings.advice_engines),
    }


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    import torch

    knowledge = get_knowledge_base()
    client = get_llm_client()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        device={
            "device": settings.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        engines=_engine_infos(),
        defaults={
            "pose": settings.pose_engine,
            "behavior": settings.behavior_engine,
            "advice": settings.advice_engine,
        },
        llm_configured=client.available,
        llm_model=settings.llm_model,
        knowledge_entries=knowledge.count(),
        knowledge_backend=knowledge.backend,
        knowledge_detail=knowledge.detail,
        overlay_codecs=[codec for codec, _, _ in CODEC_CANDIDATES],
        door_roi=list(settings.door_roi),
    )


@app.get("/api/engines")
def engines() -> dict:
    return {"engines": _engine_infos(), "defaults": settings.defaults()}


# ---------------------------------------------------------------- 上传与分析
@app.post("/api/analyses", response_model=AnalysisItem, status_code=202)
async def upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    pose_engine: str = Form(""),
    behavior_engine: str = Form(""),
    advice_engine: str = Form(""),
    compare_advice: str = Form("0"),
    door_roi: str = Form(""),
    db: Session = Depends(get_db),
) -> AnalysisItem:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "仅支持 MP4、AVI、MOV、MKV、WEBM 视频或 JPG、PNG 图片。")

    pose_key = _validate_engine(pose_engine, POSE_ENGINES, "姿态", settings.pose_engine)
    behavior_key = _validate_engine(behavior_engine, BEHAVIOR_ENGINES, "行为", "none")
    advice_key = _validate_engine(advice_engine, ADVICE_ENGINES, "建议", "template")
    roi = _parse_roi(door_roi)

    analysis_id = str(uuid4())
    safe_name = Path(video.filename or "upload").name
    destination = settings.data_dir / "uploads" / f"{analysis_id}_{safe_name}"
    with destination.open("wb") as output:
        shutil.copyfileobj(video.file, output)
    if destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(400, "上传的文件为空。")

    item = Analysis(
        id=analysis_id,
        filename=safe_name,
        status="queued",
        stage="排队中",
        progress=0,
        media_type="image" if suffix in IMAGE_SUFFIXES else "video",
        created_at=datetime.utcnow(),
        pose_engine=pose_key,
        behavior_engine=behavior_key,
        advice_engine=advice_key,
        door_roi=list(roi),
    )
    db.add(item)
    db.commit()
    db.refresh(item)

    background_tasks.add_task(
        _run,
        analysis_id,
        pose_key,
        behavior_key,
        advice_key,
        compare_advice in {"1", "true", "True", "on"},
        roi,
    )
    return _serialize(item)


def _run(*args) -> None:
    from .services.pipeline import process_analysis

    process_analysis(*args)


@app.post("/api/analyses/{analysis_id}/rerun", response_model=AnalysisItem)
def rerun(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    pose_engine: str = Form(""),
    behavior_engine: str = Form(""),
    advice_engine: str = Form(""),
    compare_advice: str = Form("0"),
    door_roi: str = Form(""),
    db: Session = Depends(get_db),
) -> AnalysisItem:
    item = db.get(Analysis, analysis_id)
    if item is None:
        raise HTTPException(404, "分析任务不存在")
    if item.status in {"queued", "processing"}:
        raise HTTPException(409, "该任务正在处理中，请稍候。")
    if item.raw_video_deleted:
        raise HTTPException(409, "原始视频已被清理，无法重新分析，请重新上传。")

    pose_key = _validate_engine(pose_engine, POSE_ENGINES, "姿态", item.pose_engine or "mec")
    behavior_key = _validate_engine(
        behavior_engine, BEHAVIOR_ENGINES, "行为", item.behavior_engine or "none"
    )
    advice_key = _validate_engine(
        advice_engine, ADVICE_ENGINES, "建议", item.advice_engine or "template"
    )
    roi = _parse_roi(door_roi) or tuple(item.door_roi or settings.door_roi)

    item.status = "queued"
    item.stage = "排队中"
    item.progress = 0
    item.error = None
    item.pose_engine = pose_key
    item.behavior_engine = behavior_key
    item.advice_engine = advice_key
    item.door_roi = list(roi)
    db.commit()
    db.refresh(item)

    background_tasks.add_task(
        _run,
        analysis_id,
        pose_key,
        behavior_key,
        advice_key,
        compare_advice in {"1", "true", "True", "on"},
        roi,
    )
    return _serialize(item)


# ---------------------------------------------------------------- 记录管理
@app.get("/api/analyses", response_model=AnalysisList)
def list_analyses(
    status: str = "",
    risk: str = "",
    q: str = "",
    archived: str = "0",
    page: int = 1,
    page_size: int = 12,
    db: Session = Depends(get_db),
) -> AnalysisList:
    page = max(1, page)
    page_size = max(1, min(100, page_size))
    query = select(Analysis)
    if status:
        query = query.where(Analysis.status.in_([s for s in status.split(",") if s]))
    if risk:
        query = query.where(Analysis.risk_level.in_([r for r in risk.split(",") if r]))
    if q.strip():
        pattern = f"%{q.strip()}%"
        query = query.where(
            or_(Analysis.filename.like(pattern), Analysis.summary.like(pattern))
        )
    query = query.where(Analysis.archived.is_(bool(archived in {"1", "true", "True"})))

    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = (
        db.scalars(
            query.order_by(Analysis.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .all()
    )
    return AnalysisList(
        items=[_serialize(item) for item in rows],
        total=total,
        page=page,
        page_size=page_size,
        stats=_stats(db),
    )


def _stats(db: Session) -> dict:
    rows = db.execute(
        select(Analysis.status, func.count()).group_by(Analysis.status)
    ).all()
    by_status = {status: count for status, count in rows}
    risk_rows = db.execute(
        select(Analysis.risk_level, func.count())
        .where(Analysis.status == "completed")
        .group_by(Analysis.risk_level)
    ).all()
    return {
        "by_status": by_status,
        "by_risk": {level or "未判定": count for level, count in risk_rows},
        "total": sum(by_status.values()),
        "archived": db.scalar(
            select(func.count()).select_from(Analysis).where(Analysis.archived.is_(True))
        )
        or 0,
    }


@app.get("/api/analyses/{analysis_id}", response_model=AnalysisItem)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)) -> AnalysisItem:
    item = db.get(Analysis, analysis_id)
    if item is None:
        raise HTTPException(404, "分析任务不存在")
    return _serialize(item)


@app.post("/api/analyses/{analysis_id}/archive", response_model=AnalysisItem)
def archive(
    analysis_id: str, archived: str = Form("1"), db: Session = Depends(get_db)
) -> AnalysisItem:
    item = db.get(Analysis, analysis_id)
    if item is None:
        raise HTTPException(404, "分析任务不存在")
    item.archived = archived in {"1", "true", "True"}
    db.commit()
    db.refresh(item)
    return _serialize(item)


@app.delete("/api/analyses/{analysis_id}", response_model=CleanupResult)
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)) -> CleanupResult:
    item = db.get(Analysis, analysis_id)
    if item is None:
        raise HTTPException(404, "分析任务不存在")
    files = _purge_files(item)
    db.delete(item)
    db.commit()
    return CleanupResult(deleted=1, detail=f"已删除任务及其 {files} 个产物文件")


@app.post("/api/analyses/bulk-delete", response_model=CleanupResult)
def bulk_delete(payload: dict, db: Session = Depends(get_db)) -> CleanupResult:
    ids = [str(item) for item in (payload.get("ids") or [])]
    if not ids:
        raise HTTPException(400, "未选择任何任务")
    removed = 0
    files = 0
    for analysis_id in ids:
        item = db.get(Analysis, analysis_id)
        if item is None:
            continue
        files += _purge_files(item)
        db.delete(item)
        removed += 1
    db.commit()
    return CleanupResult(deleted=removed, detail=f"共删除 {files} 个产物文件")


@app.post("/api/maintenance/cleanup-failed", response_model=CleanupResult)
def cleanup_failed(db: Session = Depends(get_db)) -> CleanupResult:
    rows = db.scalars(select(Analysis).where(Analysis.status == "failed")).all()
    files = 0
    for item in rows:
        files += _purge_files(item)
        db.delete(item)
    db.commit()
    return CleanupResult(deleted=len(rows), detail=f"共删除 {files} 个产物文件")


@app.post("/api/maintenance/cleanup-raw", response_model=CleanupResult)
def cleanup_raw(days: int = 0, db: Session = Depends(get_db)) -> CleanupResult:
    limit = days or settings.raw_video_retain_days
    if limit <= 0:
        raise HTTPException(400, "未指定保留天数（days），且 .env 中 RAW_VIDEO_RETAIN_DAYS 为 0")
    removed = _purge_expired_raw_videos(limit, db)
    return CleanupResult(deleted=removed, detail=f"已清理 {removed} 个超过 {limit} 天的原始视频")


def _purge_expired_raw_videos(days: int | None = None, db: Session | None = None) -> int:
    limit = days or settings.raw_video_retain_days
    if limit <= 0:
        return 0
    threshold = datetime.utcnow() - timedelta(days=limit)
    own_session = db is None
    db = db or SessionLocal()
    removed = 0
    try:
        rows = db.scalars(
            select(Analysis).where(
                Analysis.created_at < threshold,
                Analysis.raw_video_deleted.is_(False),
            )
        ).all()
        for item in rows:
            path = settings.data_dir / "uploads" / f"{item.id}_{item.filename}"
            if path.exists():
                path.unlink()
                removed += 1
            else:
                for extra in (settings.data_dir / "uploads").glob(f"{item.id}_*"):
                    extra.unlink(missing_ok=True)
                    removed += 1
            item.raw_video_deleted = True
        db.commit()
    finally:
        if own_session:
            db.close()
    return removed


def _purge_files(item: Analysis) -> int:
    """删除任务的所有产物文件，返回删除数量。"""
    count = 0
    candidates = [
        settings.data_dir / "reports" / f"{item.id}.pdf",
        settings.data_dir / "uploads" / f"{item.id}_{item.filename}",
    ]
    for extra in (settings.data_dir / "videos").glob(f"{item.id}*"):
        candidates.append(extra)
    for extra in (settings.data_dir / "reports").glob(f"{item.id}*"):
        candidates.append(extra)
    for extra in (settings.data_dir / "uploads").glob(f"{item.id}_*"):
        candidates.append(extra)
    for path in candidates:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                count += 1
        except OSError:
            continue
    return count


# ---------------------------------------------------------------- 文件下载
@app.get("/api/analyses/{analysis_id}/report")
def download_report(analysis_id: str, db: Session = Depends(get_db)) -> FileResponse:
    item = db.get(Analysis, analysis_id)
    if item is None:
        raise HTTPException(404, "分析任务不存在")
    report = settings.data_dir / "reports" / f"{analysis_id}.pdf"
    if not report.exists():
        raise HTTPException(409, "报告尚未生成")
    return FileResponse(
        report, media_type="application/pdf", filename=f"elevator-risk-{analysis_id}.pdf"
    )


@app.get("/api/analyses/{analysis_id}/video")
def download_video(analysis_id: str, db: Session = Depends(get_db)) -> FileResponse:
    item = db.get(Analysis, analysis_id)
    if item is None or not item.overlay_path:
        raise HTTPException(404, "可视化视频不存在")
    path = settings.data_dir / "videos" / item.overlay_path
    if not path.exists():
        raise HTTPException(404, "可视化视频文件已被清理")
    mime = "video/webm" if path.suffix == ".webm" else "video/mp4"
    return FileResponse(path, media_type=mime, filename=f"{analysis_id}-overlay{path.suffix}")


@app.get("/api/analyses/{analysis_id}/keyframe/{index}")
def download_keyframe(
    analysis_id: str, index: int, db: Session = Depends(get_db)
) -> FileResponse:
    item = db.get(Analysis, analysis_id)
    if item is None:
        raise HTTPException(404, "分析任务不存在")
    path = settings.data_dir / "reports" / f"{analysis_id}_key_{index}.jpg"
    if not path.exists():
        raise HTTPException(404, "关键帧不存在")
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------- 知识库
@app.get("/api/knowledge")
def knowledge_status() -> dict:
    kb = get_knowledge_base()
    return {
        "entries": kb.count(),
        "backend": kb.backend,
        "detail": kb.detail,
        "chunks": [
            {
                "entry_id": chunk.entry_id,
                "title": chunk.title,
                "action": chunk.action,
                "risk_level": chunk.risk_level,
                "source": chunk.source,
            }
            for chunk in kb.chunks
        ],
    }


@app.post("/api/knowledge/rebuild", response_model=KnowledgeRebuild)
def knowledge_rebuild() -> KnowledgeRebuild:
    entries, backend, detail = get_knowledge_base().rebuild()
    return KnowledgeRebuild(entries=entries, backend=backend, detail=detail)


@app.post("/api/knowledge/search")
def knowledge_search(payload: dict) -> dict:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise HTTPException(400, "query 不能为空")
    hits = get_knowledge_base().search(query, int(payload.get("top_k") or settings.rag_top_k))
    return {
        "query": query,
        "hits": [
            {
                "entry_id": hit.entry_id,
                "title": hit.title,
                "source": hit.source,
                "snippet": hit.snippet(),
                "score": hit.score,
                "risk_level": hit.risk_level,
            }
            for hit in hits
        ],
    }


# ---------------------------------------------------------------- 规则
@app.get("/api/rules")
def get_rules() -> dict:
    from .engines.behavior import load_rule_config

    config = load_rule_config()
    return {
        "rules": {
            key: {
                "label": value.get("label"),
                "risk_level": value.get("risk_level"),
                "params": value.get("params"),
            }
            for key, value in (config.get("rules") or {}).items()
        },
        "priority": config.get("priority") or [],
    }


def _validate_engine(value: str, table: dict, kind: str, fallback: str) -> str:
    key = (value or fallback).lower()
    if key not in table:
        raise HTTPException(400, f"未知的{kind}引擎：{value}（可选 {sorted(table)}）")
    return key


def _parse_roi(raw: str) -> tuple[float, float, float, float]:
    """解析表单里的门区域；为空或非法时返回全零（表示"未标定"）。"""
    text = (raw or "").strip()
    if not text:
        return (0.0, 0.0, 0.0, 0.0)
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError as exc:
        raise HTTPException(400, "door_roi 需要 4 个 0~1 之间的数字") from exc
    if len(values) != 4:
        raise HTTPException(400, "door_roi 需要 4 个 0~1 之间的数字")
    x1, y1, x2, y2 = (max(0.0, min(1.0, v)) for v in values)
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(400, "door_roi 需要满足 x1<x2 且 y1<y2")
    return (x1, y1, x2, y2)


@app.post("/api/settings/door-roi")
def save_door_roi(payload: dict) -> dict:
    """把门区域写回 backend/.env，作为后续新建任务的默认值。

    只影响之后新建的任务；已经跑过的任务保留自己当时的标定值，
    需要的话在结果页点「用当前门区域重新分析」。
    """
    raw = payload.get("roi")
    text = ",".join(str(v) for v in raw) if isinstance(raw, (list, tuple)) else str(raw or "")
    roi = _parse_roi(text)
    if roi == (0.0, 0.0, 0.0, 0.0):
        raise HTTPException(400, "门区域不能为空")

    env_path = BACKEND_ROOT / ".env"
    line = "DOOR_ROI=" + ",".join(f"{v:.4f}" for v in roi)
    try:
        content = env_path.read_text(encoding="utf-8-sig") if env_path.exists() else ""
        lines = [
            item for item in content.splitlines()
            if not item.strip().startswith("DOOR_ROI=")
        ]
        lines.append(line)
        env_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"写入 .env 失败：{exc}") from exc
    return {
        "door_roi": list(roi),
        "note": "已写入 backend/.env。重启服务后生效为新任务的默认值；"
                "对已有任务请用「用当前门区域重新分析」。",
    }


# 静态网页放在最后 mount，避免拦截上面的 /api 路由
app.mount(
    "/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="web"
)
