from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.app.config import settings
from backend.app.schemas import AnalyzeResponse
from backend.app.services.llm_service import ActionAnalysisAgent
from backend.app.services.pose_service import PoseService

settings.ensure_directories()
configured_model = Path(settings.yolo_model)
# Keep a downloaded default weight in the mounted models directory. Explicit paths
# (for example, a custom .pt file) are respected as-is.
model_source = (
    settings.model_dir / configured_model
    if not configured_model.is_absolute() and configured_model.parent == Path(".")
    else configured_model
)
app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)
app.mount("/files", StaticFiles(directory=settings.data_dir), name="files")
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

pose_service = PoseService(str(model_source), settings.yolo_confidence)
analysis_agent = ActionAnalysisAgent(settings.openai_api_key, settings.openai_base_url, settings.openai_model)
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "yolo_model": str(model_source)}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)) -> AnalyzeResponse:
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "仅支持 JPG、PNG、WEBP 图片")

    content = await file.read()
    if not content:
        raise HTTPException(400, "上传文件为空")
    # MAX_UPLOAD_MB=0 表示不限制大小（与 .env.example 的约定一致）。
    # 原实现直接拿 0 做比较，导致上限变成 0 字节，任何文件都会被 413 拒绝。
    if settings.max_upload_mb > 0 and len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"图片不能超过 {settings.max_upload_mb} MB")
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "文件不是有效图片") from None

    analysis_id = uuid.uuid4().hex
    original_path = settings.upload_dir / f"{analysis_id}{suffix}"
    visualization_path = settings.report_dir / f"{analysis_id}_pose.jpg"
    original_path.write_bytes(content)
    try:
        detections = pose_service.analyze(original_path, visualization_path)
        report, provider = analysis_agent.analyze(detections)
    except Exception as exc:
        original_path.unlink(missing_ok=True)
        visualization_path.unlink(missing_ok=True)
        raise HTTPException(500, f"姿态分析失败：{exc}") from exc

    return AnalyzeResponse(
        analysis_id=analysis_id,
        original_url=f"/files/uploads/{original_path.name}",
        visualization_url=f"/files/reports/{visualization_path.name}",
        model=str(model_source),
        llm_provider=provider,
        detections=detections,
        report=report,
    )
