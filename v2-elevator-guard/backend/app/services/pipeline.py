"""全流程编排。

上传视频 -> 逐帧姿态估计 -> 骨架可视化视频 -> 行为判定 -> 风险评分
        -> 处置建议（可比对多种引擎）-> PDF 报告 -> 落库

每一步用到的引擎都从注册表按名字取，pipeline 不认识具体实现，
所以换模型 / 临时关掉行为识别都不需要改这个文件。
"""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..config import settings
from ..database import Analysis, SessionLocal
from ..engines import get_advice_engine, get_behavior_engine, get_pose_engine
from ..engines.advice import ADVICE_ENGINES
from ..engines.behavior import BEHAVIOR_ENGINES, SequenceContext
from .risk_agent import RiskAgent
from .visualizer import (
    OverlayWriter,
    draw_banner,
    draw_door_roi,
    draw_people,
    target_size,
)

logger = logging.getLogger(__name__)

risk_agent = RiskAgent()

KEYFRAME_COUNT = 3


def process_analysis(
    analysis_id: str,
    pose_engine_key: str | None = None,
    behavior_engine_key: str | None = None,
    advice_engine_key: str | None = None,
    compare_advice: bool = False,
    door_roi: tuple | None = None,
) -> None:
    db = SessionLocal()
    item = db.get(Analysis, analysis_id)
    if item is None:
        db.close()
        return

    def stage(name: str, progress: int) -> None:
        item.stage = name
        item.progress = max(0, min(100, progress))
        db.commit()

    try:
        pose_key = pose_engine_key or item.pose_engine or settings.pose_engine
        behavior_key = behavior_engine_key or item.behavior_engine or settings.behavior_engine
        advice_key = advice_engine_key or item.advice_engine or settings.advice_engine
        # 本次任务的门区域：优先用调用方传入的，其次是任务记录里存的，最后是全局默认。
        roi = tuple(door_roi) if door_roi else tuple(item.door_roi or settings.door_roi)
        if len(roi) != 4:
            roi = (0.0, 0.0, 0.0, 0.0)

        item.status = "processing"
        item.stage = "准备中"
        item.progress = 0
        item.pose_engine = pose_key
        item.behavior_engine = behavior_key
        item.advice_engine = advice_key
        item.door_roi = list(roi)
        db.commit()

        video_path = _raw_video_path(item)
        if video_path is None:
            raise RuntimeError("原始视频已被清理，无法重新分析。请重新上传。")

        pose_engine = get_pose_engine(pose_key)
        if not pose_engine.available:
            raise RuntimeError(f"姿态引擎 {pose_key} 不可用，请检查权重路径。")

        stage("姿态估计", 5)
        timings: dict[str, float] = {}
        clock = time.perf_counter()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("无法读取视频文件，请确认是 MP4/AVI/MOV 等 OpenCV 支持的格式。")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps if fps else 0.0

        stride = max(1, settings.frame_stride)
        planned = total // stride if total else 0
        if planned > settings.max_sample_frames:
            stride = max(stride, int(np.ceil(total / settings.max_sample_frames)))

        out_size = target_size(width, height, settings.overlay_max_width)
        scale_x = out_size[0] / float(width) if width else 1.0
        scale_y = out_size[1] / float(height) if height else 1.0
        overlay_dir = settings.data_dir / "videos"
        # 每个采样帧要覆盖 stride/fps 秒，因此在 target_fps 下需要重复写这么多帧
        target_fps = settings.overlay_fps if settings.overlay_fps > 0 else fps
        repeat = max(1, int(round(target_fps * stride / fps))) if fps else 1
        writer = (
            OverlayWriter(overlay_dir, target_fps, out_size, stem=analysis_id)
            if settings.overlay_enabled
            else None
        )
        if writer is not None and not writer.ok:
            writer = None

        times: list[float] = []
        poses: list[list] = []
        qualities: list[float] = []
        people_max = 0
        frame_index = 0
        sampled = 0
        infer_seconds = 0.0
        encode_seconds = 0.0
        keyframe_targets = _keyframe_indices(total, stride, KEYFRAME_COUNT)
        keyframes: list[Path] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                tick = time.perf_counter()
                people = pose_engine.detect(frame)
                infer_seconds += time.perf_counter() - tick
                people_max = max(people_max, len(people))
                times.append(frame_index / fps)
                poses.append(people)
                sampled += 1
                for person in people:
                    qualities.append(float(person.keypoints[:, 2].mean()))

                # 先缩放到输出尺寸再绘制：省掉每帧一次整图 resize，
                # 图元绘制量也降到原来的 45%（1280x720 vs 1906x1080）。
                canvas = cv2.resize(frame, out_size)
                draw_people(canvas, people, scale=(scale_x, scale_y))
                draw_door_roi(canvas, roi)
                draw_banner(
                    canvas,
                    "t=%.1fs  people=%d  pose=%s  behavior=%s  door=%s"
                    % (
                        frame_index / fps,
                        len(people),
                        pose_key,
                        behavior_key,
                        "set" if roi[2] > roi[0] else "unset",
                    ),
                )
                tick = time.perf_counter()
                if writer is not None:
                    writer.write(canvas, repeat=repeat)
                encode_seconds += time.perf_counter() - tick
                if frame_index in keyframe_targets and len(keyframes) < KEYFRAME_COUNT:
                    thumb = settings.data_dir / "reports" / f"{analysis_id}_key_{len(keyframes)}.jpg"
                    cv2.imwrite(str(thumb), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    keyframes.append(thumb)

                if sampled % 20 == 0 and total:
                    progress = 5 + int(60 * min(1.0, frame_index / max(1, total)))
                    stage("姿态估计", progress)
            frame_index += 1

        timings["姿态推理"] = round(infer_seconds, 2)
        timings["绘制与视频编码"] = round(encode_seconds, 2)
        timings["姿态阶段合计"] = round(time.perf_counter() - clock, 2)
        cap.release()
        if sampled == 0:
            raise RuntimeError("未能从视频中解码出任何画面。")

        stage("生成可视化", 70)
        overlay = writer.close() if writer is not None else None
        if overlay is not None and not overlay.readable:
            logger.warning("可视化视频回读校验失败，已丢弃：%s", overlay.path)
            overlay = None

        quality = float(np.mean(qualities)) if qualities else 0.0
        item.duration_s = duration
        item.frames_total = total
        item.frames_sampled = sampled
        item.people_max = people_max
        item.posture_quality = quality
        db.commit()

        context = SequenceContext(
            times=np.asarray(times, dtype=np.float32),
            people=poses,
            fps=fps,
            frame_stride=stride,
            width=width,
            height=height,
            sample_count=sampled,
            door_roi=roi,
        )

        stage("行为判定", 78)
        clock = time.perf_counter()
        behavior = get_behavior_engine(behavior_key).predict(context)
        timings["行为判定"] = round(time.perf_counter() - clock, 2)

        stage("风险评估", 84)
        clock = time.perf_counter()
        risk = risk_agent.evaluate(
            behavior.behavior, behavior.confidence, people_max, quality
        )
        timings["风险评分"] = round(time.perf_counter() - clock, 2)

        stage("生成建议", 88)
        clock = time.perf_counter()
        advice_context = {
            "duration": duration,
            "sample_count": sampled,
            "people_max": people_max,
            "pose_engine": pose_engine.label,
            "behavior_engine": behavior_key,
        }
        advice_variants: dict[str, dict] = {}
        keys = [advice_key]
        if compare_advice:
            keys = list(ADVICE_ENGINES.keys())
        primary = None
        for key in keys:
            try:
                result = get_advice_engine(key).generate(
                    behavior, risk.to_dict(), advice_context
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("建议引擎 %s 失败：%s", key, exc)
                continue
            advice_variants[key] = result.to_dict()
            if primary is None or key == advice_key:
                primary = result
        if primary is None:
            primary = get_advice_engine("template").generate(
                behavior, risk.to_dict(), advice_context
            )
            advice_variants[primary.engine] = primary.to_dict()
        timings["建议生成"] = round(time.perf_counter() - clock, 2)

        stage("生成报告", 93)
        clock = time.perf_counter()
        report_path = settings.data_dir / "reports" / f"{analysis_id}.pdf"
        behavior_dict = behavior.to_dict()
        risk_dict = risk.to_dict()
        try:
            from .report_service import ReportService

            ReportService().create(
                report_path,
                item,
                behavior=behavior_dict,
                risk=risk_dict,
                advice=primary.to_dict(),
                advice_variants=advice_variants,
                overlay=overlay,
                keyframes=keyframes,
                pose_engine_label=pose_engine.label,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("报告生成失败：%s", traceback.format_exc())
            report_path = None
        timings["PDF 报告"] = round(time.perf_counter() - clock, 2)

        item.behavior = behavior.behavior
        item.behavior_score = behavior.confidence
        item.risk_level = risk.level
        item.risk_score = risk.score
        item.advice = primary.to_dict()
        item.overlay_path = overlay.path.name if overlay else None
        item.overlay_codec = overlay.codec if overlay else None
        item.report_path = report_path.name if report_path and report_path.exists() else None
        item.summary = _summary(pose_engine.label, behavior, sampled, people_max, overlay)
        item.result = {
            "pose_engine": pose_key,
            "behavior_engine": behavior_key,
            "advice_engine": primary.engine,
            "behavior": behavior_dict,
            "risk": risk_dict,
            "advice_variants": advice_variants,
            "overlay": overlay.to_dict() if overlay else None,
            "keyframes": [path.name for path in keyframes],
            "frames_total": total,
            "frames_sampled": sampled,
            "frame_stride": stride,
            "fps": round(fps, 3),
            "people_max": people_max,
            "posture_quality": round(quality, 4),
            "duration_s": round(duration, 2),
            "resolution": f"{width}x{height}",
            "engines": _engine_snapshot(pose_key, behavior_key, advice_key),
            "warnings": _warnings(behavior, overlay, sampled, roi, behavior_key),
            "door_roi": list(roi),
            "timings": timings,
            "overlay_codec_hint": (
                "VP8/VP9 为浏览器原生支持的 WebM；若回退到 mp4v，"
                "浏览器可能无法直接播放，请下载后用本地播放器查看。"
            ),
        }
        item.status = "completed"
        item.stage = "完成"
        item.progress = 100
        item.completed_at = datetime.utcnow()
        db.commit()
        logger.info("分析完成：%s -> %s", analysis_id, behavior.behavior)

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        item = db.get(Analysis, analysis_id)
        if item is not None:
            item.status = "failed"
            item.stage = "失败"
            item.error = f"{type(exc).__name__}: {exc}"
            item.completed_at = datetime.utcnow()
            db.commit()
        logger.error("分析失败 %s：%s", analysis_id, traceback.format_exc())
    finally:
        db.close()


def _raw_video_path(item: Analysis) -> Path | None:
    """原始视频按 `<id>_<原文件名>` 存放在 data/uploads 下。"""
    if item.raw_video_deleted:
        return None
    candidate = settings.data_dir / "uploads" / f"{item.id}_{item.filename}"
    if candidate.exists():
        return candidate
    folder = settings.data_dir / "uploads"
    for path in folder.glob(f"{item.id}_*"):
        if path.is_file():
            return path
    return None


def _keyframe_indices(total: int, stride: int, count: int) -> set[int]:
    if total <= 0:
        return {0}
    picks = set()
    for fraction in np.linspace(0.1, 0.9, count):
        index = int(total * fraction)
        picks.add(index - index % stride)
    return picks


def _summary(pose_label, behavior, sampled, people_max, overlay) -> str:
    parts = [
        f"使用 {pose_label} 分析了 {sampled} 帧，最大同时人数 {people_max}",
        f"行为判定：{behavior.label}"
        + (f"（置信度 {behavior.confidence:.2f}）" if behavior.confidence else ""),
    ]
    if overlay is not None:
        parts.append(f"已生成可视化视频（{overlay.codec}, {overlay.frames} 帧）")
    else:
        parts.append("未生成可视化视频")
    return "；".join(parts) + "。"


def _engine_snapshot(pose_key: str, behavior_key: str, advice_key: str) -> dict:
    return {
        "pose": pose_key,
        "behavior": behavior_key,
        "advice": advice_key,
        "behavior_available": sorted(BEHAVIOR_ENGINES),
        "advice_available": sorted(ADVICE_ENGINES),
    }


def _warnings(behavior, overlay, sampled: int, roi: tuple, behavior_key: str) -> list[str]:
    warnings: list[str] = []
    door_set = roi[2] > roi[0] and roi[3] > roi[1]
    if behavior_key == "rule" and not door_set:
        warnings.append(
            "门区域尚未标定，「踢门 / 推门 / 扒门 / 靠门」四类行为本次没有参与判定，"
            "只能给出摔倒与打架的结论。请在结果页拖拽标定门区域后重新分析。"
        )
    elif behavior_key == "rule" and door_set:
        warnings.append(
            "门相关结论依赖门区域标定的准确性，建议人工复核关键帧后再采信。"
        )
    if behavior.behavior == "unanalyzed":
        warnings.append(
            "本次未启用行为识别引擎（或引擎不可用），只做了姿态与可视化，没有危险行为判定。"
        )
    if behavior.engine == "stgcn" and behavior.raw not in {
        "falling", "staggering", "punching/slapping", "kicking person",
        "pushing", "kicking something", "jumping", "hopping",
    }:
        warnings.append(
            f"PCG-GCN 输出的是 NTU 通用类别「{behavior.raw}」，与电梯危险行为不对应，"
            "该结论不可作为风险依据。"
        )
    if overlay is None:
        warnings.append("可视化视频未生成，可能是编码器不可用，或本次未开启该功能。")
    elif overlay.codec == "mp4v":
        warnings.append(
            "可视化视频回退到了 mp4v 编码，多数浏览器无法直接播放，"
            "请下载后用本地播放器查看。"
        )
    if sampled < 15:
        warnings.append("有效采样帧数偏少，时序结论的稳定性有限。")
    return warnings
