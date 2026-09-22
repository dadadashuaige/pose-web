"""骨架绘制与可视化视频写出。

编码策略（按顺序尝试，取第一个能真正写出内容的）：
    VP9 / WebM  ->  VP8 / WebM  ->  mp4v / MP4

为什么不直接用 H.264：这台机器的 OpenCV 缺 openh264 库，
`avc1` 会“打开成功”但写出一个空文件——静默失败最难查。
WebM(VP8/VP9) 是 Chrome / Edge / Firefox 原生支持的格式，`<video>` 能直接播；
mp4v 只作为最后的兜底（能下载用播放器看，但浏览器多半播不了）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..engines.skeleton import COCO17_EDGES, KEYPOINT_CONF_THRESHOLD, PALETTE

logger = logging.getLogger(__name__)

# 顺序即优先级。VP8 排在 VP9 前面是有实测依据的：
# 同样的 1280x720 帧，VP9 编码约 257ms/帧，VP8 只要 24ms/帧（相差约 10 倍），
# 而两者都是浏览器原生支持的 WebM，播放体验没有差别。
CODEC_CANDIDATES: list[tuple[str, str, str]] = [
    ("VP80", ".webm", "video/webm"),
    ("VP90", ".webm", "video/webm"),
    ("mp4v", ".mp4", "video/mp4"),
]


def draw_people(
    image: np.ndarray,
    people,
    draw_boxes: bool = True,
    scale: tuple[float, float] | None = None,
) -> None:
    """在图上就地绘制骨架与人体框。people 是 PersonPose 列表。

    scale 用于把原图坐标映射到已经缩放过的画布上。
    先在原分辨率上画、再整帧缩放，比"先缩放、再按比例画"慢一倍左右——
    1080p 上每帧多一次 2MP 的 resize，还要在 2MP 上做几十次图元绘制。
    """
    sx, sy = scale if scale else (1.0, 1.0)
    for index, person in enumerate(people):
        color = PALETTE[index % len(PALETTE)]
        if draw_boxes:
            x1, y1, x2, y2 = (
                int(person.box[0] * sx),
                int(person.box[1] * sy),
                int(person.box[2] * sx),
                int(person.box[3] * sy),
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"#{index + 1} {person.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        keypoints = person.keypoints
        for a, b in COCO17_EDGES:
            if a >= len(keypoints) or b >= len(keypoints):
                continue
            if (
                keypoints[a, 2] >= KEYPOINT_CONF_THRESHOLD
                and keypoints[b, 2] >= KEYPOINT_CONF_THRESHOLD
            ):
                cv2.line(
                    image,
                    (int(keypoints[a, 0] * sx), int(keypoints[a, 1] * sy)),
                    (int(keypoints[b, 0] * sx), int(keypoints[b, 1] * sy)),
                    color,
                    3,
                )
        for k in range(len(keypoints)):
            if keypoints[k, 2] >= KEYPOINT_CONF_THRESHOLD:
                centre = (int(keypoints[k, 0] * sx), int(keypoints[k, 1] * sy))
                cv2.circle(image, centre, 5, (0, 0, 255), -1)
                cv2.circle(image, centre, 5, (255, 255, 255), 1)


def draw_banner(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(
        image, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )


def draw_door_roi(image: np.ndarray, roi: tuple, label: str = "door ROI") -> None:
    """把标定好的门区域画在画面上。

    四条门相关规则（踢门/推门/扒门/靠门）都依赖这块区域，
    把它画出来，人一眼就能判断当前标定得对不对。
    """
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return
    height, width = image.shape[:2]
    left, top = int(round(x1 * width)), int(round(y1 * height))
    right, bottom = int(round(x2 * width)), int(round(y2 * height))
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right - left < 2 or bottom - top < 2:
        return
    colour = (60, 200, 255)
    # 只对门区域那一小块做半透明叠加。
    # 早先是整帧 copy + addWeighted，1080p 下每帧两次全图运算，
    # 一个 14 秒视频的导出因此慢了整整一倍。
    patch = image[top:bottom, left:right]
    tint = patch.copy()
    cv2.rectangle(tint, (0, 0), (tint.shape[1], tint.shape[0]), colour, -1)
    cv2.addWeighted(tint, 0.12, patch, 0.88, 0, patch)
    cv2.rectangle(image, (left, top), (right, bottom), colour, 2)
    cv2.putText(
        image, label, (left + 6, max(20, top + 24)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
    )


def target_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """按最大宽度等比缩放，并保证是偶数（部分编码器要求）。"""
    if width <= max_width:
        out_w, out_h = width, height
    else:
        scale = max_width / width
        out_w, out_h = int(width * scale), int(height * scale)
    return max(2, out_w - out_w % 2), max(2, out_h - out_h % 2)


@dataclass
class OverlayVideo:
    path: Path
    codec: str
    mime: str
    frames: int
    readable: bool

    def to_dict(self) -> dict:
        return {
            "file": self.path.name,
            "codec": self.codec,
            "mime": self.mime,
            "frames": self.frames,
            "readable": self.readable,
        }


class OverlayWriter:
    """按顺序尝试多种编码，写出后回读校验，避免“文件生成了但播不了”。"""

    def __init__(
        self,
        directory: Path,
        fps: float,
        size: tuple[int, int],
        stem: str = "overlay",
    ):
        self.directory = Path(directory)
        self.fps = max(1.0, min(fps, 60.0))
        self.size = size
        # 文件名必须带任务 ID：早先固定叫 overlay.webm，导致所有任务互相覆盖，
        # 删除任务时也匹配不到这个文件。
        self.stem = stem
        self.writer = None
        self.path: Path | None = None
        self.codec = ""
        self.mime = ""
        self.frames = 0
        self._open()

    def _open(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        for codec, suffix, mime in CODEC_CANDIDATES:
            path = self.directory / f"{self.stem}{suffix}"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*codec), self.fps, self.size
            )
            if writer.isOpened():
                self.writer, self.path, self.codec, self.mime = writer, path, codec, mime
                logger.info("可视化视频编码：%s -> %s", codec, path.name)
                return
            writer.release()
        logger.warning("没有可用的视频编码器，跳过可视化视频生成")

    @property
    def ok(self) -> bool:
        return self.writer is not None

    def write(self, frame: np.ndarray, repeat: int = 1) -> None:
        if self.writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        for _ in range(max(1, repeat)):
            self.writer.write(frame)
            self.frames += 1

    def close(self) -> OverlayVideo | None:
        if self.writer is None or self.path is None:
            return None
        self.writer.release()
        self.writer = None
        readable, count = _verify(self.path)
        return OverlayVideo(
            path=self.path,
            codec=self.codec,
            mime=self.mime,
            frames=count or self.frames,
            readable=readable,
        )


def _verify(path: Path) -> tuple[bool, int]:
    """回读校验：文件存在、能打开、帧数大于 0。"""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False, 0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False, 0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, _ = cap.read()
        cap.release()
        return bool(ok) and count > 0, count
    except Exception:  # noqa: BLE001
        return False, 0
