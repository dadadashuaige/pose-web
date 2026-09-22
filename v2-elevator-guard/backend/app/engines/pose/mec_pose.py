"""MEC-Pose：自训练的 YOLOv5-Pose 权重。

要点：这个 checkpoint 是 pickle 序列化的，反序列化时需要名为 `models` 的模块，
所以必须先把训练仓库根目录插进 sys.path。这一步放在 _load() 里而不是文件顶部，
这样即使路径不对，服务也能正常启动，只有真正调用时才会报错。
"""

from __future__ import annotations

import sys
import threading

import cv2
import numpy as np

from ...config import settings
from .base import PersonPose, PoseEngine, device_str

_lock = threading.Lock()
_torch_patched = False


def _patch_torch_load() -> None:
    """torch>=2.6 默认 weights_only=True，无法反序列化 YOLOv5 的 checkpoint。"""
    global _torch_patched
    if _torch_patched:
        return
    import torch

    original = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = patched
    _torch_patched = True


def letterbox(image: np.ndarray, size: int):
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    nw, nh = int(round(w * gain)), int(round(h * gain))
    pad_w, pad_h = size - nw, size - nh
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, gain, left, top


class MecPoseEngine(PoseEngine):
    key = "mec"
    label = "MEC-Pose"

    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._img_size = 640

    @property
    def available(self) -> bool:
        return bool(
            settings.mec_weights
            and settings.mec_weights.is_file()
            and settings.mec_repo
            and settings.mec_repo.is_dir()
        )

    @property
    def unavailable_reason(self) -> str:
        if settings.mec_weights is None:
            return "未配置 MEC_WEIGHTS（自训练权重路径），可在 backend/.env 中指定"
        if not settings.mec_weights.is_file():
            return f"MEC_WEIGHTS 指向的文件不存在：{settings.mec_weights}"
        if settings.mec_repo is None:
            return "未配置 MEC_REPO（YOLOv5-Pose 源码目录）"
        if not settings.mec_repo.is_dir():
            return f"MEC_REPO 指向的目录不存在：{settings.mec_repo}"
        return ""

    def _load(self) -> None:
        if self._model is not None:
            return
        with _lock:
            if self._model is not None:
                return
            if not self.available:
                raise FileNotFoundError(self.unavailable_reason)
            assert settings.mec_weights is not None and settings.mec_repo is not None

            _patch_torch_load()
            if str(settings.mec_repo) not in sys.path:
                sys.path.insert(0, str(settings.mec_repo))

            import torch
            from models.experimental import attempt_load

            self._device = torch.device(device_str())
            self._model = attempt_load(
                str(settings.mec_weights), map_location=self._device
            ).eval()
            stride = int(self._model.stride.max())
            self._img_size = int(np.ceil(640 / stride) * stride)

    def detect(self, frame: np.ndarray) -> list[PersonPose]:
        self._load()
        import torch
        from utils.general import non_max_suppression

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, gain, pad_x, pad_y = letterbox(rgb, self._img_size)
        tensor = (
            torch.from_numpy(padded.transpose(2, 0, 1))
            .to(self._device)
            .float()
            .unsqueeze(0)
            / 255.0
        )
        with torch.no_grad():
            prediction = self._model(tensor)[0]
            detections = non_max_suppression(
                prediction, settings.pose_confidence, 0.45, kpt_label=True
            )[0]
        if detections is None or len(detections) == 0:
            return []

        h, w = frame.shape[:2]
        people: list[PersonPose] = []
        for row in detections.detach().cpu().numpy():
            x1, y1, x2, y2, confidence = row[:5]
            kpts = row[6:].reshape(-1, 3).astype(np.float32).copy()[:17]
            if len(kpts) < 17:
                continue
            kpts[:, 0] = np.clip((kpts[:, 0] - pad_x) / gain, 0, w - 1)
            kpts[:, 1] = np.clip((kpts[:, 1] - pad_y) / gain, 0, h - 1)
            box = np.array(
                [
                    (x1 - pad_x) / gain,
                    (y1 - pad_y) / gain,
                    (x2 - pad_x) / gain,
                    (y2 - pad_y) / gain,
                ],
                dtype=np.float32,
            )
            people.append(
                PersonPose(box=box, confidence=float(confidence), keypoints=kpts)
            )
        return people
