"""PCG-GCN（ST-GCN Joint+Bone）行为识别。

保留但默认不启用：当前权重是 NTU RGB+D 60 类通用模型，没有用电梯数据训练，
输出的类别无法与电梯危险行为对齐（实测会给出 cheer up / hopping 之类的高置信度错判）。
后续用电梯数据微调后，把 BEHAVIOR_ENGINE 改成 stgcn 即可启用，其余代码不用动。
"""

from __future__ import annotations

import sys
import threading

import numpy as np

from ...config import settings
from .base import BehaviorEngine, BehaviorResult, SequenceContext

_lock = threading.Lock()

NTU_ACTIONS = [
    "drink water", "eat meal", "brush teeth", "brush hair", "drop", "pickup", "throw",
    "sit down", "stand up", "clapping", "reading", "writing", "tear paper", "wear jacket",
    "take off jacket", "wear shoe", "take off shoe", "wear glasses", "take off glasses",
    "put on cap", "take off cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping", "jumping", "phone call", "play with phone", "typing",
    "point to something", "take a selfie", "check time", "rub hands", "nod head",
    "shake head", "wipe face", "salute", "cross hands", "sneeze", "staggering", "falling",
    "touch head", "touch chest", "back pain", "neck pain", "nausea", "fan self",
    "punching/slapping", "kicking person", "pushing", "patting", "pointing at person",
    "hugging", "giving object", "touching pocket", "handshake", "walking towards",
    "walking apart",
]

# 只有这几个 NTU 类别能对应到电梯危险行为，其余一律视为无法判定。
NTU_TO_BEHAVIOR = {
    "falling": "falling",
    "staggering": "falling",
    "punching/slapping": "fighting",
    "kicking person": "fighting",
    "pushing": "pushing",
    "kicking something": "kicking",
    "jumping": "jumping",
    "hopping": "jumping",
}


class STGCNEngine(BehaviorEngine):
    key = "stgcn"
    label = "PCG-GCN"

    def __init__(self) -> None:
        self._model = None
        self._device = None

    @property
    def available(self) -> bool:
        return bool(
            settings.stgcn_weights
            and settings.stgcn_weights.is_file()
            and settings.stgcn_repo
            and settings.stgcn_repo.is_dir()
        )

    @property
    def unavailable_reason(self) -> str:
        if settings.stgcn_repo is None:
            return "未配置 STGCN_REPO（PCG-GCN 源码目录）"
        if not settings.stgcn_repo.is_dir():
            return f"STGCN_REPO 指向的目录不存在：{settings.stgcn_repo}"
        if settings.stgcn_weights is None:
            return "未配置 STGCN_WEIGHTS（PCG-GCN 权重路径）"
        if not settings.stgcn_weights.is_file():
            return f"STGCN_WEIGHTS 指向的文件不存在：{settings.stgcn_weights}"
        return ""

    def _load(self) -> None:
        if self._model is not None:
            return
        with _lock:
            if self._model is not None:
                return
            if not self.available:
                raise FileNotFoundError(self.unavailable_reason)
            assert settings.stgcn_repo is not None and settings.stgcn_weights is not None
            if str(settings.stgcn_repo) not in sys.path:
                sys.path.insert(0, str(settings.stgcn_repo))
            import torch
            from net.st_gcn_joint_bone import Model

            self._device = torch.device(
                "cuda:0"
                if settings.device.startswith("cuda") and torch.cuda.is_available()
                else "cpu"
            )
            model = Model(
                in_channels=3,
                num_class=60,
                dropout=0.5,
                edge_importance_weighting=True,
                graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
            )
            state = torch.load(
                settings.stgcn_weights, map_location=self._device, weights_only=True
            )
            model.load_state_dict(state, strict=True)
            self._model = model.to(self._device).eval()

    @staticmethod
    def coco_to_ntu(sequence: np.ndarray) -> np.ndarray:
        """COCO-17 -> NTU-25，坐标以躯干中心归一化。"""
        count = len(sequence)
        out = np.zeros((3, count, 25, 1), dtype=np.float32)
        for index in range(count):
            joints = np.asarray(sequence[index], dtype=np.float32)
            if joints.shape != (17, 3):
                continue
            left_hip, right_hip = joints[11], joints[12]
            left_shoulder = joints[5]
            centre = (left_hip[:2] + right_hip[:2]) / 2
            scale = max(float(np.linalg.norm(left_shoulder[:2] - right_hip[:2])), 1.0)

            def put(target: int, source: int) -> None:
                out[0, index, target, 0] = (joints[source, 0] - centre[0]) / scale
                out[1, index, target, 0] = (joints[source, 1] - centre[1]) / scale
                out[2, index, target, 0] = joints[source, 2]

            pairs = {
                0: 11, 1: 11, 2: 5, 3: 0, 4: 5, 5: 7, 6: 9, 7: 9, 8: 6, 9: 8,
                10: 10, 11: 10, 12: 11, 13: 13, 14: 15, 15: 15, 16: 12, 17: 14,
                18: 16, 19: 16, 20: 5, 21: 9, 22: 9, 23: 10, 24: 10,
            }
            for ntu, coco in pairs.items():
                put(ntu, coco)
            out[:2, index, 1, 0] = (
                out[:2, index, 12, 0] + out[:2, index, 16, 0]
            ) / 2
            out[:2, index, 20, 0] = (
                out[:2, index, 4, 0] + out[:2, index, 8, 0]
            ) / 2
        return out

    def predict(self, context: SequenceContext) -> BehaviorResult:
        self._load()
        import torch

        # 取每帧置信度最高的人，凑成固定长度序列
        primary = []
        for people in context.people:
            if people:
                best = max(people, key=lambda item: item.confidence)
                primary.append(best.keypoints)
        if len(primary) < 8:
            return BehaviorResult(
                behavior="unanalyzed",
                confidence=0.0,
                raw="",
                engine=self.key,
                note="有效人体姿态帧不足 8 帧，PCG-GCN 无法给出结论。",
            )

        target = 64
        if len(primary) >= target:
            indices = np.linspace(0, len(primary) - 1, target).astype(int)
            sequence = [primary[i] for i in indices]
        else:
            sequence = primary + [primary[-1]] * (target - len(primary))

        data = self.coco_to_ntu(np.asarray(sequence, dtype=np.float32))
        tensor = torch.from_numpy(data).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0].cpu().numpy()
        index = int(probabilities.argmax())
        raw = NTU_ACTIONS[index] if index < len(NTU_ACTIONS) else f"class_{index}"
        mapped = NTU_TO_BEHAVIOR.get(raw, "normal")
        return BehaviorResult(
            behavior=mapped,
            confidence=float(probabilities[index]),
            raw=raw,
            engine=self.key,
            note=(
                f"PCG-GCN 原始类别为「{raw}」（NTU-60 通用权重）。"
                + ("" if raw in NTU_TO_BEHAVIOR else "该类别不在电梯危险行为映射表中，因此记为正常。")
            ),
            evidence={
                "NTU 原始类别": raw,
                "类别置信度": round(float(probabilities[index]), 4),
                "是否可映射": "是" if raw in NTU_TO_BEHAVIOR else "否",
            },
        )
