"""骨架定义。所有姿态模型统一输出 COCO-17，下游只按名字取点。"""

from __future__ import annotations

COCO17_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

COCO17_EDGES = (
    (15, 13), (13, 11),
    (16, 14), (14, 12),
    (11, 12),
    (5, 11), (6, 12),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)

PALETTE = [
    (0, 255, 0), (0, 165, 255), (255, 128, 0), (255, 0, 255),
    (255, 255, 0), (128, 0, 255), (0, 255, 255), (255, 64, 64),
]

KEYPOINT_CONF_THRESHOLD = 0.35
