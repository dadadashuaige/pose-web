"""规则引擎：六种电梯危险行为的判定。

全部用合成骨架序列测试，不依赖任何权重文件，因此可以秒级跑完。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.engines.behavior.base import SequenceContext
from app.engines.behavior.rule_engine import RuleEngine, load_rule_config
from app.engines.pose.base import PersonPose

# 归一化门区域：画面 40%~60% 宽。配合 1920 宽就是 x 768~1152。
DOOR_ROI = (0.40, 0.0, 0.60, 1.0)


def make_person(cx, cy, torso=100.0, tilt_deg=0.0, conf=0.9):
    """在 (cx, cy) 处造一个假人。cy 是髋部中点，tilt_deg 是躯干偏离竖直的角度。"""
    kps = np.zeros((17, 3), dtype=np.float32)
    rad = np.radians(tilt_deg)
    offset = np.array([np.sin(rad) * torso, -np.cos(rad) * torso])
    hip = np.array([cx, cy], dtype=np.float32)
    shoulder = hip + offset

    def put(index, point):
        kps[index] = [float(point[0]), float(point[1]), conf]

    put(5, shoulder + [-torso * 0.20, 0])
    put(6, shoulder + [torso * 0.20, 0])
    put(7, shoulder + [-torso * 0.25, torso * 0.30])
    put(8, shoulder + [torso * 0.25, torso * 0.30])
    put(9, shoulder + [-torso * 0.25, torso * 0.60])
    put(10, shoulder + [torso * 0.25, torso * 0.60])
    put(11, hip + [-torso * 0.15, 0])
    put(12, hip + [torso * 0.15, 0])
    put(13, hip + [-torso * 0.15, torso * 0.55])
    put(14, hip + [torso * 0.15, torso * 0.55])
    put(15, hip + [-torso * 0.15, torso * 1.05])
    put(16, hip + [torso * 0.15, torso * 1.05])
    put(0, shoulder + [0, -torso * 0.30])

    box = np.array(
        [cx - torso * 0.45, shoulder[1] - torso * 0.40,
         cx + torso * 0.45, hip[1] + torso * 1.15],
        dtype=np.float32,
    )
    return PersonPose(box=box, confidence=conf, keypoints=kps)


def make_context(frames, fps=10.0, roi=DOOR_ROI, width=1920, height=1080):
    """frames 里每项可以是 (cx, cy[, tilt])，也可以直接是 PersonPose 列表。"""
    people = []
    for item in frames:
        if isinstance(item, PersonPose):
            people.append([item])
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], PersonPose):
            people.append(list(item))
        else:
            values = list(item) + [0.0]
            people.append([make_person(values[0], values[1], tilt_deg=values[2])])
    times = np.arange(len(people), dtype=np.float32) / fps
    return SequenceContext(
        times=times, people=people, fps=fps, frame_stride=1,
        width=width, height=height, sample_count=len(people), door_roi=roi,
    )


def predict(frames, **kwargs):
    return RuleEngine().predict(make_context(frames, **kwargs))


# ---------------------------------------------------------------- 基准情况
def test_standing_still_far_from_door_is_normal():
    """站在轿厢另一侧不动，不该被判成任何危险行为。"""
    assert predict([(300, 700, 0.0)] * 40).behavior == "normal"


def test_standing_still_near_door_is_leaning():
    """长时间静止贴着门区域 -> 靠门。"""
    assert predict([(960, 700, 0.0)] * 40).behavior == "lean_door"


def test_fall_is_detected_by_drop_and_tilt():
    """重心骤降 + 躯干接近水平 -> 摔倒。用逐渐倒下的过程，贴近真实场景。"""
    frames = (
        [(960, 400, 0.0)] * 4
        + [
            (960, 480, 20.0),
            (960, 560, 40.0),
            (960, 640, 60.0),
            (960, 720, 75.0),
            (960, 780, 85.0),
        ]
        + [(960, 780, 85.0)] * 4
    )
    result = predict(frames)
    assert result.behavior == "fall"
    assert result.confidence > 0.55


def test_tracker_survives_fast_motion():
    """快速位移让 IoU 掉到 0 时，中心距兜底要能保持同一条轨迹。

    否则摔倒这种快速动作会被拆成两条轨迹，下降过程被切掉，反而检测不到。
    """
    from app.engines.behavior.features import build_features

    frames = (
        [(960, 400, 0.0)] * 3
        + [(960, 550, 0.0)] * 3
        + [(960, 700, 0.0)] * 3
    )
    features = build_features(make_context(frames))
    assert len(features.tracks) == 1, f"期望 1 条轨迹，实际 {len(features.tracks)} 条"


def test_two_people_close_with_fast_hands_is_fight():
    """两人靠近 + 手部快速摆动 -> 打架。"""
    frames = []
    for index in range(30):
        swing = 90.0 if index % 2 == 0 else -90.0
        left = make_person(900, 700)
        right = make_person(1080, 700)
        left.keypoints[9, 0] += swing
        left.keypoints[10, 0] += swing
        right.keypoints[9, 0] -= swing
        right.keypoints[10, 0] -= swing
        frames.append([left, right])
    assert predict(frames).behavior == "fight"


# ---------------------------------------------------------------- 门区域开关
def test_door_rules_are_skipped_without_roi():
    """没标定门区域时靠门不该被判出来，而且要给出说明。"""
    result = predict([(960, 700, 0.0)] * 40, roi=(0.0, 0.0, 0.0, 0.0))
    assert result.behavior == "normal"
    assert "门区域" in result.note


def test_same_pose_differs_with_and_without_roi():
    """同一段姿态：标了门区域判「靠门」，没标判「正常」——证明 ROI 真的参与判定。"""
    frames = [(960, 700, 0.0)] * 40
    assert predict(frames, roi=DOOR_ROI).behavior == "lean_door"
    assert predict(frames, roi=(0.0, 0.0, 0.0, 0.0)).behavior == "normal"


# ---------------------------------------------------------------- 配置完整性
def test_rules_config_is_complete():
    config = load_rule_config()
    rules = config.get("rules") or {}
    assert set(rules) == {
        "fall", "fight", "kick_door", "pry_door", "push_door", "lean_door",
    }
    for key, rule in rules.items():
        assert rule.get("label"), f"{key} 缺少 label"
        assert rule.get("risk_level") in {"high", "medium", "low"}, f"{key} 风险等级不合法"
        assert rule.get("params"), f"{key} 缺少 params"
    # 优先级列表要覆盖全部规则，否则排序时会出现未定义的 key
    assert set(config.get("priority") or []) == set(rules)


@pytest.mark.parametrize(
    "key", ["fall", "fight", "kick_door", "pry_door", "push_door", "lean_door"]
)
def test_every_rule_has_label_and_prior(key):
    from app.engines.behavior.base import BEHAVIOR_LABELS, DANGER_PRIOR

    assert key in BEHAVIOR_LABELS
    assert key in DANGER_PRIOR
