"""从采样帧的关键点里提取可解释的运动学特征。

所有距离与速度都按躯干长度归一化，因此不随人物远近变化。
带门区域的场景还会额外算出「人员/手脚距门多远」「是否朝门移动」，
这是区分踢门、扒门、推门、靠门四类行为的基础。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..pose.base import PersonPose
from .base import SequenceContext

CONF = 0.30


def rect_distance(px: float, py: float, rect: tuple[float, float, float, float]) -> float:
    """点到矩形的最短距离；点在矩形内时为 0。"""
    x1, y1, x2, y2 = rect
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


@dataclass
class Track:
    track_id: int
    times: np.ndarray        # (T,)
    boxes: np.ndarray        # (T,4)
    keypoints: np.ndarray    # (T,17,3)
    centroid: np.ndarray     # (T,2)
    torso_angle: np.ndarray  # (T,) 偏离竖直的角度
    torso_len: np.ndarray    # (T,)
    wrist_speed: np.ndarray  # (T,) 躯干长度/秒
    ankle_speed: np.ndarray
    ankle_split: np.ndarray  # (T,) 两脚踝垂直落差
    centroid_speed: np.ndarray
    door_distance: np.ndarray  # (T,) 重心到门区域的距离
    wrist_door: np.ndarray     # (T,) 双手里离门更近的那只的距离
    ankle_door: np.ndarray     # (T,) 双脚里离门更近的那只的距离
    toward_door: np.ndarray    # (T,) 朝门方向的带符号速度
    scale: float = 1.0

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) else 0.0


@dataclass
class SceneFeatures:
    tracks: list[Track] = field(default_factory=list)
    people_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    nearest_pair: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    duration: float = 0.0
    has_door: bool = False

    def summary_lines(self, limit: int = 4) -> list[str]:
        lines = []
        for track in self.tracks[:limit]:
            angle = np.nan_to_num(track.torso_angle, nan=0.0)
            text = (
                "人员#{id}：出现{frames}帧/时长{dur:.1f}秒；躯干最大偏离竖直{angle:.0f}°；"
                "手部峰值速度{wrist:.2f}、脚踝峰值速度{ankle:.2f}、重心峰值速度{centroid:.2f}"
                "（单位：躯干长度/秒）".format(
                    id=track.track_id,
                    frames=len(track.times),
                    dur=track.duration,
                    angle=float(np.nanmax(angle)),
                    wrist=float(np.nanmax(track.wrist_speed)),
                    ankle=float(np.nanmax(track.ankle_speed)),
                    centroid=float(np.nanmax(track.centroid_speed)),
                )
            )
            if self.has_door:
                text += "；距门 最小{door_min:.2f}/平均{door_mean:.2f}，朝门峰值速度{toward:.2f}".format(
                    door_min=float(np.nanmin(track.door_distance)),
                    door_mean=float(np.nanmean(track.door_distance)),
                    toward=float(np.nanmax(track.toward_door)),
                )
            lines.append(text)
        return lines


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _midpoint(points: np.ndarray, a: int, b: int) -> np.ndarray | None:
    if points[a, 2] < CONF and points[b, 2] < CONF:
        return None
    if points[a, 2] < CONF:
        return points[b, :2]
    if points[b, 2] < CONF:
        return points[a, :2]
    return (points[a, :2] + points[b, :2]) / 2.0


def _centroid(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    valid = points[points[:, 2] >= CONF]
    if len(valid) >= 3:
        return valid[:, :2].mean(axis=0)
    return (box[:2] + box[2:4]) / 2.0


def _torso_angle(points: np.ndarray) -> float:
    shoulder = _midpoint(points, 5, 6)
    hip = _midpoint(points, 11, 12)
    if shoulder is None or hip is None:
        return float("nan")
    dx = float(shoulder[0] - hip[0])
    dy = float(shoulder[1] - hip[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float("nan")
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def _torso_length(points: np.ndarray, box: np.ndarray) -> float:
    shoulder = _midpoint(points, 5, 6)
    hip = _midpoint(points, 11, 12)
    if shoulder is not None and hip is not None:
        length = float(np.linalg.norm(shoulder - hip))
        if length > 5:
            return length
    return max(30.0, float(box[3] - box[1]) * 0.35)


def _limb_speed(
    positions: np.ndarray, valid: np.ndarray, times: np.ndarray, scale: float
) -> np.ndarray:
    """positions: (T,2,2) 左右两个同名关节点；valid: (T,2) 可见性。"""
    flat = positions.reshape(len(positions), -1)
    flat = np.where(np.repeat(valid, 2, axis=1), flat, np.nan)
    if len(flat) < 2:
        return np.zeros(len(flat), dtype=np.float32)
    filled = flat.copy()
    for column in range(filled.shape[1]):
        series = filled[:, column]
        mask = ~np.isnan(series)
        if mask.sum() >= 2:
            series[~mask] = np.interp(
                np.flatnonzero(~mask), np.flatnonzero(mask), series[mask]
            )
        else:
            series[:] = np.nanmean(series) if mask.any() else 0.0
        filled[:, column] = series
    deltas = np.linalg.norm(np.diff(filled, axis=0), axis=-1)
    dt = np.diff(times)
    dt[dt <= 0] = 1e-3
    return np.concatenate([[0.0], deltas / dt]).astype(np.float32) / max(scale, 1.0)


def _track_from_poses(
    track_id: int,
    times: np.ndarray,
    poses: list[PersonPose],
    door_rect: tuple[float, float, float, float] | None,
) -> Track:
    boxes = np.stack([p.box for p in poses]).astype(np.float32)
    kpts = np.stack([p.keypoints for p in poses]).astype(np.float32)
    centroid = np.stack([_centroid(k, b) for k, b in zip(kpts, boxes)]).astype(np.float32)
    torso_angle = np.array([_torso_angle(k) for k in kpts], dtype=np.float32)
    torso_len = np.array([_torso_length(k, b) for k, b in zip(kpts, boxes)], dtype=np.float32)
    scale = float(np.nanmedian(torso_len)) if len(torso_len) else 1.0
    scale = scale if scale > 1 else 1.0

    def limb(a: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        return kpts[:, [a, b], :2], kpts[:, [a, b], 2] >= CONF

    wrist_pos, wrist_ok = limb(9, 10)
    ankle_pos, ankle_ok = limb(15, 16)
    wrist_speed = _limb_speed(wrist_pos, wrist_ok, times, scale)
    ankle_speed = _limb_speed(ankle_pos, ankle_ok, times, scale)

    if len(centroid) >= 2:
        dt = np.diff(times)
        dt[dt <= 0] = 1e-3
        speed = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(centroid, axis=0), axis=-1) / dt]
        ) / scale
    else:
        speed = np.zeros(len(centroid), dtype=np.float32)

    # 两脚踝垂直落差：用于区分「真的抬腿踢」与「原地站立」
    ankle_visible = (
        (kpts[:, 15, 2] >= CONF) & (kpts[:, 16, 2] >= CONF)
    )
    ankle_split = np.where(ankle_visible, np.abs(kpts[:, 15, 1] - kpts[:, 16, 1]), 0.0)
    ankle_split = (ankle_split / scale).astype(np.float32)

    # ---- 门相关特征 -------------------------------------------------
    count = len(centroid)
    if door_rect is None:
        zero = np.zeros(count, dtype=np.float32)
        door_distance = wrist_door = ankle_door = toward_door = zero
    else:
        door_distance = np.array(
            [rect_distance(c[0], c[1], door_rect) for c in centroid], dtype=np.float32
        ) / scale
        wrist_door = np.array(
            [min(rect_distance(k[i, 0], k[i, 1], door_rect) for i in (9, 10)) for k in kpts],
            dtype=np.float32,
        ) / scale
        ankle_door = np.array(
            [min(rect_distance(k[i, 0], k[i, 1], door_rect) for i in (15, 16)) for k in kpts],
            dtype=np.float32,
        ) / scale
        toward_door = np.zeros(count, dtype=np.float32)
        if count >= 2:
            centre = np.array(
                [(door_rect[0] + door_rect[2]) / 2.0, (door_rect[1] + door_rect[3]) / 2.0]
            )
            velocity = np.diff(centroid, axis=0)
            dt = np.diff(times)
            dt[dt <= 0] = 1e-3
            velocity = velocity / dt[:, None]
            direction = centre[None, :] - centroid[:-1]
            norm = np.linalg.norm(direction, axis=1)
            norm[norm < 1e-6] = 1e-6
            toward_door[1:] = (
                (velocity * (direction / norm[:, None])).sum(axis=1).astype(np.float32) / scale
            )

    return Track(
        track_id=track_id,
        times=times,
        boxes=boxes,
        keypoints=kpts,
        centroid=centroid,
        torso_angle=torso_angle,
        torso_len=torso_len,
        wrist_speed=wrist_speed.astype(np.float32),
        ankle_speed=ankle_speed.astype(np.float32),
        ankle_split=ankle_split,
        centroid_speed=speed.astype(np.float32),
        door_distance=door_distance,
        wrist_door=wrist_door,
        ankle_door=ankle_door,
        toward_door=toward_door,
        scale=scale,
    )


def build_features(context: SequenceContext, min_frames: int = 3) -> SceneFeatures:
    """用 IoU 做简易关联，把逐帧检测整理成人员轨迹，并算出场景级特征。"""
    times = np.asarray(context.times, dtype=np.float32)
    door_rect = context.door_rect if context.has_door else None
    features = SceneFeatures(duration=context.duration, has_door=door_rect is not None)

    tracks: dict[int, list[tuple[int, PersonPose]]] = {}
    next_id = 1
    last_box: dict[int, np.ndarray] = {}
    centroid_frames: list[list[np.ndarray]] = []

    for position, people in enumerate(context.people):
        used: set[int] = set()
        assigned: dict[int, int] = {}
        current_centroids: list[np.ndarray] = []

        for index, person in enumerate(people):
            current_centroids.append(_centroid(person.keypoints, person.box))

        # 第一轮：按 IoU 匹配，这是最可靠的依据
        for index, person in enumerate(people):
            best_id, best_iou = None, 0.30
            for track_id, box in last_box.items():
                if track_id in used:
                    continue
                score = _iou(person.box, box)
                if score > best_iou:
                    best_id, best_iou = track_id, score
            if best_id is not None:
                used.add(best_id)
                assigned[index] = best_id

        # 第二轮：IoU 匹配不上的，用中心距兜底。
        # 摔倒这类快速动作会让 IoU 直接掉到 0，只按 IoU 会把同一个人拆成两条轨迹，
        # 结果就是"下降过程"被切没了，摔倒反而检测不到。
        for index, person in enumerate(people):
            if index in assigned:
                continue
            centre = current_centroids[index]
            height = float(person.box[3] - person.box[1])
            best_id, best_dist = None, float("inf")
            for track_id, box in last_box.items():
                if track_id in used:
                    continue
                other = np.array(
                    [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32
                )
                distance = float(np.linalg.norm(centre - other))
                # 允许的位移不超过大半个身高；再远就当作另一个人
                limit = 0.75 * max(height, float(box[3] - box[1]))
                if distance < limit and distance < best_dist:
                    best_id, best_dist = track_id, distance
            if best_id is not None:
                used.add(best_id)
                assigned[index] = best_id

        for index, person in enumerate(people):
            track_id = assigned.get(index)
            if track_id is None:
                track_id = next_id
                next_id += 1
            last_box[track_id] = person.box
            tracks.setdefault(track_id, []).append((position, person))
        centroid_frames.append(current_centroids)

    for track_id, items in tracks.items():
        if len(items) < min_frames:
            continue
        items.sort(key=lambda item: item[0])
        features.tracks.append(
            _track_from_poses(
                track_id,
                times[[position for position, _ in items]],
                [person for _, person in items],
                door_rect,
            )
        )

    features.people_count = np.array(
        [len(items) for items in context.people], dtype=np.float32
    )
    nearest = np.zeros(len(context.people), dtype=np.float32)
    for position, centroids in enumerate(centroid_frames):
        if len(centroids) < 2:
            nearest[position] = 0.0
            continue
        matrix = np.stack(centroids)
        distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        nearest[position] = float(distances.min())
    features.nearest_pair = nearest
    return features
