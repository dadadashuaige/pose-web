"""规则引擎：用可解释的确定性阈值替代未标定的 GCN。

覆盖六种电梯危险行为：
    摔倒 / 打架        —— 只看人体自身姿态与多人关系
    踢门 / 扒门 / 推门 / 靠门 —— 需要门区域（ROI）作为参照

每条结论都附带实测数值，报告里能看出"为什么这么判"。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .base import BEHAVIOR_LABELS, BehaviorEngine, BehaviorResult, SequenceContext
from .features import SceneFeatures, build_features

RULES_PATH = Path(__file__).with_name("rules.yaml")


def ramp(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    best = (-1, -1)
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start > best[1] - best[0]:
                best = (start, index - 1)
            start = None
    if start is not None and len(mask) - start > best[1] - best[0]:
        best = (start, len(mask) - 1)
    return best


@dataclass
class Candidate:
    behavior: str
    confidence: float
    evidence: dict


def load_rule_config(path: Path | None = None) -> dict:
    with open(path or RULES_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class RuleEngine(BehaviorEngine):
    key = "rule"
    label = "规则引擎"

    def __init__(self) -> None:
        self._config = load_rule_config()

    @property
    def config(self) -> dict:
        return self._config

    def reload(self) -> None:
        self._config = load_rule_config()

    def _params(self, name: str) -> dict:
        return ((self._config.get("rules") or {}).get(name) or {}).get("params") or {}

    def predict(self, context: SequenceContext) -> BehaviorResult:
        features = build_features(context)
        candidates: list[Candidate] = []
        candidates += self._fall(features)
        candidates += self._fight(features, context)
        if features.has_door:
            candidates += self._kick_door(features)
            candidates += self._pry_door(features)
            candidates += self._push_door(features, context)
            candidates += self._lean_door(features, context)

        door_note = "" if features.has_door else "（未标定门区域，跳过了踢门/推门/扒门/靠门的判定）"

        if not candidates:
            return BehaviorResult(
                behavior="normal",
                confidence=0.6,
                raw="no_rule_hit",
                engine=self.key,
                note="未命中任何危险行为规则。" + door_note,
                evidence={
                    "采样帧数": context.sample_count,
                    "最大同时人数": context.people_max,
                    "稳定轨迹数": len(features.tracks),
                    "各人运动强度": features.summary_lines(),
                },
            )

        priority = {
            key: index for index, key in enumerate(self._config.get("priority") or [])
        }
        candidates.sort(
            key=lambda item: (priority.get(item.behavior, 99), -item.confidence)
        )
        best = candidates[0]
        return BehaviorResult(
            behavior=best.behavior,
            confidence=best.confidence,
            raw=f"rule:{best.behavior}",
            engine=self.key,
            note=f"规则引擎判定为「{BEHAVIOR_LABELS.get(best.behavior, best.behavior)}」。" + door_note,
            evidence=best.evidence,
            reasoning="；".join(f"{k}={v}" for k, v in best.evidence.items()),
        )

    # ---- 通用的场景尺度：用于把像素距离换算成躯干长度倍数 ----
    @staticmethod
    def _scene_scale(features: SceneFeatures) -> float:
        scales = [track.scale for track in features.tracks if track.scale > 1]
        return float(np.median(scales)) if scales else 1.0

    # ---- 摔倒 -----------------------------------------------------------
    def _fall(self, features: SceneFeatures) -> list[Candidate]:
        params = self._params("fall")
        angle_min = params.get("min_torso_angle", 55)
        window = params.get("drop_window_s", 1.2)
        min_drop = params.get("min_drop", 0.45)
        conf_min = params.get("min_confidence", 0.55)
        found: list[Candidate] = []
        for track in features.tracks:
            if len(track.times) < 3:
                continue
            y = track.centroid[:, 1] / track.scale
            angle = np.nan_to_num(track.torso_angle, nan=0.0)
            for index in range(len(track.times)):
                low = np.searchsorted(track.times, track.times[index] - window)
                if index - low < 1:
                    continue
                drop = float(y[index] - np.min(y[low:index]))
                if angle[index] < angle_min or drop < min_drop:
                    continue
                confidence = 0.45 * ramp(drop, min_drop, min_drop * 2.2) + 0.55 * ramp(
                    angle[index], angle_min, 90.0
                )
                if confidence < conf_min:
                    continue
                found.append(
                    Candidate(
                        "fall",
                        confidence,
                        {
                            "说明": "重心在短时间内大幅下降，且躯干接近水平",
                            "躯干角度(度)": round(float(angle[index]), 1),
                            "重心下降(躯干倍数)": round(drop, 2),
                            "涉及人员": track.track_id,
                        },
                    )
                )
        return found

    # ---- 打架 -----------------------------------------------------------
    def _fight(self, features: SceneFeatures, context: SequenceContext) -> list[Candidate]:
        params = self._params("fight")
        min_people = params.get("min_people", 2)
        pair_max = params.get("max_pair_ratio", 2.4)
        speed_min = params.get("wrist_speed_min", 1.8)
        duration = params.get("min_duration_s", 0.6)
        conf_min = params.get("min_confidence", 0.55)
        if not features.tracks:
            return []
        scale = self._scene_scale(features)
        pair = features.nearest_pair / scale
        length = len(context.times)
        active = np.zeros(length, dtype=bool)
        detail = np.zeros(length, dtype=np.float32)
        for index in range(length):
            if features.people_count[index] < min_people or pair[index] > pair_max:
                continue
            best = 0.0
            for track in features.tracks:
                slot = np.argmin(np.abs(track.times - context.times[index]))
                if abs(track.times[slot] - context.times[index]) > 1e-3:
                    continue
                best = max(best, float(track.wrist_speed[slot]))
            if best >= speed_min:
                active[index] = True
                detail[index] = best
        start, end = longest_run(active)
        if start < 0:
            return []
        span = float(context.times[end] - context.times[start])
        if span < duration:
            return []
        window = slice(start, end + 1)
        confidence = (
            0.4 * ramp(span, duration, duration * 3)
            + 0.3 * ramp(float(np.max(detail[window])), speed_min, speed_min * 3)
            + 0.3 * (1.0 - ramp(float(np.mean(pair[window])), 0.0, pair_max))
        )
        if confidence < conf_min:
            return []
        return [
            Candidate(
                "fight",
                confidence,
                {
                    "说明": "多人近距离且上肢动作剧烈",
                    "最大同时人数": int(np.max(features.people_count[window])),
                    "最近两人距离(躯干倍数)": round(float(np.mean(pair[window])), 2),
                    "手部峰值速度": round(float(np.max(detail[window])), 2),
                    "持续时间(秒)": round(span, 2),
                },
            )
        ]

    # ---- 踢门 -----------------------------------------------------------
    def _kick_door(self, features: SceneFeatures) -> list[Candidate]:
        params = self._params("kick_door")
        speed_min = params.get("ankle_speed_min", 1.6)
        dist_max = params.get("ankle_door_max_dist", 1.0)
        split_min = params.get("min_ankle_split", 0.25)
        conf_min = params.get("min_confidence", 0.5)
        found: list[Candidate] = []
        for track in features.tracks:
            hits = np.flatnonzero(
                (track.ankle_speed >= speed_min)
                & (track.ankle_door <= dist_max)
                & (track.ankle_split >= split_min)
            )
            for index in hits:
                confidence = (
                    0.45 * ramp(track.ankle_speed[index], speed_min, speed_min * 2.5)
                    + 0.25 * (1.0 - ramp(track.ankle_door[index], 0.0, dist_max))
                    + 0.30 * ramp(track.ankle_split[index], split_min, split_min * 3)
                )
                if confidence < conf_min:
                    continue
                found.append(
                    Candidate(
                        "kick_door",
                        confidence,
                        {
                            "说明": "有抬腿动作且脚部高速接近门区域",
                            "脚踝速度(躯干倍数/秒)": round(float(track.ankle_speed[index]), 2),
                            "脚踝距门(躯干倍数)": round(float(track.ankle_door[index]), 2),
                            "两脚踝落差(躯干倍数)": round(float(track.ankle_split[index]), 2),
                            "涉及人员": track.track_id,
                        },
                    )
                )
        return found

    # ---- 扒门 -----------------------------------------------------------
    def _pry_door(self, features: SceneFeatures) -> list[Candidate]:
        params = self._params("pry_door")
        dist_max = params.get("wrist_door_max_dist", 0.55)
        speed_min = params.get("wrist_speed_min", 1.1)
        duration = params.get("min_duration_s", 0.6)
        conf_min = params.get("min_confidence", 0.5)
        found: list[Candidate] = []
        for track in features.tracks:
            if len(track.times) < 3:
                continue
            mask = (track.wrist_door <= dist_max) & (track.wrist_speed >= speed_min)
            start, end = longest_run(mask)
            if start < 0:
                continue
            span = float(track.times[end] - track.times[start])
            if span < duration:
                continue
            window = slice(start, end + 1)
            confidence = 0.5 * ramp(span, duration, duration * 3) + 0.5 * ramp(
                float(np.nanmax(track.wrist_speed[window])), speed_min, speed_min * 3
            )
            if confidence < conf_min:
                continue
            found.append(
                Candidate(
                    "pry_door",
                    confidence,
                    {
                        "说明": "双手贴近门区域并持续做开合动作",
                        "持续时间(秒)": round(span, 2),
                        "手部峰值速度": round(float(np.nanmax(track.wrist_speed[window])), 2),
                        "双手距门(躯干倍数)": round(float(np.nanmin(track.wrist_door[window])), 2),
                        "涉及人员": track.track_id,
                    },
                )
            )
        return found

    # ---- 推门 -----------------------------------------------------------
    def _push_door(self, features: SceneFeatures, context: SequenceContext) -> list[Candidate]:
        params = self._params("push_door")
        dist_max = params.get("near_door_max_dist", 1.1)
        toward_min = params.get("toward_speed_min", 0.35)
        frac_min = params.get("toward_fraction_min", 0.55)
        duration = params.get("min_duration_s", 0.8)
        conf_min = params.get("min_confidence", 0.5)
        found: list[Candidate] = []
        for track in features.tracks:
            if len(track.times) < 3:
                continue
            mask = (track.door_distance <= dist_max) & (track.toward_door >= toward_min)
            start, end = longest_run(mask)
            if start < 0:
                continue
            span = float(track.times[end] - track.times[start])
            if span < duration:
                continue
            window = slice(start, end + 1)
            toward_fraction = float(np.mean(track.toward_door[window] > 0))
            if toward_fraction < frac_min:
                continue
            confidence = (
                0.45 * ramp(span, duration, duration * 3)
                + 0.35 * ramp(float(np.max(track.toward_door[window])), toward_min, toward_min * 3)
                + 0.20 * toward_fraction
            )
            if confidence < conf_min:
                continue
            found.append(
                Candidate(
                    "push_door",
                    confidence,
                    {
                        "说明": "身体重心持续朝门区域推进",
                        "持续时间(秒)": round(span, 2),
                        "朝门速度峰值(躯干倍数/秒)": round(float(np.max(track.toward_door[window])), 2),
                        "推进帧占比": round(toward_fraction, 2),
                        "涉及人员": track.track_id,
                    },
                )
            )
        return found

    # ---- 靠门 -----------------------------------------------------------
    def _lean_door(self, features: SceneFeatures, context: SequenceContext) -> list[Candidate]:
        params = self._params("lean_door")
        dist_max = params.get("near_door_max_dist", 0.7)
        dwell_min = params.get("min_dwell_s", 1.5)
        speed_max = params.get("max_speed", 0.55)
        conf_min = params.get("min_confidence", 0.45)
        found: list[Candidate] = []
        for track in features.tracks:
            if len(track.times) < 3:
                continue
            mask = (track.door_distance <= dist_max) & (track.centroid_speed <= speed_max)
            start, end = longest_run(mask)
            if start < 0:
                continue
            span = float(track.times[end] - track.times[start])
            if span < dwell_min:
                continue
            window = slice(start, end + 1)
            closeness = 1.0 - ramp(float(np.mean(track.door_distance[window])), 0.0, dist_max)
            confidence = 0.6 * ramp(span, dwell_min, dwell_min * 3) + 0.4 * closeness
            if confidence < conf_min:
                continue
            found.append(
                Candidate(
                    "lean_door",
                    confidence,
                    {
                        "说明": "人体长时间静止贴近门区域",
                        "持续贴门(秒)": round(span, 2),
                        "平均距门(躯干倍数)": round(float(np.mean(track.door_distance[window])), 2),
                        "平均速度": round(float(np.mean(track.centroid_speed[window])), 2),
                        "涉及人员": track.track_id,
                    },
                )
            )
        return found
