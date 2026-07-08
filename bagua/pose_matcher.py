"""Pose matching system -- match MediaPipe landmarks against reference poses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_LANDMARK_IDS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot": 31,
    "right_foot": 32,
}


LEFT_RIGHT_PAIRS = {
    "left_shoulder": "right_shoulder",
    "left_elbow": "right_elbow",
    "left_wrist": "right_wrist",
    "left_hip": "right_hip",
    "left_knee": "right_knee",
    "left_ankle": "right_ankle",
    "left_foot": "right_foot",
}


FEATURE_WEIGHTS = {
    "left_foot_y": 2.2,
    "right_foot_y": 2.2,
    "left_knee_y": 2.0,
    "right_knee_y": 2.0,
    "foot_y_gap": 5.0,
    "knee_y_gap": 3.6,
    "left_foot_x": 1.1,
    "right_foot_x": 1.1,
    "foot_x_span": 1.8,
    "left_knee_angle": 1.8,
    "right_knee_angle": 1.8,
    "left_shin_angle": 1.2,
    "right_shin_angle": 1.2,
}


FEATURE_TOLERANCES = {
    "left_foot_y": 0.18,
    "right_foot_y": 0.18,
    "left_knee_y": 0.18,
    "right_knee_y": 0.18,
    "foot_y_gap": 0.20,
    "knee_y_gap": 0.20,
    "left_foot_x": 0.25,
    "right_foot_x": 0.25,
    "foot_x_span": 0.24,
    "left_knee_angle": 0.22,
    "right_knee_angle": 0.22,
    "left_shin_angle": 0.25,
    "right_shin_angle": 0.25,
}


@dataclass
class PoseMatch:
    name: str
    confidence: float  # 0-1
    frame_count: int  # consecutive frames this pose has been active
    keypoint_scores: dict[str, float]


class PoseMatcher:
    """Match incoming MediaPipe pose landmarks against stored reference poses.

    Reference poses are stored as JSON files with landmark coordinates.
    Each file can have multiple keypoints; matching uses weighted similarity
    with per-joint tolerance.
    """

    def __init__(self, reference_dir: str | Path = "reference_poses") -> None:
        self.reference_dir = Path(reference_dir)
        self.templates: dict[str, dict] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        if not self.reference_dir.exists():
            return
        for f in sorted(self.reference_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                name = data.get("name", f.stem)
                self.templates[name] = data
            except (json.JSONDecodeError, KeyError):
                continue

    @property
    def known_moves(self) -> list[str]:
        return list(self.templates.keys())

    def match(self, landmarks: list, visibility_threshold: float = 0.35) -> list[PoseMatch]:
        """Match current landmarks against all templates.

        Args:
            landmarks: MediaPipe PoseLandmark list (33 items)
            visibility_threshold: ignore landmarks below this visibility

        Returns:
            List of PoseMatch sorted by confidence descending
        """
        if not self.templates:
            return []

        results = []
        for name, template in self.templates.items():
            confidence = self._compute_similarity(landmarks, template, visibility_threshold)
            if confidence > 0:
                results.append(PoseMatch(
                    name=name,
                    confidence=confidence,
                    frame_count=0,
                    keypoint_scores={},
                ))
        return sorted(results, key=lambda m: m.confidence, reverse=True)

    def _compute_similarity(
        self, landmarks: list, template: dict, vis_threshold: float
    ) -> float:
        weights = template.get("match_weights", {})
        keypoints = template.get("keypoints", {})
        if not keypoints:
            return 0.0

        landmark_ids = template.get("landmark_ids", {})
        use_body_normalized = template.get("normalization") == "body_center_scale"
        normalized = None
        if use_body_normalized:
            normalized = normalize_landmarks(landmarks, landmark_ids, vis_threshold)
            if not normalized:
                return 0.0
            template_points = {
                joint_name: (float(ref["x"]), float(ref["y"]), float(ref["z"]))
                for joint_name, ref in keypoints.items()
            }
            best_score = 0.0
            for candidate in pose_variants(normalized):
                point_score = self._compute_point_similarity(candidate, template, weights)
                leg_score = compute_feature_similarity(
                    extract_pose_features(candidate),
                    extract_pose_features(template_points),
                )
                score = point_score * 0.54 + leg_score * 0.46
                best_score = max(best_score, score)
            return best_score

        total_weight = 0.0
        weighted_score = 0.0

        for joint_name, ref in keypoints.items():
            idx = landmark_ids.get(joint_name)
            if idx is None:
                continue

            lm = landmarks[idx]
            if float(lm.visibility) < vis_threshold:
                continue

            w = weights.get(joint_name, 1.0)
            total_weight += w

            # Spatial similarity -- how close is this joint to the reference?
            if normalized is not None:
                cur = normalized.get(joint_name)
                if cur is None:
                    continue
                dx = cur[0] - ref["x"]
                dy = cur[1] - ref["y"]
                dz = (cur[2] - ref["z"]) * 0.35
            else:
                dx = float(lm.x) - ref["x"]
                dy = float(lm.y) - ref["y"]
                dz = float(lm.z) - ref["z"]
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))

            # Tolerance from template or default. Use a continuous falloff so
            # similar Tai Chi stances do not all tie at a perfect score.
            tolerance = ref.get("tolerance", 0.12)
            score = max(0.0, 1.0 - dist / (tolerance * 2.8))
            weighted_score += w * score

        if total_weight == 0:
            return 0.0
        return weighted_score / total_weight

    def _compute_point_similarity(
        self,
        normalized: dict[str, tuple[float, float, float]],
        template: dict,
        weights: dict,
    ) -> float:
        keypoints = template.get("keypoints", {})
        total_weight = 0.0
        weighted_score = 0.0
        for joint_name, ref in keypoints.items():
            cur = normalized.get(joint_name)
            if cur is None:
                continue
            w = weights.get(joint_name, 1.0)
            total_weight += w
            dx = cur[0] - ref["x"]
            dy = cur[1] - ref["y"]
            dz = (cur[2] - ref["z"]) * 0.35
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            tolerance = ref.get("tolerance", 0.12)
            score = max(0.0, 1.0 - dist / (tolerance * 2.8))
            weighted_score += w * score
        if total_weight == 0.0:
            return 0.0
        return weighted_score / total_weight


def normalize_landmarks(
    landmarks: list,
    landmark_ids: dict[str, int] | None = None,
    visibility_threshold: float = 0.0,
) -> dict[str, tuple[float, float, float]]:
    """Normalize MediaPipe landmarks around the hip center and body size.

    Raw MediaPipe x/y coordinates depend heavily on crop size and where the
    performer stands in frame. Reference templates from a cut-up infographic
    therefore need a body-relative coordinate system to match real video.
    """
    landmark_ids = landmark_ids or DEFAULT_LANDMARK_IDS
    visible_points: list[tuple[float, float, float]] = []
    for idx in landmark_ids.values():
        lm = landmarks[idx]
        if float(lm.visibility) >= visibility_threshold:
            visible_points.append((float(lm.x), float(lm.y), float(lm.z)))
    if len(visible_points) < 6:
        return {}

    def midpoint(a: int, b: int) -> np.ndarray | None:
        la = landmarks[a]
        lb = landmarks[b]
        if float(la.visibility) < visibility_threshold or float(lb.visibility) < visibility_threshold:
            return None
        return np.array(
            [
                (float(la.x) + float(lb.x)) * 0.5,
                (float(la.y) + float(lb.y)) * 0.5,
                (float(la.z) + float(lb.z)) * 0.5,
            ],
            dtype=np.float32,
        )

    hip_center = midpoint(23, 24)
    shoulder_center = midpoint(11, 12)
    if hip_center is not None:
        center = hip_center
    elif shoulder_center is not None:
        center = shoulder_center
    else:
        center = np.mean(np.asarray(visible_points, dtype=np.float32), axis=0)

    pts = np.asarray(visible_points, dtype=np.float32)
    span_xy = np.ptp(pts[:, :2], axis=0)
    bbox_scale = float(max(span_xy[0], span_xy[1]))
    torso_scale = 0.0
    if hip_center is not None and shoulder_center is not None:
        torso_scale = float(np.linalg.norm(hip_center[:2] - shoulder_center[:2]) * 2.25)
    scale = max(bbox_scale, torso_scale, 0.08)

    normalized = {}
    for name, idx in landmark_ids.items():
        lm = landmarks[idx]
        if float(lm.visibility) < visibility_threshold:
            continue
        p = np.array([float(lm.x), float(lm.y), float(lm.z)], dtype=np.float32)
        rel = (p - center) / scale
        normalized[name] = (float(rel[0]), float(rel[1]), float(rel[2]))
    return normalized


def pose_variants(
    points: dict[str, tuple[float, float, float]]
) -> list[dict[str, tuple[float, float, float]]]:
    variants = []
    seen = set()
    for mirror_x in (False, True):
        for swap_lr in (False, True):
            variant = transform_pose(points, mirror_x=mirror_x, swap_lr=swap_lr)
            signature = tuple(sorted((k, round(v[0], 4), round(v[1], 4)) for k, v in variant.items()))
            if signature in seen:
                continue
            seen.add(signature)
            variants.append(variant)
    return variants


def transform_pose(
    points: dict[str, tuple[float, float, float]],
    mirror_x: bool,
    swap_lr: bool,
) -> dict[str, tuple[float, float, float]]:
    transformed: dict[str, tuple[float, float, float]] = {}
    for name, point in points.items():
        target_name = swap_joint_name(name) if swap_lr else name
        x = -point[0] if mirror_x else point[0]
        transformed[target_name] = (x, point[1], point[2])
    return transformed


def swap_joint_name(name: str) -> str:
    if name in LEFT_RIGHT_PAIRS:
        return LEFT_RIGHT_PAIRS[name]
    for left, right in LEFT_RIGHT_PAIRS.items():
        if name == right:
            return left
    return name


def extract_pose_features(
    points: dict[str, tuple[float, float, float]]
) -> dict[str, float]:
    features: dict[str, float] = {}

    def p(name: str) -> np.ndarray | None:
        point = points.get(name)
        if point is None:
            return None
        return np.array(point[:2], dtype=np.float32)

    left_hip = p("left_hip")
    right_hip = p("right_hip")
    left_knee = p("left_knee")
    right_knee = p("right_knee")
    left_ankle = p("left_ankle")
    right_ankle = p("right_ankle")
    left_foot = p("left_foot")
    right_foot = p("right_foot")

    for side, foot, knee in (
        ("left", left_foot, left_knee),
        ("right", right_foot, right_knee),
    ):
        if foot is not None:
            features[f"{side}_foot_y"] = float(foot[1])
            features[f"{side}_foot_x"] = float(foot[0])
        if knee is not None:
            features[f"{side}_knee_y"] = float(knee[1])

    if left_foot is not None and right_foot is not None:
        features["foot_y_gap"] = float(left_foot[1] - right_foot[1])
        features["foot_x_span"] = float(abs(left_foot[0] - right_foot[0]))
    if left_knee is not None and right_knee is not None:
        features["knee_y_gap"] = float(left_knee[1] - right_knee[1])

    angle_specs = (
        ("left_knee_angle", left_hip, left_knee, left_ankle),
        ("right_knee_angle", right_hip, right_knee, right_ankle),
    )
    for name, hip, knee, ankle in angle_specs:
        if hip is not None and knee is not None and ankle is not None:
            features[name] = joint_angle(hip, knee, ankle)

    for name, knee, ankle in (
        ("left_shin_angle", left_knee, left_ankle),
        ("right_shin_angle", right_knee, right_ankle),
    ):
        if knee is not None and ankle is not None:
            delta = ankle - knee
            features[name] = float(np.arctan2(delta[1], delta[0]) / np.pi)

    return features


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-6:
        return 0.0
    cos_value = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.arccos(cos_value) / np.pi)


def compute_feature_similarity(current: dict[str, float], reference: dict[str, float]) -> float:
    total_weight = 0.0
    weighted_score = 0.0
    for name, ref_value in reference.items():
        if name not in current:
            continue
        weight = FEATURE_WEIGHTS.get(name, 1.0)
        tolerance = FEATURE_TOLERANCES.get(name, 0.25)
        diff = abs(current[name] - ref_value)
        score = max(0.0, 1.0 - diff / (tolerance * 2.5))
        total_weight += weight
        weighted_score += weight * score
    if total_weight == 0.0:
        return 0.0
    return weighted_score / total_weight


def create_kick_template() -> dict:
    """Create the reference template for '转身左蹬脚' (Turn Body Left Kick).

    Based on the reference image analysis. Key characteristics:
    - Left foot raised high (y much smaller than right foot)
    - Left knee bent and lifted
    - Left leg extending outward (x beyond knee)
    - Arms spread wide for balance
    - Body turned (shoulders more rotated than hips)
    """
    return {
        "name": "转身左蹬脚",
        "description": "Turn body, left foot kick — a fajin moment in Tai Chi",
        "landmark_ids": {
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_foot": 31,
            "right_foot": 32,
        },
        "keypoints": {
            "left_shoulder": {"x": 0.514, "y": 0.351, "z": -0.147, "tolerance": 0.10},
            "right_shoulder": {"x": 0.353, "y": 0.353, "z": -0.302, "tolerance": 0.10},
            "left_elbow": {"x": 0.616, "y": 0.358, "z": -0.129, "tolerance": 0.12},
            "right_elbow": {"x": 0.248, "y": 0.350, "z": -0.437, "tolerance": 0.12},
            "left_wrist": {"x": 0.706, "y": 0.332, "z": -0.296, "tolerance": 0.12},
            "right_wrist": {"x": 0.148, "y": 0.316, "z": -0.575, "tolerance": 0.12},
            "left_hip": {"x": 0.533, "y": 0.451, "z": 0.024, "tolerance": 0.08},
            "right_hip": {"x": 0.464, "y": 0.483, "z": -0.024, "tolerance": 0.08},
            "left_knee": {"x": 0.661, "y": 0.418, "z": -0.570, "tolerance": 0.10},
            "right_knee": {"x": 0.480, "y": 0.567, "z": 0.044, "tolerance": 0.08},
            "left_ankle": {"x": 0.813, "y": 0.363, "z": -0.891, "tolerance": 0.10},
            "right_ankle": {"x": 0.494, "y": 0.644, "z": 0.397, "tolerance": 0.08},
            "left_foot": {"x": 0.799, "y": 0.307, "z": -1.149, "tolerance": 0.12},
            "right_foot": {"x": 0.497, "y": 0.671, "z": 0.266, "tolerance": 0.08},
        },
        "match_weights": {
            # Heavily weight the kicking leg joints
            "left_ankle": 3.0,
            "left_foot": 3.0,
            "left_knee": 2.5,
            # Supporting leg important for stance identity
            "right_ankle": 1.5,
            "right_knee": 1.5,
            "right_foot": 1.0,
            # Arms for balance
            "left_wrist": 1.2,
            "right_wrist": 1.2,
            "left_elbow": 0.8,
            "right_elbow": 0.8,
            # Torso reference
            "left_hip": 0.6,
            "right_hip": 0.6,
            "left_shoulder": 0.5,
            "right_shoulder": 0.5,
        },
        # Effect configuration
        "effect": {
            "type": "directional_burst",
            "origin_joint": "left_foot",
            "direction": {"dx": 0.55, "dy": -0.65, "dz": -0.52},  # kick direction: right + up + forward
            "particle_count": 45,
            "burst_force": 14.0,
            "color": "warm_gold",  # warm gold for fajin moment
            "lifetime": 1.8,
            "trigger_cooldown": 0.8,  # seconds between repeated triggers
        },
    }
