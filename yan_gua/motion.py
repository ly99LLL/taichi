"""身份稳定的双手运动分析。

检测器返回的列表顺序并不可靠。本模块先依据 handedness 和位置预测把观测
分配到两个持久槽位，再分别计算速度、曲率与纵深速度。
"""

from __future__ import annotations

from collections import deque
from itertools import permutations

import numpy as np

from yan_gua.config import (
    CURVATURE_REF,
    HISTORY_SIZE,
    REACQUIRE_GAP_SECONDS,
    SMOOTH_ALPHA,
)


class MotionAnalyzer:
    """追踪两只手的稳定身份，并提取连续运动特征。"""

    MAX_HANDS = 2

    def __init__(self, win_w: int, win_h: int):
        self.win_w = win_w
        self.win_h = win_h
        self.states = [self._new_state(slot) for slot in range(self.MAX_HANDS)]

    def process(self, hands_data, timestamp: float) -> list[dict]:
        """处理一帧；返回固定为左/右两个槽位的状态列表。"""
        observations = self._make_observations(hands_data or [])
        assignments = self._associate(observations)

        for slot, state in enumerate(self.states):
            obs = assignments.get(slot)
            state["newly_acquired"] = False
            state["observed"] = obs is not None

            if obs is not None:
                gap = (
                    float(timestamp) - state["last_seen_time"]
                    if state["last_seen_time"] is not None
                    else float("inf")
                )
                if gap > REACQUIRE_GAP_SECONDS:
                    state["history"].clear()
                    state["newly_acquired"] = True
                    state["speed"] *= 0.25
                    state["curvature"] *= 0.25
                    state["z_velocity"] *= 0.25
                    state["hand_velocity"] *= 0.25

                state["id_hint"] = obs["id_hint"] or state["id_hint"]
                state["hand_world_pos"] = obs["position"].copy()
                state["history"].append((obs["position"].copy(), obs["depth"], float(timestamp)))
                state["last_seen_time"] = float(timestamp)
                state["presence_counter"] = min(state["presence_counter"] + 2, 8)
                self._compute_features(state)
            else:
                state["presence_counter"] = max(state["presence_counter"] - 1, 0)
                state["speed"] *= 0.86
                state["curvature"] *= 0.82
                state["z_velocity"] *= 0.82
                state["hand_velocity"] *= 0.86

            # hand_detected 是供 UI 使用的短迟滞；物理生命周期使用 observed。
            state["hand_detected"] = state["presence_counter"] > 0

        return self.states

    def _make_observations(self, hands_data: list[dict]) -> list[dict]:
        observations = []
        for hand in hands_data[: self.MAX_HANDS]:
            palm = hand["palm_center"]
            observations.append(
                {
                    "position": np.array(
                        [palm["x"] * self.win_w, palm["y"] * self.win_h],
                        dtype=np.float32,
                    ),
                    "depth": -float(palm.get("z", 0.0)) * self.win_w,
                    "id_hint": self._normalise_hint(hand.get("id_hint")),
                }
            )
        return observations

    @staticmethod
    def _normalise_hint(value) -> str | None:
        if not value:
            return None
        label = str(value).strip().lower()
        if label.startswith("left"):
            return "left"
        if label.startswith("right"):
            return "right"
        return None

    def _associate(self, observations: list[dict]) -> dict[int, dict]:
        """以 handedness 优先、预测距离其次，求最多 2×2 的最小匹配。"""
        assignments: dict[int, dict] = {}
        used_observations: set[int] = set()

        # 明确标签直接落到固定槽位：left=0, right=1。
        for obs_index, obs in enumerate(observations):
            hint = obs["id_hint"]
            slot = 0 if hint == "left" else 1 if hint == "right" else None
            if slot is not None and slot not in assignments:
                assignments[slot] = obs
                used_observations.add(obs_index)

        free_slots = [slot for slot in range(self.MAX_HANDS) if slot not in assignments]
        free_observations = [
            index for index in range(len(observations)) if index not in used_observations
        ]
        if not free_slots or not free_observations:
            return assignments

        count = min(len(free_slots), len(free_observations))
        best_cost = float("inf")
        best_pairs = None
        for slot_order in permutations(free_slots, count):
            for obs_order in permutations(free_observations, count):
                cost = 0.0
                pairs = []
                for slot, obs_index in zip(slot_order, obs_order, strict=True):
                    state = self.states[slot]
                    current = state["hand_world_pos"]
                    if current is None:
                        # 已建立的轨迹优先于空槽位；两槽皆空时仍由槽号稳定破同分。
                        pair_cost = 1_000_000.0 + slot * 0.01
                    else:
                        predicted = current + state["hand_velocity"] * 0.016
                        pair_cost = float(
                            np.linalg.norm(observations[obs_index]["position"] - predicted)
                        )
                    cost += pair_cost
                    pairs.append((slot, obs_index))
                if cost < best_cost:
                    best_cost = cost
                    best_pairs = pairs

        for slot, obs_index in best_pairs or []:
            assignments[slot] = observations[obs_index]
        return assignments

    @staticmethod
    def _new_state(slot: int) -> dict:
        return {
            "slot": slot,
            "id_hint": "left" if slot == 0 else "right",
            "history": deque(maxlen=HISTORY_SIZE),
            "speed": 0.0,
            "curvature": 0.0,
            "z_velocity": 0.0,
            "hand_velocity": np.zeros(2, dtype=np.float32),
            "hand_world_pos": None,
            "last_direction": np.zeros(2, dtype=np.float32),
            "last_seen_time": None,
            "observed": False,
            "newly_acquired": False,
            "hand_detected": False,
            "presence_counter": 0,
        }

    @staticmethod
    def _compute_features(state: dict) -> None:
        history = list(state["history"])
        if len(history) < 2:
            return

        total_velocity = np.zeros(2, dtype=np.float32)
        total_weight = 0.0
        newest_index = len(history) - 1
        for offset in range(min(len(history) - 1, 7)):
            current = history[newest_index - offset]
            previous = history[newest_index - offset - 1]
            dt = current[2] - previous[2]
            if 0.001 < dt <= REACQUIRE_GAP_SECONDS:
                weight = 1.0 / (1.0 + offset * 0.35)
                total_velocity += ((current[0] - previous[0]) / dt) * weight
                total_weight += weight

        if total_weight > 0.0:
            raw_velocity = total_velocity / total_weight
            raw_speed = float(np.linalg.norm(raw_velocity))
            state["speed"] += (raw_speed - state["speed"]) * SMOOTH_ALPHA
            state["hand_velocity"] += (raw_velocity - state["hand_velocity"]) * SMOOTH_ALPHA

        if state["speed"] > 1.5:
            direction = state["hand_velocity"] / (state["speed"] + 0.0001)
            previous_direction = state["last_direction"]
            if float(np.linalg.norm(previous_direction)) > 0.1:
                dot = float(np.clip(np.dot(direction, previous_direction), -1.0, 1.0))
                angle = float(np.arccos(dot))
                raw_curvature = angle * min(state["speed"] / CURVATURE_REF, 1.0)
                state["curvature"] += (raw_curvature - state["curvature"]) * SMOOTH_ALPHA
            state["last_direction"] = direction
        else:
            state["curvature"] *= 0.82

        newest = history[-1]
        oldest = history[max(0, len(history) - 6)]
        z_dt = newest[2] - oldest[2]
        if z_dt > 0.01:
            raw_z_velocity = (newest[1] - oldest[1]) / z_dt
            state["z_velocity"] += (raw_z_velocity - state["z_velocity"]) * SMOOTH_ALPHA
