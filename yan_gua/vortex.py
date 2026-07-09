"""双手涡旋的生命周期与相干性控制。

这一层把不稳定的“是否检测到手”转换为连续的物理场。它不关心粒子，
只负责回答每只手的涡旋此刻在哪里、保持得多完整、正在成形还是消散。
"""

from __future__ import annotations

import math

import numpy as np

from yan_gua.config import (
    VORTEX_BREAK_SPEED,
    VORTEX_ECHO_SECONDS,
    VORTEX_FORM_SECONDS,
    VORTEX_MAX_DRIFT_SPEED,
    VORTEX_SLOW_SPEED,
)


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return float(value >= edge1)
    x = min(max((value - edge0) / (edge1 - edge0), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _approach(current: float, target: float, dt: float, seconds: float) -> float:
    if seconds <= 0.0:
        return target
    blend = 1.0 - math.exp(-max(dt, 0.0) / seconds)
    return current + (target - current) * blend


class VortexController:
    """将两只手转换成两个身份稳定、可渐生渐灭的涡场。"""

    MAX_VORTICES = 2

    def __init__(self) -> None:
        self._states = [self._new_state(slot) for slot in range(self.MAX_VORTICES)]

    @staticmethod
    def _new_state(slot: int) -> dict:
        return {
            "slot": slot,
            "position": None,
            "velocity": np.zeros(2, dtype=np.float32),
            "strength": 0.0,
            "coherence": 0.0,
            "scatter": 0.0,
            "release": 1.0,
            "maturity": 0.0,
            "aperture": 0.0,
            "spin": 1.0 if slot == 0 else -1.0,
            "missing_time": math.inf,
            "observed": False,
            "active": False,
            "phase": "dormant",
        }

    def reset(self) -> None:
        self._states = [self._new_state(slot) for slot in range(self.MAX_VORTICES)]

    def update(self, hand_states: list[dict], dt: float) -> list[dict]:
        """推进生命周期，返回可直接交给粒子物理层的两个涡场。"""
        dt = min(max(float(dt), 0.0), 0.1)
        for slot, field in enumerate(self._states):
            hand = hand_states[slot] if slot < len(hand_states) else {}
            observed = bool(hand.get("observed")) and hand.get("hand_world_pos") is not None
            field["observed"] = observed

            if observed:
                self._update_observed(field, hand, dt)
            else:
                self._update_echo(field, dt)

            field["active"] = field["position"] is not None and field["strength"] > 0.012

        return self.snapshot()

    def snapshot(self) -> list[dict]:
        """返回不暴露内部可变数组的轻量快照。"""
        result = []
        for field in self._states:
            item = field.copy()
            if field["position"] is not None:
                item["position"] = field["position"].copy()
            item["velocity"] = field["velocity"].copy()
            result.append(item)
        return result

    def _update_observed(self, field: dict, hand: dict, dt: float) -> None:
        target = np.asarray(hand["hand_world_pos"], dtype=np.float32)
        was_dormant = field["position"] is None or not field["active"]

        if was_dormant:
            field["position"] = target.copy()
            field["maturity"] = 0.0
        else:
            position_blend = 1.0 - math.exp(-dt / 0.065)
            field["position"] += (target - field["position"]) * position_blend

        raw_velocity = np.asarray(hand.get("hand_velocity", (0.0, 0.0)), dtype=np.float32)
        speed = float(np.linalg.norm(raw_velocity))
        if speed > VORTEX_MAX_DRIFT_SPEED:
            raw_velocity *= VORTEX_MAX_DRIFT_SPEED / max(speed, 0.001)
        velocity_blend = 1.0 - math.exp(-dt / 0.12)
        field["velocity"] += (raw_velocity - field["velocity"]) * velocity_blend

        speed = float(hand.get("speed", speed))
        breakup = _smoothstep(VORTEX_SLOW_SPEED, VORTEX_BREAK_SPEED, speed)
        target_coherence = 1.0 - breakup

        field["coherence"] = _approach(field["coherence"], target_coherence, dt, 0.18)
        field["scatter"] = _approach(field["scatter"], breakup, dt, 0.11)
        field["strength"] = _approach(field["strength"], 1.0, dt, VORTEX_FORM_SECONDS * 0.45)
        field["maturity"] = min(1.0, field["maturity"] + dt / VORTEX_FORM_SECONDS)
        field["release"] = 0.0
        field["missing_time"] = 0.0

        z_velocity = float(hand.get("z_velocity", 0.0))
        target_aperture = min(max(z_velocity / 260.0, -0.32), 0.32)
        field["aperture"] = _approach(field["aperture"], target_aperture, dt, 0.2)

        if field["maturity"] < 0.98:
            field["phase"] = "forming"
        elif field["scatter"] > 0.55:
            field["phase"] = "dispersing"
        else:
            field["phase"] = "holding"

    def _update_echo(self, field: dict, dt: float) -> None:
        if field["position"] is None:
            return

        if math.isinf(field["missing_time"]):
            field["missing_time"] = 0.0
        field["missing_time"] += dt
        field["position"] += field["velocity"] * dt * 0.22
        field["velocity"] *= math.exp(-dt / 0.65)
        field["strength"] *= math.exp(-dt / VORTEX_ECHO_SECONDS)
        field["coherence"] = _approach(field["coherence"], 0.58, dt, 0.85)
        field["scatter"] *= math.exp(-dt / 0.28)
        field["aperture"] *= math.exp(-dt / 0.45)
        field["release"] = min(field["missing_time"] / VORTEX_ECHO_SECONDS, 1.0)
        field["maturity"] *= math.exp(-dt / (VORTEX_ECHO_SECONDS * 1.4))
        field["phase"] = "echo" if field["strength"] > 0.012 else "dormant"

        if field["phase"] == "dormant":
            field["strength"] = 0.0
            field["position"] = None
