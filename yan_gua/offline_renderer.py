"""OpenCV 离线帧渲染器。

该模块只负责把 ``CloudParticles`` 与 ``VortexController`` 的状态绘制为画面，
不读取摄像头，也不持有跟踪器。真实视频和合成演示因此可以复用同一条视觉管线。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from yan_gua.config import (
    BG_B,
    BG_G,
    BG_R,
    COHERENT_COLOR,
    ECHO_COLOR,
    INK_COLORS,
    NUM_INK_LEVELS,
    SCATTER_COLOR,
    TRAIL_ALPHA,
    UI_MUTED,
    UI_PRIMARY,
    VORTEX_ORBIT_RADIUS,
)

if TYPE_CHECKING:
    from yan_gua.physics import CloudParticles

BG_BGR = (BG_B, BG_G, BG_R)

_ALPHA_BANDS = (
    (1, 8, 0.028),
    (8, 20, 0.070),
    (20, 45, 0.150),
    (45, 256, 0.270),
)


def _as_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]


def field_color_bgr(field: dict) -> tuple[int, int, int]:
    """返回与涡场阶段一致的 OpenCV BGR 颜色。"""
    if field["phase"] == "echo":
        color = ECHO_COLOR
    elif field["scatter"] > field["coherence"]:
        color = SCATTER_COLOR
    else:
        color = COHERENT_COLOR
    return _as_bgr(color)


def draw_particles(
    canvas: np.ndarray,
    particles: CloudParticles,
    vortices: list[dict],
) -> None:
    """按亮度分层混合，近似 py5 的逐粒子透明度。"""
    order = np.argsort(particles.alpha)
    for low, high, blend_weight in _ALPHA_BANDS:
        layer = np.full_like(canvas, BG_BGR)
        drew = False
        for index in order:
            particle_alpha = particles.alpha[index]
            if particle_alpha < low or particle_alpha >= high:
                continue

            ink = min(int(particles.ink_level[index]), NUM_INK_LEVELS - 1)
            rgb = particles.tint_color(
                INK_COLORS[ink],
                float(particles.px[index]),
                float(particles.py[index]),
                vortices,
            )
            center = (int(particles.px[index]), int(particles.py[index]))
            radius = max(1, int(round(particles.radius[index])))
            cv2.circle(layer, center, radius, _as_bgr(rgb), -1, cv2.LINE_AA)
            drew = True

        if drew:
            cv2.addWeighted(
                layer,
                blend_weight,
                canvas,
                1.0 - blend_weight,
                0,
                canvas,
            )


def draw_vortex_marks(canvas: np.ndarray, vortices: list[dict]) -> None:
    """绘制与实时界面一致的低存在感轨道缺口。"""
    for field in vortices:
        if not field["active"] or field["position"] is None:
            continue

        x, y = (int(value) for value in field["position"])
        radius = int(VORTEX_ORBIT_RADIUS * (1.0 + field["release"] * 0.9 + field["aperture"]))
        angle = int(field["slot"] * 180 + field["strength"] * 35)
        color = field_color_bgr(field)
        cv2.ellipse(canvas, (x, y), (radius, radius), 0, angle, angle + 42, color, 1)
        cv2.ellipse(
            canvas,
            (x, y),
            (radius, radius),
            0,
            angle + 180,
            angle + 222,
            color,
            1,
        )
        if field["observed"]:
            cv2.circle(canvas, (x, y), 1, color, -1, cv2.LINE_AA)


def draw_status(
    canvas: np.ndarray,
    phase_label: str,
    *,
    synthetic: bool = False,
) -> None:
    """绘制不会进入拖尾缓冲的克制状态标识。"""
    primary = _as_bgr(UI_PRIMARY)
    muted = _as_bgr(UI_MUTED)
    cv2.putText(
        canvas,
        "YAN GUA / TWIN VORTEX FIELD",
        (24, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        primary,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        phase_label,
        (24, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        muted,
        1,
        cv2.LINE_AA,
    )
    if synthetic:
        text = "SYNTHETIC MOTION / NO CAMERA INPUT"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)[0]
        cv2.putText(
            canvas,
            text,
            (canvas.shape[1] - size[0] - 24, canvas.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            muted,
            1,
            cv2.LINE_AA,
        )


class ParticleFrameRenderer:
    """维护拖尾缓冲并输出一帧不含摄像头画面的粒子场。"""

    def __init__(self, width: int, height: int, fps: float):
        self.width = width
        self.height = height
        decay_per_second = (1.0 - TRAIL_ALPHA / 255.0) ** 60.0
        self.trail_alpha = 1.0 - decay_per_second ** (1.0 / max(fps, 1.0))
        self.trail = np.full((height, width, 3), BG_BGR, dtype=np.uint8)
        self.dark = np.full_like(self.trail, BG_BGR)

    def render(
        self,
        particles: CloudParticles,
        vortices: list[dict],
        *,
        phase_label: str,
        synthetic: bool = False,
    ) -> np.ndarray:
        """推进拖尾并返回可继续叠加 UI 的独立画面。"""
        cv2.addWeighted(
            self.trail,
            1.0 - self.trail_alpha,
            self.dark,
            self.trail_alpha,
            0,
            self.trail,
        )
        draw_particles(self.trail, particles, vortices)
        draw_vortex_marks(self.trail, vortices)

        frame = self.trail.copy()
        draw_status(frame, phase_label, synthetic=synthetic)
        return frame
