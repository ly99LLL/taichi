"""OpenCV 离线渲染器测试。"""

import numpy as np

from yan_gua.offline_renderer import (
    ParticleFrameRenderer,
    draw_vortex_marks,
    field_color_bgr,
)


class FakeParticles:
    def __init__(self):
        self.alpha = np.array([6.0, 32.0, 80.0], dtype=np.float32)
        self.ink_level = np.array([0, 2, 5], dtype=np.int32)
        self.px = np.array([8.0, 16.0, 24.0], dtype=np.float32)
        self.py = np.array([8.0, 16.0, 24.0], dtype=np.float32)
        self.radius = np.array([1.0, 1.5, 2.0], dtype=np.float32)

    @staticmethod
    def tint_color(color, _px, _py, _vortices):
        return color


def vortex(phase="holding", *, active=True, observed=True):
    return {
        "slot": 0,
        "active": active,
        "observed": observed,
        "position": np.array([32.0, 32.0], dtype=np.float32),
        "phase": phase,
        "strength": 1.0,
        "coherence": 1.0 if phase == "holding" else 0.0,
        "scatter": 1.0 if phase == "dispersing" else 0.0,
        "release": 0.5 if phase == "echo" else 0.0,
        "aperture": 0.0,
    }


def test_particle_frame_renderer_returns_independent_bgr_frame():
    renderer = ParticleFrameRenderer(64, 48, 30.0)
    frame = renderer.render(
        FakeParticles(),
        [],
        phase_label="FORMING",
        synthetic=True,
    )

    assert frame.shape == (48, 64, 3)
    assert frame.dtype == np.uint8
    assert frame is not renderer.trail
    assert np.any(frame != frame[0, 0])


def test_vortex_marks_and_phase_colours_cover_all_states():
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    draw_vortex_marks(canvas, [vortex()])

    assert np.any(canvas)
    assert field_color_bgr(vortex("holding")) != field_color_bgr(vortex("dispersing"))
    assert field_color_bgr(vortex("echo")) != field_color_bgr(vortex("holding"))
