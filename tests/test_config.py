"""测试 config 模块 — 验证常量值正确。"""

import yan_gua.config as cfg


def test_colors():
    """色彩常量应为有效 RGB 元组。"""
    assert cfg.BG_R == 16
    assert cfg.BG_G == 12
    assert cfg.BG_B == 8
    assert len(cfg.INK_COLORS) == 8
    assert cfg.NUM_INK_LEVELS == 8
    assert cfg.WARM_ACCENT == (200, 155, 90)
    assert cfg.WARM_LIGHT == (240, 210, 150)


def test_particle_constants():
    """粒子系统常量应在合理范围内。"""
    assert cfg.PARTICLE_COUNT == 6000
    assert cfg.INFLUENCE_RADIUS == 240
    assert cfg.MAX_SPEED == 800
    assert cfg.CURVATURE_REF == 400
    assert 0 < cfg.SMOOTH_ALPHA < 1
    assert cfg.BASE_DAMPING < 1.0
    assert cfg.CENTER_GRAVITY > 0.00003
    assert cfg.HAND_FORCE_MULTIPLIER > 1.0
    assert cfg.TRAIL_ALPHA > 0


def test_window_defaults():
    """窗口默认值应为正数。"""
    assert cfg.WINDOW_W == 1280
    assert cfg.WINDOW_H == 720
    assert cfg.CAM_W == 280
    assert cfg.CAM_H == 210
    assert cfg.CAM_MARGIN == 20


def test_camera_constants():
    """摄像头参数应在合理范围。"""
    assert cfg.CAMERA_WIDTH == 1280
    assert cfg.CAMERA_HEIGHT == 720
    assert cfg.CAMERA_FPS == 30
    assert cfg.HANDS_MODEL_COMPLEXITY in (0, 1, 2)
    assert cfg.POSE_MODEL_COMPLEXITY in (0, 1, 2)
    assert 0 < cfg.HANDS_DETECTION_CONFIDENCE <= 1
    assert 0 < cfg.POSE_WRIST_VISIBILITY <= 1
