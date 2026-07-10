"""配置边界测试。"""

import yan_gua.config as cfg


def test_cinematic_palette():
    assert (cfg.BG_R, cfg.BG_G, cfg.BG_B) == (3, 3, 4)
    assert len(cfg.INK_COLORS) == 8
    assert cfg.NUM_INK_LEVELS == 8
    for color in cfg.INK_COLORS:
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


def test_vortex_constants():
    assert cfg.PARTICLE_COUNT == 7200
    assert cfg.PARTICLE_SIZE_MIN >= 1.0
    assert cfg.PARTICLE_SIZE_MAX > 3.0
    assert cfg.VORTEX_ORBIT_RADIUS < cfg.VORTEX_INFLUENCE_RADIUS
    assert cfg.VORTEX_SLOW_SPEED < cfg.VORTEX_BREAK_SPEED
    assert 0 < cfg.VORTEX_FORM_SECONDS <= 0.25
    assert cfg.VORTEX_ECHO_SECONDS > cfg.VORTEX_FORM_SECONDS
    assert cfg.VORTEX_HAND_CARRY > 0.7
    assert cfg.VORTEX_STOP_SPLASH_SPEED > 0
    assert 0 < cfg.FAST_LOST_PREDICT_SECONDS < 0.3
    assert cfg.FAST_LOST_PREDICT_SPEED > cfg.VORTEX_SLOW_SPEED
    assert 0 < cfg.SHORT_LOST_MAINTAIN_SECONDS < cfg.FAST_LOST_PREDICT_SECONDS
    assert cfg.INFLUENCE_RADIUS == cfg.VORTEX_INFLUENCE_RADIUS
    assert 0 < cfg.BASE_DAMPING < 1
    assert cfg.TRAIL_ALPHA > 0


def test_window_defaults():
    assert cfg.WINDOW_W == 1280
    assert cfg.WINDOW_H == 720
    assert cfg.CAM_W == 280
    assert cfg.CAM_H == 210
    assert cfg.CAM_MARGIN == 20


def test_camera_constants():
    assert cfg.CAMERA_WIDTH == 1280
    assert cfg.CAMERA_HEIGHT == 720
    assert cfg.CAMERA_FPS == 30
    assert cfg.HANDS_MODEL_COMPLEXITY in (0, 1, 2)
    assert cfg.POSE_MODEL_COMPLEXITY in (0, 1, 2)
    assert 0 < cfg.HANDS_DETECTION_CONFIDENCE <= 1
    assert 0 < cfg.POSE_WRIST_VISIBILITY <= 1
