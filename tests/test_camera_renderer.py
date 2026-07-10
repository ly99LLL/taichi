"""CameraRenderer 的纯图像管线测试。"""

import numpy as np

from yan_gua.camera_renderer import CameraRenderer


def test_none_frame_returns_empty_result():
    renderer = CameraRenderer()
    assert renderer.process_bgr(None, None) == (None, 0, 0)


def test_process_bgr_resizes_and_positions_preview():
    renderer = CameraRenderer(cam_w=80, cam_h=60, margin=10)
    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    landmarks = [{"x": i / 20, "y": 1.0 - i / 20, "z": 0.0} for i in range(21)]

    output, x, y = renderer.process_bgr(
        frame,
        [{"palm_center": landmarks[0], "landmarks": landmarks}],
        canvas_w=320,
        canvas_h=180,
    )

    assert output.shape == (60, 80, 3)
    assert output.dtype == np.uint8
    assert (x, y) == (230, 110)
    assert not np.array_equal(output, frame[:60, :80])


def test_landmark_pixels_are_clamped():
    pixels = CameraRenderer._landmarks_to_pixels(
        [{"x": -1, "y": 2}, {"x": 0.5, "y": 0.25}],
        100,
        80,
    )
    assert pixels == [(0, 79), (50, 20)]


def test_color_grade_preserves_original_hue_and_brightness():
    """原彩模式只能轻调，不得把彩色画面变成冷灰。"""
    frame = np.empty((20, 20, 3), dtype=np.uint8)
    frame[:] = (32, 96, 220)

    output = CameraRenderer._preserve_color(frame)
    original = frame[10, 10].astype(int)
    graded = output[10, 10].astype(int)

    assert int(np.argmax(graded)) == int(np.argmax(original))
    assert np.max(np.abs(graded - original)) < 12


def test_pose_fallback_palm_still_gets_recognition_orbit():
    """只有腕部降级点时也应显示掌心识别环。"""
    renderer = CameraRenderer(cam_w=100, cam_h=80)
    frame = np.full((80, 100, 3), (60, 110, 170), dtype=np.uint8)
    hand = {
        "id_hint": "Right",
        "palm_center": {"x": 0.5, "y": 0.5, "z": 0.0},
        "landmarks": [],
    }

    baseline = renderer._vignette(renderer._preserve_color(frame))
    with_effect = renderer._draw_hand_effect(baseline.copy(), [hand])

    assert not np.array_equal(with_effect[28:53, 38:63], baseline[28:53, 38:63])


def test_predicted_hand_draws_motion_memory_orbit():
    renderer = CameraRenderer(cam_w=100, cam_h=80)
    frame = np.full((80, 100, 3), (50, 80, 120), dtype=np.uint8)
    hand = {
        "id_hint": "Left",
        "palm_center": {"x": 0.5, "y": 0.5, "z": 0.0},
        "landmarks": [],
        "predicted": True,
        "velocity": {"x": 0.8, "y": 0.0},
    }

    baseline = renderer._vignette(renderer._preserve_color(frame))
    with_effect = renderer._draw_hand_effect(baseline.copy(), [hand])

    assert not np.array_equal(with_effect[36:45, 18:52], baseline[36:45, 18:52])
