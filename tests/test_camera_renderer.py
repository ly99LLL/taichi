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
