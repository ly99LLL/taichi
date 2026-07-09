"""测试 MotionAnalyzer — 验证运动特征提取。"""

from yan_gua.motion import MotionAnalyzer


def test_init():
    """MotionAnalyzer 初始化应创建 2 个手部状态。"""
    ma = MotionAnalyzer(1280, 720)
    assert len(ma.states) == 2
    for st in ma.states:
        assert st["speed"] == 0.0
        assert st["curvature"] == 0.0
        assert st["z_velocity"] == 0.0
        assert st["hand_detected"] is False
        assert st["observed"] is False


def test_no_hands():
    """无手部数据时特征保持衰减。"""
    ma = MotionAnalyzer(1280, 720)
    results = ma.process(None, 0.0)
    assert len(results) == 2
    for st in results:
        assert st["hand_detected"] is False


def test_single_hand_detection():
    """单手数据应被正确检测。"""
    ma = MotionAnalyzer(1280, 720)
    hand_data = [
        {
            "palm_center": {"x": 0.5, "y": 0.5, "z": 0.0},
            "landmarks": [],
        }
    ]
    results = ma.process(hand_data, 0.0)
    assert results[0]["hand_detected"] is True
    assert results[0]["observed"] is True
    assert results[0]["hand_world_pos"] is not None
    # 第二只手应未检测到
    assert results[1]["hand_detected"] is False


def test_presence_hysteresis():
    """短暂消失不应立即丢失追踪 (presence_counter 迟滞)。"""
    ma = MotionAnalyzer(1280, 720)
    hand_data = [
        {
            "palm_center": {"x": 0.5, "y": 0.5, "z": 0.0},
            "landmarks": [],
        }
    ]

    # 先注册几帧
    for t in range(3):
        ma.process(hand_data, float(t) * 0.016)

    # 一帧无手
    results = ma.process(None, 0.05)
    # 应该仍然 detected (presence_counter 从 3 降到 2, still >= 1)
    assert results[0]["hand_detected"] is True
    assert results[0]["observed"] is False


def test_features_with_movement():
    """手部移动时应产生速度信号。"""
    ma = MotionAnalyzer(1280, 720)
    # 模拟手从左到右移动
    for i in range(10):
        x = 0.3 + i * 0.02  # 每帧向右移动 0.02 (归一化坐标)
        hand_data = [
            {
                "palm_center": {"x": x, "y": 0.5, "z": 0.0},
                "landmarks": [],
            }
        ]
        results = ma.process(hand_data, float(i) * 0.016)

    # 最后应该有非零速度
    assert results[0]["speed"] > 0, f"Expected speed > 0, got {results[0]['speed']}"


def test_screen_y_movement_is_not_depth():
    """屏幕纵向移动不应被误认为纵深移动。"""
    ma = MotionAnalyzer(1280, 720)
    for i in range(8):
        hand_data = [
            {
                "palm_center": {"x": 0.5, "y": 0.2 + i * 0.05, "z": 0.0},
                "landmarks": [],
            }
        ]
        results = ma.process(hand_data, i * 0.02)

    assert results[0]["speed"] > 0
    assert results[0]["z_velocity"] == 0.0


def test_moving_toward_camera_has_positive_depth_velocity():
    """MediaPipe z 减小时表示前推，呼吸速度应为正。"""
    ma = MotionAnalyzer(1280, 720)
    for i in range(8):
        hand_data = [
            {
                "palm_center": {"x": 0.5, "y": 0.5, "z": -i * 0.01},
                "landmarks": [],
            }
        ]
        results = ma.process(hand_data, i * 0.02)

    assert results[0]["speed"] == 0.0
    assert results[0]["z_velocity"] > 0


def test_moving_away_from_camera_has_negative_depth_velocity():
    """MediaPipe z 增大时表示回拉，呼吸速度应为负。"""
    ma = MotionAnalyzer(1280, 720)
    for i in range(8):
        hand_data = [
            {
                "palm_center": {"x": 0.5, "y": 0.5, "z": i * 0.01},
                "landmarks": [],
            }
        ]
        results = ma.process(hand_data, i * 0.02)

    assert results[0]["z_velocity"] < 0


def test_handedness_keeps_slots_stable_when_detector_order_flips():
    """检测列表换序时，左右手身份不能跟着互换。"""
    ma = MotionAnalyzer(1000, 500)
    left = {
        "id_hint": "Left",
        "palm_center": {"x": 0.2, "y": 0.5, "z": 0.0},
        "landmarks": [],
    }
    right = {
        "id_hint": "Right",
        "palm_center": {"x": 0.8, "y": 0.5, "z": 0.0},
        "landmarks": [],
    }

    ma.process([left, right], 0.0)
    states = ma.process([right, left], 0.016)

    assert states[0]["id_hint"] == "left"
    assert states[0]["hand_world_pos"][0] == 200
    assert states[1]["id_hint"] == "right"
    assert states[1]["hand_world_pos"][0] == 800


def test_reacquisition_clears_stale_velocity_history():
    """长时间丢失后在远处重现，不应计算出跨越空白期的巨幅手速。"""
    ma = MotionAnalyzer(1000, 500)
    hand = {
        "id_hint": "Left",
        "palm_center": {"x": 0.2, "y": 0.5, "z": 0.0},
        "landmarks": [],
    }
    ma.process([hand], 0.0)
    ma.process(None, 0.3)
    hand["palm_center"]["x"] = 0.9
    states = ma.process([hand], 0.6)

    assert states[0]["newly_acquired"] is True
    assert states[0]["speed"] == 0.0
