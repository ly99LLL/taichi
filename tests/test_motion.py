"""测试 MotionAnalyzer — 验证运动特征提取。"""

import numpy as np

from yan_gua.motion import MotionAnalyzer


def test_init():
    """MotionAnalyzer 初始化应创建 2 个手部状态。"""
    ma = MotionAnalyzer(1280, 720)
    assert len(ma.states) == 2
    for st in ma.states:
        assert st['speed'] == 0.0
        assert st['curvature'] == 0.0
        assert st['z_velocity'] == 0.0
        assert st['hand_detected'] is False


def test_no_hands():
    """无手部数据时特征保持衰减。"""
    ma = MotionAnalyzer(1280, 720)
    results = ma.process(None, 0.0)
    assert len(results) == 2
    for st in results:
        assert st['hand_detected'] is False


def test_single_hand_detection():
    """单手数据应被正确检测。"""
    ma = MotionAnalyzer(1280, 720)
    hand_data = [{
        'palm_center': {'x': 0.5, 'y': 0.5, 'z': 0.0},
        'landmarks': [],
    }]
    results = ma.process(hand_data, 0.0)
    assert results[0]['hand_detected'] is True
    assert results[0]['hand_world_pos'] is not None
    # 第二只手应未检测到
    assert results[1]['hand_detected'] is False


def test_presence_hysteresis():
    """短暂消失不应立即丢失追踪 (presence_counter 迟滞)。"""
    ma = MotionAnalyzer(1280, 720)
    hand_data = [{
        'palm_center': {'x': 0.5, 'y': 0.5, 'z': 0.0},
        'landmarks': [],
    }]

    # 先注册几帧
    for t in range(3):
        ma.process(hand_data, float(t) * 0.016)

    # 一帧无手
    results = ma.process(None, 0.05)
    # 应该仍然 detected (presence_counter 从 3 降到 2, still >= 1)
    assert results[0]['hand_detected'] is True


def test_features_with_movement():
    """手部移动时应产生速度信号。"""
    ma = MotionAnalyzer(1280, 720)
    # 模拟手从左到右移动
    for i in range(10):
        x = 0.3 + i * 0.02  # 每帧向右移动 0.02 (归一化坐标)
        hand_data = [{
            'palm_center': {'x': x, 'y': 0.5, 'z': 0.0},
            'landmarks': [],
        }]
        results = ma.process(hand_data, float(i) * 0.016)

    # 最后应该有非零速度
    assert results[0]['speed'] > 0, f"Expected speed > 0, got {results[0]['speed']}"
