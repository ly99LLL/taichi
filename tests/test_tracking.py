"""HandTracker 的无摄像头行为测试。"""

from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from yan_gua.tracking import HandTracker


def _tracker(hand_result, pose_landmarks=None):
    tracker = HandTracker.__new__(HandTracker)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    tracker.cap = Mock()
    tracker.cap.read.return_value = (True, frame)
    tracker._is_video = True
    tracker._mirror_video = False
    tracker.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    tracker.hands = Mock()
    tracker.hands.process.return_value = hand_result
    tracker.pose = Mock()
    tracker.pose.process.return_value = SimpleNamespace(pose_landmarks=pose_landmarks)
    return tracker


def test_read_preserves_real_z_coordinate():
    landmarks = [SimpleNamespace(x=i / 20, y=i / 40, z=-i / 100) for i in range(21)]
    hand_result = SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=landmarks)])
    tracker = _tracker(hand_result)

    frame, hands, pose = tracker.read()

    assert frame.shape == (24, 32, 3)
    assert pose is None
    assert len(hands[0]["landmarks"]) == 21
    assert hands[0]["palm_center"]["z"] == (-0.0 - 0.09) / 2


def test_pose_wrist_is_used_as_fallback():
    hidden = SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.0)
    landmarks = [hidden for _ in range(33)]
    landmarks[15] = SimpleNamespace(x=0.3, y=0.4, z=-0.2, visibility=0.9)
    pose = SimpleNamespace(landmark=landmarks)
    tracker = _tracker(SimpleNamespace(multi_hand_landmarks=None), pose)

    _, hands, returned_pose = tracker.read()

    assert returned_pose is pose
    assert hands == [
        {
            "palm_center": {"x": 0.3, "y": 0.4, "z": -0.2},
            "landmarks": [],
        }
    ]


def test_failed_read_returns_empty_result():
    tracker = HandTracker.__new__(HandTracker)
    tracker.cap = Mock()
    tracker.cap.read.return_value = (False, None)

    assert tracker.read() == (None, None, None)


def test_release_closes_all_resources():
    tracker = HandTracker.__new__(HandTracker)
    tracker.cap = Mock()
    tracker.hands = Mock()
    tracker.pose = Mock()

    tracker.release()

    tracker.cap.release.assert_called_once()
    tracker.hands.close.assert_called_once()
    tracker.pose.close.assert_called_once()
