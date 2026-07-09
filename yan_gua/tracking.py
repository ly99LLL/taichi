"""摄像头 + MediaPipe 手部/骨架检测.

双轨检测策略:
1. MediaPipe Hands (21 点手指) — 优先, 手部够大时使用。
2. MediaPipe Pose (33 点全身骨架) — 降级, 手太小/远时用腕关节补位。

CLAHE 增强用于改善低对比度画面的关键点检测条件。
"""

import cv2
import mediapipe as mp

from yan_gua.config import (
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    HANDS_DETECTION_CONFIDENCE,
    HANDS_MODEL_COMPLEXITY,
    HANDS_TRACKING_CONFIDENCE,
    POSE_DETECTION_CONFIDENCE,
    POSE_MODEL_COMPLEXITY,
    POSE_TRACKING_CONFIDENCE,
    POSE_WRIST_VISIBILITY,
)


class HandTracker:
    """摄像头采集 + CLAHE 增强 + MediaPipe Hands/Pose 推理。

    Attributes:
        cap: OpenCV 摄像头捕获对象。
        clahe: CLAHE 对比度增强器。
        hands: MediaPipe Hands 模型。
        pose: MediaPipe Pose 模型。
    """

    def __init__(
        self,
        camera_id=0,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        video_path=None,
        mirror_video=False,
    ):
        """初始化摄像头和 MediaPipe 模型。

        Args:
            camera_id: 摄像头设备 ID (默认 0), video_path 非空时忽略。
            width: 采集分辨率宽度 (摄像头模式)。
            height: 采集分辨率高度 (摄像头模式)。
            fps: 目标帧率 (摄像头模式)。
            video_path: 视频文件路径, 传入后优先使用视频替代摄像头。
            mirror_video: 视频模式也水平镜像，以模拟本项目的摄像头画面。
        """
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self._is_video = True
        else:
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self._is_video = False
        self._mirror_video = bool(mirror_video)

        if not self.cap.isOpened():
            source = video_path if video_path else f"camera {camera_id}"
            raise RuntimeError(f"无法打开输入源: {source}")

        captured_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = float(captured_fps) if captured_fps and captured_fps > 0 else float(fps)

        # CLAHE 增强 — 改善低对比度画面
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE,
        )

        # MediaPipe 模型
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=HANDS_MODEL_COMPLEXITY,
            min_detection_confidence=HANDS_DETECTION_CONFIDENCE,
            min_tracking_confidence=HANDS_TRACKING_CONFIDENCE,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_MODEL_COMPLEXITY,
            min_detection_confidence=POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING_CONFIDENCE,
        )

    def read(self):
        """读取一帧, 返回 (BGR帧, 手部数据, Pose关键点)。

        Returns:
            tuple: (frame, hands_list, pose_landmarks)
                   hands_list 为 None 或手部字典列表。
                   frame 为 None 表示读取失败。
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        if not self._is_video or self._mirror_video:
            frame = cv2.flip(frame, 1)

        # CLAHE 增强 — LAB 色彩空间 L 通道均衡化
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self.clahe.apply(l_ch)
        enhanced = cv2.cvtColor(
            cv2.merge([l_eq, a_ch, b_ch]),
            cv2.COLOR_LAB2BGR,
        )
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        # 手部检测
        hand_results = self.hands.process(rgb)
        # 全身骨架检测
        pose_results = self.pose.process(rgb)
        pose_lms = pose_results.pose_landmarks

        # 解析手部数据
        hands = []
        if hand_results.multi_hand_landmarks:
            handedness = getattr(hand_results, "multi_handedness", None) or []
            for index, lm in enumerate(hand_results.multi_hand_landmarks):
                wrist = lm.landmark[0]
                mid_mcp = lm.landmark[9]
                all_lms = [{"x": p.x, "y": p.y, "z": p.z} for p in lm.landmark]
                classification = (
                    handedness[index].classification[0]
                    if index < len(handedness) and handedness[index].classification
                    else None
                )
                hands.append(
                    {
                        "id_hint": classification.label if classification else None,
                        "id_confidence": (float(classification.score) if classification else 0.0),
                        "palm_center": {
                            "x": (wrist.x + mid_mcp.x) / 2,
                            "y": (wrist.y + mid_mcp.y) / 2,
                            "z": (wrist.z + mid_mcp.z) / 2,
                        },
                        "landmarks": all_lms,
                    }
                )

        # Pose 腕关节降级 — Hands 检测不到时使用
        if not hands and pose_lms:
            for wrist_id in (15, 16):  # left_wrist, right_wrist
                lm = pose_lms.landmark[wrist_id]
                if lm.visibility > POSE_WRIST_VISIBILITY:
                    hands.append(
                        {
                            "id_hint": "Left" if wrist_id == 15 else "Right",
                            "id_confidence": float(lm.visibility),
                            "palm_center": {"x": lm.x, "y": lm.y, "z": lm.z},
                            "landmarks": [],
                        }
                    )

        return frame, (hands if hands else None), pose_lms

    def release(self):
        """释放摄像头和 MediaPipe 资源。"""
        self.cap.release()
        self.hands.close()
        self.pose.close()
