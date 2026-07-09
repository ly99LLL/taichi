"""原彩摄像头窗 — 真实影像 + 双手涡场识别光学层。

影像本身只做极轻的对比度、饱和度和暗角处理；所有风格化都限制在手部
关键点、掌心断环和微型状态标识上，避免破坏肤色与现场环境。
"""

import cv2
import numpy as np

from yan_gua.config import (
    BG_B,
    BG_G,
    BG_R,
    CAM_BRUSH_BLUR,
    CAM_BRUSH_OPACITY,
    CAM_COLOR_CONTRAST,
    CAM_COLOR_SATURATION,
    CAM_FINGERTIP_R,
    CAM_H,
    CAM_JOINT_R,
    CAM_MARGIN,
    CAM_VIGNETTE,
    CAM_W,
)


class CameraRenderer:
    """保留原彩的摄像头画面，并叠加与双生涡场一致的手部识别效果。"""

    HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (5, 9),
        (9, 13),
        (13, 17),
    ]
    TIP_IDS = {4, 8, 12, 16, 20}

    POSE_CONNECTIONS = [
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (15, 17),
        (17, 19),
        (19, 21),
        (15, 21),
        (16, 18),
        (18, 20),
        (20, 22),
        (16, 22),
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        (29, 31),
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        (30, 32),
    ]
    KEY_JOINTS = {11, 12, 15, 16, 23, 24, 25, 26, 27, 28}

    # BGR：左手旧金，右手月青；都取自主画面的粒子色温。
    LEFT_COLOR = (148, 195, 218)
    LEFT_LIGHT = (193, 222, 239)
    RIGHT_COLOR = (174, 145, 111)
    RIGHT_LIGHT = (220, 190, 152)
    NEUTRAL_COLOR = (176, 184, 194)

    def __init__(self, cam_w=CAM_W, cam_h=CAM_H, margin=CAM_MARGIN):
        self.w = cam_w
        self.h = cam_h
        self.margin = margin
        self._py5_img = None

    def create_py5_image(self):
        """预分配 Py5Image；真正画面仍在每帧从原始 BGR 图像创建。"""
        import py5

        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        arr[:, :] = (BG_R, BG_G, BG_B)
        self._py5_img = py5.create_image_from_numpy(arr, "RGB")

    def process(self, bgr_frame, hands_data, pose_landmarks=None, show_pose=False):
        """处理一帧并转换为 Py5Image。"""
        import py5

        result, x, y = self.process_bgr(
            bgr_frame,
            hands_data,
            pose_landmarks,
            canvas_w=py5.width,
            canvas_h=py5.height,
            show_pose=show_pose,
        )
        if result is None:
            return None, 0, 0

        rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        self._py5_img = py5.create_image_from_numpy(rgb, "RGB")
        return self._py5_img, x, y

    def process_bgr(
        self,
        bgr_frame,
        hands_data,
        pose_landmarks=None,
        canvas_w=1280,
        canvas_h=720,
        show_pose=False,
    ):
        """返回原彩小窗及其在主画布右下角的位置。"""
        if bgr_frame is None:
            return None, 0, 0

        small = cv2.resize(
            bgr_frame,
            (self.w, self.h),
            interpolation=cv2.INTER_AREA,
        )
        result = self._preserve_color(small)
        result = self._vignette(result)

        if pose_landmarks and show_pose:
            result = self._draw_pose_skeleton(result, pose_landmarks)
        if hands_data:
            result = self._draw_hand_effect(result, hands_data)

        x = canvas_w - self.w - self.margin
        y = canvas_h - self.h - self.margin
        return result, x, y

    @staticmethod
    def _preserve_color(img):
        """保持原始色相，仅做 2% 量级的电影对比与饱和度校正。"""
        source = img.astype(np.float32) / 255.0
        contrasted = np.clip(
            (source - 0.5) * CAM_COLOR_CONTRAST + 0.5,
            0.0,
            1.0,
        )

        hsv = cv2.cvtColor(
            (contrasted * 255.0).astype(np.uint8),
            cv2.COLOR_BGR2HSV,
        )
        saturation = hsv[:, :, 1].astype(np.float32) * CAM_COLOR_SATURATION
        hsv[:, :, 1] = np.clip(saturation, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _draw_hand_effect(self, img, hands_data):
        """绘制身份双色骨架、指尖光点和掌心断环。"""
        h, w = img.shape[:2]
        glow = np.zeros_like(img)
        crisp = np.zeros_like(img)

        for hand_index, hand in enumerate(hands_data[:2]):
            color, light = self._hand_colors(hand, hand_index)
            landmarks = hand.get("landmarks", [])
            points = self._landmarks_to_pixels(landmarks, w, h) if len(landmarks) >= 21 else []
            palm = self._palm_pixel(hand, points, w, h)

            if points:
                for a, b in self.HAND_CONNECTIONS:
                    is_palm = (a == 0 and b in (5, 9, 13, 17)) or (
                        a in (5, 9, 13) and b in (9, 13, 17)
                    )
                    glow_width = 5 if is_palm else 4
                    crisp_width = 2 if is_palm else 1
                    cv2.line(
                        glow,
                        points[a],
                        points[b],
                        color,
                        glow_width,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        crisp,
                        points[a],
                        points[b],
                        light,
                        crisp_width,
                        cv2.LINE_AA,
                    )

                for joint_index, point in enumerate(points):
                    if joint_index in self.TIP_IDS:
                        cv2.circle(
                            glow,
                            point,
                            CAM_FINGERTIP_R + 4,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            crisp,
                            point,
                            CAM_FINGERTIP_R + 1,
                            light,
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.circle(crisp, point, 1, light, -1, cv2.LINE_AA)
                    elif joint_index in (0, 5, 9, 13, 17):
                        cv2.circle(
                            crisp,
                            point,
                            CAM_JOINT_R,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

            if palm is not None:
                orbit_radius = self._palm_orbit_radius(points)
                self._draw_palm_orbit(
                    glow,
                    crisp,
                    palm,
                    orbit_radius,
                    color,
                    light,
                    hand_index,
                )

        blurred = cv2.GaussianBlur(glow, (CAM_BRUSH_BLUR, CAM_BRUSH_BLUR), 0)
        result = cv2.addWeighted(img, 1.0, blurred, CAM_BRUSH_OPACITY, 0)
        return cv2.addWeighted(result, 1.0, crisp, 0.72, 0)

    @classmethod
    def _hand_colors(cls, hand, hand_index):
        hint = str(hand.get("id_hint", "")).lower()
        is_right = hint.startswith("right") or (not hint and hand_index == 1)
        if is_right:
            return cls.RIGHT_COLOR, cls.RIGHT_LIGHT
        return cls.LEFT_COLOR, cls.LEFT_LIGHT

    @staticmethod
    def _palm_pixel(hand, points, w, h):
        palm = hand.get("palm_center")
        if palm:
            return (
                max(0, min(w - 1, int(palm["x"] * w))),
                max(0, min(h - 1, int(palm["y"] * h))),
            )
        if len(points) >= 10:
            return (
                (points[0][0] + points[9][0]) // 2,
                (points[0][1] + points[9][1]) // 2,
            )
        return None

    @staticmethod
    def _palm_orbit_radius(points):
        if len(points) < 18:
            return 12
        palm_span = float(np.linalg.norm(np.subtract(points[5], points[17])))
        return int(np.clip(palm_span * 0.72, 9, 24))

    @staticmethod
    def _draw_palm_orbit(
        glow,
        crisp,
        palm,
        radius,
        color,
        light,
        hand_index,
    ):
        angle_offset = 28 if hand_index == 0 else -28
        cv2.ellipse(
            glow,
            palm,
            (radius, radius),
            angle_offset,
            18,
            148,
            color,
            4,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            glow,
            palm,
            (radius, radius),
            angle_offset,
            202,
            332,
            color,
            4,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            crisp,
            palm,
            (radius, radius),
            angle_offset,
            18,
            148,
            light,
            1,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            crisp,
            palm,
            (radius, radius),
            angle_offset,
            202,
            332,
            light,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(crisp, palm, 2, light, -1, cv2.LINE_AA)

    def _draw_pose_skeleton(self, img, pose_landmarks):
        """可选的低亮度 Pose 骨架；默认关闭。"""
        h, w = img.shape[:2]
        overlay = np.zeros_like(img)
        points = {}
        for index in range(33):
            landmark = pose_landmarks.landmark[index]
            if landmark.visibility > 0.4:
                points[index] = (
                    max(0, min(w - 1, int(landmark.x * w))),
                    max(0, min(h - 1, int(landmark.y * h))),
                )

        for a, b in self.POSE_CONNECTIONS:
            if a in points and b in points:
                cv2.line(
                    overlay,
                    points[a],
                    points[b],
                    self.NEUTRAL_COLOR,
                    1,
                    cv2.LINE_AA,
                )
        return cv2.addWeighted(img, 1.0, overlay, 0.22, 0)

    @staticmethod
    def _landmarks_to_pixels(landmarks, w, h):
        return [
            (
                max(0, min(w - 1, int(landmark["x"] * w))),
                max(0, min(h - 1, int(landmark["y"] * h))),
            )
            for landmark in landmarks
        ]

    @staticmethod
    def _vignette(img):
        """仅在边缘压暗约 10%，中心完全保留原始曝光。"""
        h, w = img.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        center_x, center_y = (w - 1) / 2, (h - 1) / 2
        distance = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
        maximum = np.sqrt(center_x**2 + center_y**2)
        mask = 1.0 - (distance / maximum) ** 1.8 * CAM_VIGNETTE
        mask = np.clip(mask, 1.0 - CAM_VIGNETTE, 1.0)
        return (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
