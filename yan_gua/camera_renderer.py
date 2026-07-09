"""摄像头水墨渲染器 — 右下角小窗.

水墨滤镜 + 毛笔笔触手部/Pose骨架可视化 → 右下角小窗。
"""

import cv2
import numpy as np

from yan_gua.config import (
    BG_B,
    BG_G,
    BG_R,
    CAM_BILATERAL_D,
    CAM_BILATERAL_SIGMA,
    CAM_BRUSH_BLUR,
    CAM_BRUSH_OPACITY,
    CAM_EDGE_STRENGTH,
    CAM_FINGERTIP_R,
    CAM_H,
    CAM_JOINT_R,
    CAM_MARGIN,
    CAM_VIGNETTE,
    CAM_W,
    CAM_WARM_BLEND,
)


class CameraRenderer:
    """水墨滤镜 + 毛笔笔触手部/Pose骨架可视化 → 右下角小窗。"""

    # 手部骨架连接 (MediaPipe Hands 21 点拓扑)
    HAND_CONNECTIONS = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # 拇指
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # 食指
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # 中指
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # 无名指
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # 小指
        (5, 9),
        (9, 13),
        (13, 17),  # 掌纹横线
    ]
    TIP_IDS = {4, 8, 12, 16, 20}

    # Pose 骨架连接
    POSE_CONNECTIONS = [
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),  # 躯干
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # 手臂
        (15, 17),
        (17, 19),
        (19, 21),
        (15, 21),  # 左手
        (16, 18),
        (18, 20),
        (20, 22),
        (16, 22),  # 右手
        (23, 25),
        (25, 27),
        (27, 29),
        (27, 31),
        (29, 31),  # 左腿
        (24, 26),
        (26, 28),
        (28, 30),
        (28, 32),
        (30, 32),  # 右腿
    ]
    KEY_JOINTS = {11, 12, 15, 16, 23, 24, 25, 26, 27, 28}

    # BGR 暖金色
    LINE_COLOR = (90, 155, 200)  # WARM_ACCENT
    GLOW_COLOR = (150, 210, 240)  # WARM_LIGHT
    JOINT_COLOR = (80, 140, 190)

    def __init__(self, cam_w=CAM_W, cam_h=CAM_H, margin=CAM_MARGIN):
        """初始化水墨渲染器。

        Args:
            cam_w: 小窗宽度。
            cam_h: 小窗高度。
            margin: 距窗口边缘的距离。
        """
        self.w = cam_w
        self.h = cam_h
        self.margin = margin
        self._py5_img = None

    def create_py5_image(self):
        """预分配 Py5Image (在 setup 中调用一次)。"""
        import py5

        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        arr[:, :] = (BG_R, BG_G, BG_B)
        self._py5_img = py5.create_image_from_numpy(arr, "RGB")

    def process(self, bgr_frame, hands_data, pose_landmarks=None, show_pose=False):
        """处理一帧: 滤镜 → 骨架笔触 → 手部笔触 → 暗角 → Py5Image。

        Args:
            show_pose: 是否绘制全身 Pose 骨架 (默认 False, 仅手部)。

        Returns:
            tuple: (py5_img, x, y) 或 (None, 0, 0)
        """
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
        """处理一帧: 滤镜 → 骨架笔触 → 手部笔触 → 暗角 → BGR numpy 数组。

        与 process() 相同的处理管线, 但返回 BGR numpy 数组而非 Py5Image,
        供离线渲染等非 py5 场景使用。

        Args:
            canvas_w: 画布宽度 (用于计算右下角位置)。
            canvas_h: 画布高度 (用于计算右下角位置)。

        Returns:
            tuple: (bgr_array, x, y) 或 (None, 0, 0)
        """
        if bgr_frame is None:
            return None, 0, 0

        # 1. 缩放
        small = cv2.resize(bgr_frame, (self.w, self.h))

        # 2. 水墨滤镜
        filtered = self._ink_wash(small)

        # 3. Pose 骨架笔触 (默认关闭, 仅当 show_pose=True 时绘制)
        if pose_landmarks and show_pose:
            filtered = self._draw_pose_skeleton(filtered, pose_landmarks)

        # 4. 手部毛笔笔触
        if hands_data:
            filtered = self._draw_hand_brush(filtered, hands_data)

        # 5. 暗角
        filtered = self._vignette(filtered)

        x = canvas_w - self.w - self.margin
        y = canvas_h - self.h - self.margin
        return filtered, x, y

    # ---- 内部渲染方法 ----

    def _ink_wash(self, img):
        """轻量水墨滤镜: 双边磨皮 → 暖色调映射 → 墨线叠加。

        人保持清晰可辨, 不完全风格化。
        """
        # 双边滤波 (磨皮保边)
        smooth = cv2.bilateralFilter(
            img,
            CAM_BILATERAL_D,
            CAM_BILATERAL_SIGMA,
            CAM_BILATERAL_SIGMA,
        )

        # 灰度 → 暖色映射 (暗→深褐, 亮→暖米)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        b = np.clip(16 + gray * 120, 0, 255).astype(np.uint8)
        g = np.clip(22 + gray * 155, 0, 255).astype(np.uint8)
        r = np.clip(28 + gray * 180, 0, 255).astype(np.uint8)
        warm = cv2.merge([b, g, r])

        # 混合原色 (保留真实色彩)
        result = cv2.addWeighted(smooth, 1.0 - CAM_WARM_BLEND, warm, CAM_WARM_BLEND, 0)

        # Canny 边缘 → 墨线叠加
        edges = cv2.Canny(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), 40, 120)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges_blur = (
            cv2.GaussianBlur(
                edges.astype(np.float32),
                (3, 3),
                0,
            )
            / 255.0
        )

        ink = np.array([20, 28, 38], dtype=np.float32)  # 深褐墨色 BGR
        ef = edges_blur * CAM_EDGE_STRENGTH
        result = (
            result.astype(np.float32) * (1.0 - ef[:, :, np.newaxis])
            + ink.reshape(1, 1, 3) * ef[:, :, np.newaxis]
        ).astype(np.uint8)

        return result

    def _draw_hand_brush(self, img, hands_data):
        """毛笔笔触风格手部关键点 — 暖金流动曲线 + 指尖光点。"""
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for hand in hands_data:
            lms = hand.get("landmarks", [])
            if len(lms) < 21:
                continue

            pts = self._landmarks_to_pixels(lms, w, h)

            # 连线 — 掌部略粗, 指部略细
            for a, b in self.HAND_CONNECTIONS:
                is_palm = (a == 0 and b in (5, 9, 13, 17)) or (a in (5, 9, 13) and b in (9, 13, 17))
                thick = 3 if is_palm else 2
                cv2.line(overlay, pts[a], pts[b], self.LINE_COLOR, thick, cv2.LINE_AA)

            # 光点 — 指尖大/亮, 关节小
            for j, pt in enumerate(pts):
                if j in self.TIP_IDS:
                    cv2.circle(overlay, pt, CAM_FINGERTIP_R + 2, self.GLOW_COLOR, 1, cv2.LINE_AA)
                    cv2.circle(overlay, pt, CAM_FINGERTIP_R, self.LINE_COLOR, -1, cv2.LINE_AA)
                else:
                    cv2.circle(overlay, pt, CAM_JOINT_R, self.LINE_COLOR, -1, cv2.LINE_AA)

        # 高斯扩散 → 毛笔晕染 + 锐利底层叠加
        overlay_blur = cv2.GaussianBlur(
            overlay,
            (CAM_BRUSH_BLUR, CAM_BRUSH_BLUR),
            0,
        )
        result = cv2.addWeighted(img, 1.0, overlay_blur, CAM_BRUSH_OPACITY, 0)
        result = cv2.addWeighted(result, 1.0, overlay, 0.12, 0)
        return result

    def _draw_pose_skeleton(self, img, pose_lms):
        """全身 Pose 骨架笔触 — 暖金流动线条 + 关节光点 + 腕部加强。"""
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # 有效关键点
        pts = {}
        for j in range(33):
            lm = pose_lms.landmark[j]
            if lm.visibility > 0.4:
                pts[j] = (max(0, min(w - 1, int(lm.x * w))), max(0, min(h - 1, int(lm.y * h))))

        # 骨架连线
        for a, b in self.POSE_CONNECTIONS:
            if a in pts and b in pts:
                thick = 2 if a in self.KEY_JOINTS and b in self.KEY_JOINTS else 1
                cv2.line(overlay, pts[a], pts[b], self.LINE_COLOR, thick, cv2.LINE_AA)

        # 关节光点
        for j, pt in pts.items():
            if j in self.KEY_JOINTS:
                cv2.circle(overlay, pt, 4, self.GLOW_COLOR, 1, cv2.LINE_AA)
                cv2.circle(overlay, pt, 3, self.JOINT_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(overlay, pt, 2, self.LINE_COLOR, -1, cv2.LINE_AA)

        # 腕关节特别加强 (粒子效果中心)
        for wid in (15, 16):
            if wid in pts:
                pt = pts[wid]
                cv2.circle(overlay, pt, 7, self.GLOW_COLOR, 2, cv2.LINE_AA)
                cv2.circle(overlay, pt, 4, (110, 185, 230), -1, cv2.LINE_AA)

        # 晕染 + 叠加
        overlay_blur = cv2.GaussianBlur(overlay, (3, 3), 0)
        result = cv2.addWeighted(img, 1.0, overlay_blur, 0.35, 0)
        result = cv2.addWeighted(result, 1.0, overlay, 0.15, 0)
        return result

    @staticmethod
    def _landmarks_to_pixels(lms, w, h):
        """将归一化关键点坐标转换为像素坐标。"""
        return [
            (max(0, min(w - 1, int(lm["x"] * w))), max(0, min(h - 1, int(lm["y"] * h))))
            for lm in lms
        ]

    @staticmethod
    def _vignette(img):
        """柔和暗角效果。"""
        h, w = img.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx_m, cy_m = (w - 1) / 2, (h - 1) / 2
        dist = np.sqrt((xs - cx_m) ** 2 + (ys - cy_m) ** 2)
        max_d = np.sqrt(cx_m**2 + cy_m**2)
        v = 1.0 - (dist / max_d) ** 1.5 * CAM_VIGNETTE
        v = np.clip(v, 0.0, 1.0)
        return (img.astype(np.float32) * v[:, :, np.newaxis]).astype(np.uint8)
