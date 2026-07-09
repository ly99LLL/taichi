#!/usr/bin/env python3
"""
演卦 · 视频渲染器 v3
=====================
程序界面风格：全屏星尘粒子 + 右下角参考视频小窗。
纯 OpenCV + NumPy，不依赖 py5。
从 yan_gua 包导入共享模块 (config, motion)。

用法: python scripts/render_video.py
输入: 参考视频.mp4
输出: 效果视频.mp4
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# 确保项目根目录在 sys.path 中 (script 在子目录时也能导入 yan_gua)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from yan_gua.config import (
    BG_B, BG_G, BG_R,
    CURVATURE_REF,
    INFLUENCE_RADIUS,
    MAX_SPEED,
    SMOOTH_ALPHA,
    WARM_ACCENT,
    WARM_LIGHT,
)
from yan_gua.motion import MotionAnalyzer

# ============================================================
# 视频渲染专用常量 (与 yan_gua/config 不同)
# ============================================================
BG_BGR = (BG_B, BG_G, BG_R)

INK_COLORS = [
    (180, 160, 130),
    (140, 120, 95),
    (105, 85, 65),
    (75, 58, 42),
    (48, 35, 24),
    (28, 18, 12),
]
INK_COLORS_BGR = [(b, g, r) for r, g, b in INK_COLORS]

WARM_ACCENT_BGR = (WARM_ACCENT[2], WARM_ACCENT[1], WARM_ACCENT[0])
WARM_LIGHT_BGR = (WARM_LIGHT[2], WARM_LIGHT[1], WARM_LIGHT[0])

# 画布 & 小窗
CANVAS_W, CANVAS_H = 1280, 720
CAM_W, CAM_H = 280, 210
CAM_MARGIN = 20

# 粒子物理参数 (视频专用)
PARTICLE_COUNT = 4000
HISTORY_SIZE = 45
BASE_DAMPING = 0.985
TRAIL_ALPHA = 15                        # 视频拖尾 (琥珀流动感)
PARTICLE_ALPHA_MAX = 22
PARTICLE_SIZE_MAX = 32


# MotionAnalyzer 从 yan_gua.motion 导入 (顶部 import)


# ============================================================
# 视频手部检测器
# ============================================================
class VideoHandTracker:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            model_complexity=0, min_detection_confidence=0.3,
            min_tracking_confidence=0.15)

    def read(self):
        ret, frame = self.cap.read()
        if not ret: return None, None, None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(rgb)
        hands = []
        if hand_results.multi_hand_landmarks:
            for lm in hand_results.multi_hand_landmarks:
                w = lm.landmark[0]; m = lm.landmark[9]
                all_lms = [{'x': p.x, 'y': p.y, 'z': p.z} for p in lm.landmark]
                hands.append({
                    'palm_center': {
                        'x': (w.x + m.x) / 2, 'y': (w.y + m.y) / 2,
                        'z': (w.z + m.z) / 2,
                    },
                    'landmarks': all_lms,
                })

        return frame, (hands if hands else None), None

    def release(self):
        self.cap.release()
        self.hands.close()


# ============================================================
# 粒子系统 (纯 NumPy)
# ============================================================
class CloudParticles:
    def __init__(self, count, w, h):
        self.c = count; self.w = w; self.h = h
        rng = np.random.default_rng()
        self.px = rng.uniform(0, w, count).astype(np.float32)
        self.py = rng.uniform(0, h, count).astype(np.float32)
        self.vx = rng.uniform(-0.5, 0.5, count).astype(np.float32)
        self.vy = rng.uniform(-0.5, 0.5, count).astype(np.float32)
        self.alpha = rng.uniform(2, 12, count).astype(np.float32)
        self.radius = rng.uniform(3, 15, count).astype(np.float32)
        self.ink_level = rng.integers(0, len(INK_COLORS_BGR), count)
        self.base_alpha = self.alpha.copy()
        self.base_radius = self.radius.copy()
        self.base_ink = self.ink_level.copy()
        self.rng = rng

    def update(self, dt, hands):
        dt = min(dt, 0.1)
        c, w, h = self.c, self.w, self.h

        self.vx += (self.rng.uniform(-0.6, 0.6, c) * 1.0 * dt).astype(np.float32)
        self.vy += (self.rng.uniform(-0.6, 0.6, c) * 1.0 * dt).astype(np.float32)
        cx_m, cy_m = w / 2, h / 2
        self.vx -= (self.px - cx_m) * 0.00003
        self.vy -= (self.py - cy_m) * 0.00003

        influenced = np.zeros(c, dtype=bool)

        for hand_pos, feat in hands:
            if hand_pos is None: continue
            hx, hy = hand_pos[0], hand_pos[1]
            dx = self.px - hx; dy = self.py - hy
            dist = np.sqrt(dx * dx + dy * dy)
            mask = dist < INFLUENCE_RADIUS
            if not np.any(mask): continue

            influenced = influenced | mask
            mdx, mdy = dx[mask], dy[mask]
            mdist = dist[mask]
            mvx, mvy = self.vx[mask], self.vy[mask]

            t_val = 1.0 - mdist / INFLUENCE_RADIUS
            falloff = t_val ** 2 * (3.0 - 2.0 * t_val)
            rnx = mdx / (mdist + 0.001)
            rny = mdy / (mdist + 0.001)

            norm_speed = min(feat['speed'] / MAX_SPEED, 1.0)
            viscosity = 1.0 - norm_speed
            hvx, hvy = feat['hand_velocity'][0], feat['hand_velocity'][1]

            # 中空: 0-25% 粒子向外猛推 (不回吸)
            inner_zone = mdist < INFLUENCE_RADIUS * 0.25
            if np.any(inner_zone):
                push = falloff[inner_zone] * (8.0 + viscosity * 6.0)
                mvx[inner_zone] += rnx[inner_zone] * push
                mvy[inner_zone] += rny[inner_zone] * push

            # 环壁: 25-40% 轻吸维持边界
            ring_zone = (mdist >= INFLUENCE_RADIUS * 0.25) & (mdist < INFLUENCE_RADIUS * 0.4)
            if np.any(ring_zone):
                attract = falloff[ring_zone] * (0.8 + viscosity * 1.5)
                mvx[ring_zone] -= rnx[ring_zone] * attract
                mvy[ring_zone] -= rny[ring_zone] * attract

            # 粘性拖拽
            mvx += hvx * falloff * viscosity * 0.08
            mvy += hvy * falloff * viscosity * 0.08

            # 飞溅
            mvx += rnx * falloff * norm_speed * 2.5
            mvy += rny * falloff * norm_speed * 2.5

            # 旋涡: 基线 + 曲率
            tx, ty = -rny, rnx
            vortex_base = viscosity * 3.0
            vortex_curve = feat['curvature'] * (3.0 + viscosity * 10.0)
            vortex_force = vortex_base + vortex_curve
            mvx += tx * falloff * vortex_force
            mvy += ty * falloff * vortex_force

            # 呼吸
            breath = feat['z_velocity'] * 0.6
            mvx += rnx * falloff * breath
            mvy += rny * falloff * breath

            self.vx[mask] = mvx; self.vy[mask] = mvy

            activity = np.clip(
                norm_speed * 0.4 + feat['curvature'] * 2.0
                + abs(feat['z_velocity']) * 0.003, 0.0, 1.0)
            target_alpha = np.where(
                activity > 0.3, PARTICLE_ALPHA_MAX * activity * falloff,
                self.base_alpha[mask])
            target_radius = np.where(
                activity > 0.1, self.base_radius[mask] + falloff * 30 * activity,
                self.base_radius[mask])
            self.alpha[mask] += (target_alpha - self.alpha[mask]) * 0.2
            self.radius[mask] += (target_radius - self.radius[mask]) * 0.2
            warm_mask = mask & (activity > 0.4)
            if np.any(warm_mask):
                self.ink_level[warm_mask] = np.minimum(
                    self.ink_level[warm_mask] + 1, len(INK_COLORS_BGR) - 1)

            # 环带粒子额外增亮 → 漩涡环明显 (映射回全数组)
            ring_full = np.zeros(c, dtype=bool)
            if np.any(ring_zone):
                ring_full[np.where(mask)[0][ring_zone]] = True
            if np.any(ring_full):
                act_full = np.zeros(c)
                act_full[mask] = activity
                ring_bright = ring_full & (act_full > 0.15)
                if np.any(ring_bright):
                    self.alpha[ring_bright] = np.minimum(
                        self.alpha[ring_bright] + 3, PARTICLE_ALPHA_MAX + 5)
                    self.radius[ring_bright] = np.minimum(
                        self.radius[ring_bright] + 2, PARTICLE_SIZE_MAX + 5)

        not_inf = ~influenced
        if np.any(not_inf):
            self.alpha[not_inf] += (self.base_alpha[not_inf] - self.alpha[not_inf]) * 0.03
            self.radius[not_inf] += (self.base_radius[not_inf] - self.radius[not_inf]) * 0.02
            self.ink_level[not_inf] = self.base_ink[not_inf]

        self.px += self.vx * dt * 80
        self.py += self.vy * dt * 80
        self.vx *= BASE_DAMPING; self.vy *= BASE_DAMPING

        self.px = np.where(self.px < -50, w + 50,
                  np.where(self.px > w + 50, -50, self.px))
        self.py = np.where(self.py < -50, h + 50,
                  np.where(self.py > h + 50, -50, self.py))


# ============================================================
# 粒子渲染 (OpenCV)
# ============================================================
def draw_particles(trail_buf, particles, active_hands):
    """绘制粒子 + 手部光晕到 trail_buf。粒子用 alpha 混合匹配 py5 效果。"""
    c = particles.c
    order = np.argsort(particles.alpha)

    hand_positions = [hp for hp, _ in active_hands if hp is not None]

    # 所有粒子画到一张临时画布 (背景色)
    temp = np.full_like(trail_buf, BG_BGR)
    for i in range(c):
        idx = order[i]
        a = particles.alpha[idx]
        if a < 2.0: continue
        r = particles.radius[idx]
        ink = particles.ink_level[idx]
        color = list(INK_COLORS_BGR[min(ink, len(INK_COLORS_BGR) - 1)])

        px_val, py_val = int(particles.px[idx]), int(particles.py[idx])
        for hx, hy in hand_positions:
            d = np.sqrt((px_val - hx) ** 2 + (py_val - hy) ** 2)
            if d < INFLUENCE_RADIUS * 0.7:
                t = 1.0 - d / (INFLUENCE_RADIUS * 0.7)
                color[0] = int(color[0] + (WARM_LIGHT_BGR[0] - color[0]) * t * 0.4)
                color[1] = int(color[1] + (WARM_LIGHT_BGR[1] - color[1]) * t * 0.4)
                color[2] = int(color[2] + (WARM_LIGHT_BGR[2] - color[2]) * t * 0.4)

        cv2.circle(temp, (px_val, py_val), int(r), tuple(color), -1, cv2.LINE_AA)

    # 混合: 粒子权重 — 可见但克制，匹配暗星尘氛围
    cv2.addWeighted(temp, 0.10, trail_buf, 0.90, 0, trail_buf)

    # 手部光晕 — 柔光风格
    for hx, hy in hand_positions:
        ix, iy = int(hx), int(hy)
        for rad, al in [(35, 6), (22, 12), (12, 22)]:
            ov = trail_buf.copy()
            cv2.circle(ov, (ix, iy), rad, WARM_ACCENT_BGR, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, al / 255.0, trail_buf, 1.0 - al / 255.0, 0, trail_buf)
        ov = trail_buf.copy()
        cv2.circle(ov, (ix, iy), 10, WARM_LIGHT_BGR, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 35 / 255.0, trail_buf, 1.0 - 35 / 255.0, 0, trail_buf)


# ============================================================
# 右下角视频小窗 (水墨滤镜 + 毛笔笔触 + 暗角 + 边框)
# ============================================================
def process_camera_window(bgr_frame, hands_data, pose_landmarks=None):
    """处理参考视频帧 → 水墨小窗。返回 (cam_img, cam_x, cam_y)"""
    h, w = bgr_frame.shape[:2]

    # 计算缩放: 保持宽高比，适配 CAM_W × CAM_H
    scale = min(CAM_W / w, CAM_H / h)
    new_w, new_h = int(w * scale), int(h * scale)

    small = cv2.resize(bgr_frame, (new_w, new_h))

    # 水墨滤镜
    filtered = ink_wash(small)

    # 全身骨架笔触
    if pose_landmarks:
        filtered = draw_pose_skeleton(filtered, pose_landmarks, new_w, new_h)

    # 毛笔笔触手部关键点
    if hands_data:
        filtered = brush_landmarks(filtered, hands_data, new_w, new_h)

    # 暗角
    filtered = vignette(filtered)

    # 居中放入 CAM_W × CAM_H 的黑色画布
    cam_img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    cam_img[:, :] = BG_BGR
    ox = (CAM_W - new_w) // 2
    oy = (CAM_H - new_h) // 2
    cam_img[oy:oy + new_h, ox:ox + new_w] = filtered

    # 计算在画布上的位置 (右下角)
    cam_x = CANVAS_W - CAM_W - CAM_MARGIN
    cam_y = CANVAS_H - CAM_H - CAM_MARGIN

    return cam_img, cam_x, cam_y


def ink_wash(img):
    """轻水墨滤镜：磨皮 → 暖调 → 墨线"""
    smooth = cv2.bilateralFilter(img, 5, 40, 40)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    b = np.clip(16 + gray * 120, 0, 255).astype(np.uint8)
    g = np.clip(22 + gray * 155, 0, 255).astype(np.uint8)
    r = np.clip(28 + gray * 180, 0, 255).astype(np.uint8)
    warm = cv2.merge([b, g, r])
    result = cv2.addWeighted(smooth, 0.65, warm, 0.35, 0)
    edges = cv2.Canny(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_blur = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0) / 255.0
    ink_color = np.array([20, 28, 38], dtype=np.float32)
    ef = edges_blur * 0.2
    result = (result.astype(np.float32) * (1.0 - ef[:, :, np.newaxis])
              + ink_color.reshape(1, 1, 3) * ef[:, :, np.newaxis]).astype(np.uint8)
    return result


def draw_pose_skeleton(img, pose_lms, w, h):
    """全身骨架笔触 — 暖金流动线条 + 关节光点"""
    POSE_CONNECTIONS = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 17), (17, 19), (19, 21), (15, 21),
        (16, 18), (18, 20), (20, 22), (16, 22),
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    ]
    KEY_JOINTS = {11, 12, 15, 16, 23, 24, 25, 26, 27, 28}
    LINE_COLOR = (90, 155, 200)
    GLOW_COLOR = (150, 210, 240)
    JOINT_COLOR = (80, 140, 190)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    pts = {}
    for i in range(33):
        lm = pose_lms.landmark[i]
        if lm.visibility > 0.4:
            pts[i] = (max(0, min(w - 1, int(lm.x * w))),
                      max(0, min(h - 1, int(lm.y * h))))

    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            thick = 2 if a in KEY_JOINTS and b in KEY_JOINTS else 1
            cv2.line(overlay, pts[a], pts[b], LINE_COLOR, thick, cv2.LINE_AA)

    for i, pt in pts.items():
        if i in KEY_JOINTS:
            cv2.circle(overlay, pt, 4, GLOW_COLOR, 1, cv2.LINE_AA)
            cv2.circle(overlay, pt, 3, JOINT_COLOR, -1, cv2.LINE_AA)
        else:
            cv2.circle(overlay, pt, 2, LINE_COLOR, -1, cv2.LINE_AA)

    for wid in [15, 16]:
        if wid in pts:
            pt = pts[wid]
            cv2.circle(overlay, pt, 7, GLOW_COLOR, 2, cv2.LINE_AA)
            cv2.circle(overlay, pt, 4, (110, 185, 230), -1, cv2.LINE_AA)

    overlay_blur = cv2.GaussianBlur(overlay, (3, 3), 0)
    result = cv2.addWeighted(img, 1.0, overlay_blur, 0.35, 0)
    result = cv2.addWeighted(result, 1.0, overlay, 0.15, 0)
    return result


def brush_landmarks(img, hands_data, w, h):
    """毛笔笔触风格手部关键点"""
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    TIP_IDS = {4, 8, 12, 16, 20}
    LINE_COLOR = (90, 155, 200)
    GLOW_COLOR = (150, 210, 240)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for hand in hands_data:
        lms = hand.get('landmarks', [])
        if len(lms) < 21: continue
        pts = [(max(0, min(w - 1, int(lm['x'] * w))),
                max(0, min(h - 1, int(lm['y'] * h)))) for lm in lms]
        for a, b in CONNECTIONS:
            palm = (a == 0 and b in (5, 9, 13, 17)) or (a in (5, 9, 13) and b in (9, 13, 17))
            thick = 3 if palm else 2
            cv2.line(overlay, pts[a], pts[b], LINE_COLOR, thick, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            if i in TIP_IDS:
                cv2.circle(overlay, pt, 6, GLOW_COLOR, 1, cv2.LINE_AA)
                cv2.circle(overlay, pt, 4, LINE_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(overlay, pt, 2, LINE_COLOR, -1, cv2.LINE_AA)

    overlay_blur = cv2.GaussianBlur(overlay, (3, 3), 0)
    result = cv2.addWeighted(img, 1.0, overlay_blur, 0.35, 0)
    result = cv2.addWeighted(result, 1.0, overlay, 0.12, 0)
    return result


def vignette(img):
    """柔和暗角"""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx_m, cy_m = (w - 1) / 2, (h - 1) / 2
    dist = np.sqrt((xs - cx_m) ** 2 + (ys - cy_m) ** 2)
    max_d = np.sqrt(cx_m ** 2 + cy_m ** 2)
    v = 1.0 - (dist / max_d) ** 1.5 * 0.12
    v = np.clip(v, 0.0, 1.0)
    return (img.astype(np.float32) * v[:, :, np.newaxis]).astype(np.uint8)


def draw_camera_border(canvas, x, y):
    """水墨画装裱风格边框 (OpenCV 绘制)"""
    w, h = CAM_W, CAM_H
    # 外层阴影 (圆角近似 — 用略大的矩形)
    cv2.rectangle(canvas, (x - 3, y - 3), (x + w + 3, y + h + 3),
                  INK_COLORS_BGR[4], 3)
    # 中层墨线
    cv2.rectangle(canvas, (x - 1, y - 1), (x + w + 1, y + h + 1),
                  INK_COLORS_BGR[2], 2)
    # 内层暖金细线
    cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 2, y + h + 2),
                  WARM_ACCENT_BGR, 1)


# ============================================================
# 主流程
# ============================================================
def main():
    video_path = "参考视频.mp4"
    output_path = "效果视频.mp4"

    print(f"Reading: {video_path}")
    tracker = VideoHandTracker(video_path)
    print(f"  Source: {tracker.width}x{tracker.height} @ {tracker.fps:.1f}fps, "
          f"{tracker.total_frames} frames")
    print(f"  Canvas: {CANVAS_W}x{CANVAS_H} (landscape)")
    print(f"  Corner: {CAM_W}x{CAM_H} @ bottom-right")

    analyzer = MotionAnalyzer(CANVAS_W, CANVAS_H)
    particles = CloudParticles(PARTICLE_COUNT, CANVAS_W, CANVAS_H)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, tracker.fps, (CANVAS_W, CANVAS_H))
    print(f"Writing: {output_path}")

    # 拖尾缓冲 — 模拟 py5 的半透明覆盖
    trail_buf = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    trail_buf[:, :] = BG_BGR

    frame_idx = 0
    start_time = time.perf_counter()

    while True:
        bgr_frame, hands, pose_lms = tracker.read()
        if bgr_frame is None:
            break

        now = time.perf_counter()
        dt = 1.0 / tracker.fps

        # 手部检测 + 特征提取
        hand_states = analyzer.process(hands, now)
        active_hands = []
        for st in hand_states:
            if st['hand_detected'] and st['hand_world_pos'] is not None:
                feat = {
                    'speed': st['speed'], 'curvature': st['curvature'],
                    'z_velocity': st['z_velocity'],
                    'hand_velocity': st['hand_velocity'].copy(),
                    'hand_detected': True,
                }
                active_hands.append((st['hand_world_pos'], feat))

        any_hand = len(active_hands) > 0

        # ---- 粒子物理 ----
        particles.update(dt, active_hands)

        # ---- 拖尾消退: 旧粒子淡出 (模拟 py5 半透明覆盖) ----
        ta = TRAIL_ALPHA / 255.0
        dark_overlay = np.zeros_like(trail_buf)
        dark_overlay[:, :] = BG_BGR
        trail_buf = cv2.addWeighted(trail_buf, 1.0 - ta, dark_overlay, ta, 0)

        # ---- 粒子直接画在 trail_buf 上 (全不透明度) ----
        draw_particles(trail_buf, particles, active_hands)

        # 最终输出
        final = trail_buf.copy()

        # ---- 右下角视频小窗 ----
        cam_img, cam_x, cam_y = process_camera_window(bgr_frame, hands, pose_lms)
        draw_camera_border(final, cam_x, cam_y)
        final[cam_y:cam_y + CAM_H, cam_x:cam_x + CAM_W] = cam_img

        # ---- "举起双手" 提示 (无手时) ----
        if not any_hand:
            cv2.putText(final, "Raise your hands / Ju Qi Shuang Shou",
                        (CANVAS_W // 2 - 200, CANVAS_H - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 190, 170), 1, cv2.LINE_AA)

        out.write(final)

        frame_idx += 1
        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - start_time
            fps = frame_idx / elapsed
            eta = (tracker.total_frames - frame_idx) / fps if fps > 0 else 0
            print(f"  {frame_idx}/{tracker.total_frames} "
                  f"({frame_idx / tracker.total_frames * 100:.0f}%)  "
                  f"{fps:.1f} fps  ETA {eta:.0f}s")

    tracker.release()
    out.release()
    elapsed = time.perf_counter() - start_time
    print(f"\nDone! {frame_idx} frames in {elapsed:.0f}s → {output_path}")


if __name__ == '__main__':
    main()
