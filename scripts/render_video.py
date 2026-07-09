#!/usr/bin/env python3
"""
演卦 · 视频渲染器 v4
=====================
复用主程序模块:
- CloudParticles (Taichi GPU 物理 — 与实时 py5 完全一致)
- MotionAnalyzer (手势特征提取)
- CameraRenderer (水墨滤镜 + 毛笔笔触 + 暗角)
- HandTracker (视频文件输入)

纯 OpenCV 渲染，输出效果与实时 py5 界面一致。

用法: python scripts/render_video.py [输入视频] [输出视频]
默认: 参考视频.mp4 → 效果视频.mp4
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import taichi as ti

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ---- Taichi GPU 初始化 (必须在 CloudParticles 导入前) ----
ti.init(arch=ti.cuda, random_seed=42)

from yan_gua.camera_renderer import CameraRenderer
from yan_gua.config import (
    BG_B,
    BG_G,
    BG_R,
    CAM_H,
    CAM_MARGIN,
    CAM_W,
    INFLUENCE_RADIUS,
    INK_COLORS,
    NUM_INK_LEVELS,
    PARTICLE_COUNT,
    TRAIL_ALPHA,
    WARM_ACCENT,
    WARM_LIGHT,
)
from yan_gua.motion import MotionAnalyzer
from yan_gua.physics import CloudParticles
from yan_gua.tracking import HandTracker

# ============================================================
# 渲染常量
# ============================================================
CANVAS_W, CANVAS_H = 1280, 720

# BGR 色值 (OpenCV 用)
BG_BGR = (BG_B, BG_G, BG_R)
INK_BGR = [(b, g, r) for r, g, b in INK_COLORS]
WARM_ACCENT_BGR = (WARM_ACCENT[2], WARM_ACCENT[1], WARM_ACCENT[0])
WARM_LIGHT_BGR = (WARM_LIGHT[2], WARM_LIGHT[1], WARM_LIGHT[0])

# 粒子 alpha 分带 — 近似 py5 逐粒子 alpha 混合
# py5 粒子 alpha 范围 2-22, 分 4 档
_ALPHA_BANDS = [
    (1, 6, 0.010),  # 极淡 (alpha 1-5,  blend ≈ 3/255)
    (6, 14, 0.035),  # 中层 (alpha 6-13, blend ≈ 10/255)
    (14, 22, 0.065),  # 亮层 (alpha 14-21, blend ≈ 18/255)
    (22, 255, 0.095),  # 高亮 (alpha 22+,   blend ≈ 24/255)
]


# ============================================================
# 粒子渲染 (OpenCV — 匹配 py5 CloudParticles.draw)
# ============================================================
def draw_particles(trail_buf, particles, active_hands):
    """逐粒子渲染 + alpha 分带混合，匹配 py5 视觉效果。

    py5 每粒子独立 alpha 混合 → OpenCV 按 alpha 分 4 档，
    每档内粒子集中绘制到临时画布后统一混合。
    """
    c = particles.c
    order = np.argsort(particles.alpha)

    hand_positions = [(int(hp[0]), int(hp[1])) for hp, _ in active_hands if hp is not None]

    for lo, hi, blend_weight in _ALPHA_BANDS:
        temp = np.full_like(trail_buf, BG_BGR)
        has_particle = False

        for j in range(c):
            idx = order[j]
            a = particles.alpha[idx]
            if a < lo or a >= hi:
                continue

            r = max(1, int(particles.radius[idx]))
            px_val = int(particles.px[idx])
            py_val = int(particles.py[idx])

            ink = min(particles.ink_level[idx], NUM_INK_LEVELS - 1)
            color = list(INK_BGR[ink])

            # 手部附近混暖金色 (与 py5 _blend_warm 一致)
            _blend_warm(color, px_val, py_val, hand_positions)

            cv2.circle(temp, (px_val, py_val), r, tuple(color), -1, cv2.LINE_AA)
            has_particle = True

        if has_particle:
            cv2.addWeighted(temp, blend_weight, trail_buf, 1.0 - blend_weight, 0, trail_buf)


def _blend_warm(color, px_val, py_val, hand_positions):
    """手部附近粒子注入暖金色 (与 physics._blend_warm 一致)。"""
    for hx, hy in hand_positions:
        d = np.sqrt((px_val - hx) ** 2 + (py_val - hy) ** 2)
        if d < INFLUENCE_RADIUS * 0.7:
            t = 1.0 - d / (INFLUENCE_RADIUS * 0.7)
            color[0] = int(color[0] + (WARM_LIGHT_BGR[0] - color[0]) * t * 0.5)
            color[1] = int(color[1] + (WARM_LIGHT_BGR[1] - color[1]) * t * 0.5)
            color[2] = int(color[2] + (WARM_LIGHT_BGR[2] - color[2]) * t * 0.5)


# ============================================================
# 手部光晕 (匹配 py5 _draw_hand_glows)
# ============================================================
def draw_hand_glows(canvas, active_hands):
    """双手光晕 — 三层柔光 + 内核。"""
    for hand_pos, _ in active_hands:
        if hand_pos is None:
            continue
        hx, hy = int(hand_pos[0]), int(hand_pos[1])
        for rad, al in [(35, 6), (22, 12), (12, 22)]:
            ov = canvas.copy()
            cv2.circle(ov, (hx, hy), rad, WARM_ACCENT_BGR, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, al / 255, canvas, 1 - al / 255, 0, canvas)
        ov = canvas.copy()
        cv2.circle(ov, (hx, hy), 10, WARM_LIGHT_BGR, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 35 / 255, canvas, 1 - 35 / 255, 0, canvas)


# ============================================================
# 水墨小窗边框 (匹配 py5 _draw_camera_border)
# ============================================================
def draw_camera_border(canvas, x, y):
    """摄像头小窗装饰边框 — 水墨画装裱风格。"""
    w, h = CAM_W, CAM_H
    # 外层阴影
    cv2.rectangle(canvas, (x - 3, y - 3), (x + w + 3, y + h + 3), INK_BGR[4], 3)
    # 中层墨线
    cv2.rectangle(canvas, (x - 1, y - 1), (x + w + 1, y + h + 1), INK_BGR[2], 2)
    # 内层暖金细线
    cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 2, y + h + 2), WARM_ACCENT_BGR, 1)


# ============================================================
# 无手提示
# ============================================================
def draw_idle_prompt(canvas):
    """无手时的提示文字。"""
    cv2.putText(
        canvas,
        "Raise your hands / Ju Qi Shuang Shou",
        (CANVAS_W // 2 - 200, CANVAS_H - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 190, 170),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# 主流程
# ============================================================
def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "参考视频.mp4"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "效果视频.mp4"

    print("演卦 · 视频渲染器 v4 (Taichi GPU + 共享模块)")
    print(f"  输入: {video_path}")
    print(f"  输出: {output_path}")

    # ---- 从视频读取帧率和尺寸 ----
    probe = cv2.VideoCapture(video_path)
    src_fps = probe.get(cv2.CAP_PROP_FPS)
    src_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    if src_fps <= 0:
        src_fps = 30.0

    print(f"  源: {src_w}x{src_h} @ {src_fps:.1f}fps, {total_frames} 帧")
    print(f"  画布: {CANVAS_W}x{CANVAS_H}, 粒子: {PARTICLE_COUNT}")

    # ---- 拖尾透明度 — 按帧率匹配 py5 60fps TRAIL_ALPHA=7 ----
    # py5 60fps: (1 - 7/255)^60 ≈ 0.188/s 衰退
    # 适配到源帧率: trail_alpha = (1 - 0.188^(1/src_fps)) * 255
    decay_per_second = (1.0 - TRAIL_ALPHA / 255.0) ** 60.0
    trail_alpha = (1.0 - decay_per_second ** (1.0 / src_fps)) * 255.0
    print(
        f"  拖尾: alpha={trail_alpha:.1f} (匹配 py5 TRAIL_ALPHA={TRAIL_ALPHA} "
        f"@ 60fps → {src_fps:.0f}fps)"
    )

    # ---- 子系统 ----
    print("Camera...", end=" ", flush=True)
    tracker = HandTracker(video_path=video_path)
    print("OK")

    print("Cloud...", end=" ", flush=True)
    analyzer = MotionAnalyzer(CANVAS_W, CANVAS_H)
    cloud = CloudParticles(PARTICLE_COUNT, CANVAS_W, CANVAS_H)
    print("OK")

    print("Camera renderer...", end=" ", flush=True)
    cam_renderer = CameraRenderer(CAM_W, CAM_H, CAM_MARGIN)
    print("OK")

    # ---- 输出视频 ----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, src_fps, (CANVAS_W, CANVAS_H))

    # 拖尾缓冲 — 初始化为背景色
    trail_buf = np.full((CANVAS_H, CANVAS_W, 3), BG_BGR, dtype=np.uint8)

    frame_idx = 0
    start_time = time.perf_counter()
    ta = trail_alpha / 255.0

    print("渲染中...")

    while True:
        # ---- 视频帧 + 手部检测 ----
        frame, hands, pose_lms = tracker.read()
        if frame is None:
            break

        now = time.perf_counter()
        dt = 1.0 / src_fps

        # ---- 手势特征提取 (MotionAnalyzer — 与主程序一致) ----
        hand_states = analyzer.process(hands, now)
        active_hands = []
        for st in hand_states:
            if st["hand_detected"] and st["hand_world_pos"] is not None:
                feat = {
                    "speed": st["speed"],
                    "curvature": st["curvature"],
                    "z_velocity": st["z_velocity"],
                    "hand_velocity": st["hand_velocity"].copy(),
                    "hand_detected": True,
                }
                active_hands.append((st["hand_world_pos"], feat))

        any_hand = len(active_hands) > 0

        # ---- 粒子物理 (Taichi GPU — 与主程序完全一致) ----
        cloud.update(dt, active_hands)

        # ---- 拖尾消退 (匹配 py5: fill(BG, alpha) + rect) ----
        dark = np.full_like(trail_buf, BG_BGR)
        cv2.addWeighted(trail_buf, 1.0 - ta, dark, ta, 0, trail_buf)

        # ---- 粒子渲染 ----
        draw_particles(trail_buf, cloud, active_hands)

        # ---- 手部光晕 ----
        draw_hand_glows(trail_buf, active_hands)

        # ---- 最终画布 (复制一份, 叠加小窗等) ----
        final = trail_buf.copy()

        # ---- 右下角摄像头小窗 (CameraRenderer — 与主程序一致) ----
        cam_img, cam_x, cam_y = cam_renderer.process_bgr(
            frame,
            hands,
            pose_lms,
            CANVAS_W,
            CANVAS_H,
            show_pose=False,
        )
        if cam_img is not None:
            draw_camera_border(final, cam_x, cam_y)
            # 将 cam_img 居中放入 CAM_W×CAM_H 区域
            ch, cw = cam_img.shape[:2]
            ox = cam_x + (CAM_W - cw) // 2
            oy = cam_y + (CAM_H - ch) // 2
            final[oy : oy + ch, ox : ox + cw] = cam_img

        # ---- 无手提示 ----
        if not any_hand:
            draw_idle_prompt(final)

        out.write(final)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - start_time
            fps = frame_idx / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / fps if fps > 0 else 0
            print(
                f"  {frame_idx}/{total_frames} "
                f"({frame_idx / total_frames * 100:.0f}%)  "
                f"{fps:.1f} fps  ETA {eta:.0f}s"
            )

    tracker.release()
    out.release()
    elapsed = time.perf_counter() - start_time
    print(f"\n完成! {frame_idx} 帧 / {elapsed:.0f}s → {output_path}")


if __name__ == "__main__":
    main()
