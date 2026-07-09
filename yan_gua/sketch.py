"""py5 Sketch 生命周期 — 演卦 · 星尘太极主程序.

将所有子系统连接起来: 摄像头 → 手势分析 → GPU粒子物理 → 渲染。
"""

import time
from collections import deque

import numpy as np
import py5

from yan_gua.camera_renderer import CameraRenderer
from yan_gua.config import (
    BG_B,
    BG_G,
    BG_R,
    CAM_H,
    CAM_MARGIN,
    CAM_W,
    INK_COLORS,
    MAX_SPEED,
    PARTICLE_COUNT,
    TRAIL_ALPHA,
    WARM_ACCENT,
    WARM_LIGHT,
    WINDOW_H,
    WINDOW_W,
)
from yan_gua.motion import MotionAnalyzer
from yan_gua.physics import CloudParticles
from yan_gua.tracking import HandTracker

# ---- 全局单例 ----
_tracker = None
_analyzer = None
_cloud = None
_cam_renderer = None
_last_time = 0.0
_show_debug = False
_fps_buffer = deque(maxlen=30)


# ---- 辅助绘图函数 ----


def _draw_camera_border(x, y, w, h):
    """摄像头小窗装饰边框 — 水墨画装裱风格。"""
    # 外层阴影
    py5.no_fill()
    py5.stroke_weight(3)
    py5.stroke(*INK_COLORS[4], 35)
    py5.rect(x - 3, y - 3, w + 6, h + 6, 8)
    # 中层墨线
    py5.stroke_weight(2)
    py5.stroke(*INK_COLORS[2], 70)
    py5.rect(x - 1, y - 1, w + 2, h + 2, 6)
    # 内层暖金细线
    py5.stroke_weight(1)
    py5.stroke(*WARM_ACCENT, 45)
    py5.rect(x - 2, y - 2, w + 4, h + 4, 7)
    py5.no_stroke()


def _collect_active_hands(hand_states):
    """从 MotionAnalyzer 状态列表提取活跃手。

    Returns:
        list[tuple]: [(pos_2d, features_dict), ...]
    """
    active = []
    for st in hand_states:
        if st['hand_detected'] and st['hand_world_pos'] is not None:
            feat = {
                'speed': st['speed'],
                'curvature': st['curvature'],
                'z_velocity': st['z_velocity'],
                'hand_velocity': st['hand_velocity'].copy(),
                'hand_detected': True,
            }
            active.append((st['hand_world_pos'], feat))
    return active


def _draw_hand_glows(active_hands):
    """双手光晕 — 柔光 + 内核。"""
    for hand_pos, _ in active_hands:
        hx, hy = hand_pos[0], hand_pos[1]
        # 三层柔光晕
        for r, a in [(35, 6), (22, 12), (12, 22)]:
            py5.fill(*WARM_ACCENT, a)
            py5.circle(hx, hy, r * 2)
        # 内核亮点
        py5.fill(*WARM_LIGHT, 35)
        py5.circle(hx, hy, 10)


def _draw_debug_info(active_hands):
    """调试信息叠加 (D 键切换)。"""
    py5.fill(255, 255, 255, 200)
    py5.text_size(13)
    py5.text_align(py5.LEFT)

    any_hand = len(active_hands) > 0
    status = f"TRACKING x{len(active_hands)}" if any_hand else "waiting..."
    py5.text(f"status: {status}", 15, 25)

    if any_hand:
        primary = active_hands[0][1]
        v = 1.0 - primary['speed'] / MAX_SPEED
        py5.text(
            f"viscosity: {v:.2f}  curvature: {primary['curvature']:.3f}"
            f"  breath: {primary['z_velocity']:+.1f}",
            15, 45,
        )

    py5.text(
        f"particles: {PARTICLE_COUNT}  fps: {py5.frame_rate:.0f}",
        15, 65,
    )

    for i, (hand_pos, _) in enumerate(active_hands):
        py5.text(
            f"hand{i + 1}: ({hand_pos[0]:.0f}, {hand_pos[1]:.0f})",
            15, 85 + i * 20,
        )


def _draw_idle_prompt():
    """无手时的提示文字。"""
    py5.fill(200, 190, 170, 60)
    py5.text_align(py5.CENTER)
    py5.text_size(18)
    py5.text(
        "Raise your hands / 举起双手",
        WINDOW_W / 2, WINDOW_H - 60,
    )


# ---- py5 生命周期回调 ----


def settings():
    """py5 在 setup() 前调用 — 计算自适应窗口大小。"""
    global WINDOW_W, WINDOW_H
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        WINDOW_W = int(sw * 0.85)
        WINDOW_H = int(sh * 0.85)
    except Exception:
        pass  # 回退到默认 1280×720

    # 同步更新 config 模块中的值
    import yan_gua.config as cfg
    cfg.WINDOW_W = WINDOW_W
    cfg.WINDOW_H = WINDOW_H

    py5.size(WINDOW_W, WINDOW_H, py5.P2D)


def setup():
    """py5 初始化 — 创建所有子系统。"""
    global _tracker, _analyzer, _cloud, _cam_renderer, _last_time

    py5.frame_rate(60)
    py5.window_title("演卦 · 星尘太极")

    print("Camera...", end=" ", flush=True)
    _tracker = HandTracker()
    print("OK")

    print("Cloud...", end=" ", flush=True)
    _analyzer = MotionAnalyzer(WINDOW_W, WINDOW_H)
    _cloud = CloudParticles(PARTICLE_COUNT, WINDOW_W, WINDOW_H)
    _last_time = time.perf_counter()
    print("OK")

    print("Camera renderer...", end=" ", flush=True)
    _cam_renderer = CameraRenderer(CAM_W, CAM_H, CAM_MARGIN)
    _cam_renderer.create_py5_image()
    print("OK")

    print("=== 演卦 · 星尘太极 (py5 + Taichi GPU) ===")
    print("  [ESC] quit  [F] fullscreen  [D] debug")


def draw():
    """py5 主循环 — 每帧: 摄像头 → 手势 → 粒子物理(GPU) → 渲染。"""
    global _last_time, _show_debug, _cam_renderer

    # 帧时间
    now = time.perf_counter()
    dt = now - _last_time
    _last_time = now

    # ---- 摄像头 + 手势 + 骨架 ----
    frame, hands, pose_lms = _tracker.read()
    hand_states = _analyzer.process(hands, now)
    active_hands = _collect_active_hands(hand_states)
    any_hand = len(active_hands) > 0

    # ---- 摄像头小窗 (水墨滤镜) ----
    cam_img, cam_x, cam_y = _cam_renderer.process(frame, hands, pose_lms)

    # ---- 拖尾: 半透明覆盖 ----
    py5.no_stroke()
    py5.fill(BG_R, BG_G, BG_B, TRAIL_ALPHA)
    py5.rect(0, 0, WINDOW_W, WINDOW_H)

    # ---- 粒子物理 + 渲染 ----
    _cloud.update(dt, active_hands)
    main_hx = active_hands[0][0][0] if any_hand else None
    main_hy = active_hands[0][0][1] if any_hand else None
    _cloud.draw(any_hand, main_hx, main_hy)

    # ---- 双手光晕 ----
    _draw_hand_glows(active_hands)

    # ---- 摄像头小窗 (右下角) ----
    if cam_img is not None:
        py5.blend_mode(py5.BLEND)
        _draw_camera_border(cam_x, cam_y, CAM_W, CAM_H)
        py5.image(cam_img, cam_x, cam_y)

    # ---- 调试信息 / 提示 ----
    if _show_debug:
        _draw_debug_info(active_hands)

    if not any_hand:
        _draw_idle_prompt()


def key_pressed():
    """键盘事件处理。"""
    global _show_debug
    if py5.key == py5.ESC:
        py5.exit_sketch()
    elif py5.key in ('f', 'F'):
        py5.full_screen(not py5.is_full_screen)
    elif py5.key in ('d', 'D'):
        _show_debug = not _show_debug


def exiting():
    """py5 退出清理。"""
    global _tracker
    if _tracker:
        _tracker.release()
    print("\nYanGua - exited")
