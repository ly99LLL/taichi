"""py5 主循环 — 摄像头、双手涡场、常驻尘场与极简界面。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import py5

from yan_gua.camera_renderer import CameraRenderer
from yan_gua.config import (
    BG_B,
    BG_G,
    BG_R,
    CAM_H,
    CAM_MARGIN,
    CAM_W,
    COHERENT_COLOR,
    ECHO_COLOR,
    PARTICLE_COUNT,
    SCATTER_COLOR,
    TRAIL_ALPHA,
    UI_BORDER,
    UI_MUTED,
    UI_PRIMARY,
    VORTEX_ORBIT_RADIUS,
    WINDOW_H,
    WINDOW_W,
)
from yan_gua.motion import MotionAnalyzer
from yan_gua.physics import CloudParticles
from yan_gua.tracking import HandTracker
from yan_gua.vortex import VortexController

_tracker = None
_analyzer = None
_vortex_controller = None
_cloud = None
_cam_renderer = None
_last_time = 0.0
_show_debug = False
_video_path = None
_record_path = None
_mirror_video = True
_source_fps = 0.0
_source_frame_idx = 0
_record_writer = None

_btn_cx = 0
_btn_cy = 30
_btn_r = 20
_btn_hover = False
_btn_flash = 0.0


def configure_input(video_path=None, record_path=None, mirror_video=True):
    """在 py5 启动前配置输入源和可选的主窗口录制。"""
    global _video_path, _record_path, _mirror_video
    _video_path = str(Path(video_path).resolve()) if video_path else None
    _record_path = str(Path(record_path).resolve()) if record_path else None
    _mirror_video = bool(mirror_video)


def _open_record_writer():
    global _record_writer
    if not _record_path:
        return

    output = Path(_record_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _record_writer = cv2.VideoWriter(
        str(output),
        fourcc,
        _source_fps,
        (WINDOW_W, WINDOW_H),
    )
    if not _record_writer.isOpened():
        raise RuntimeError(f"无法创建录制文件: {output}")


def _record_current_frame():
    if _record_writer is None:
        return
    rgb = py5.get_np_pixels(bands="RGB")
    _record_writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _draw_camera_border(x, y, w, h):
    """深色细框 + 双生涡场双色角标。"""
    py5.no_fill()
    py5.stroke(*UI_BORDER, 190)
    py5.stroke_weight(1)
    py5.rect(x - 1, y - 1, w + 2, h + 2, 4)

    corner = 20
    py5.stroke(*COHERENT_COLOR, 175)
    py5.line(x - 2, y - 2, x + corner, y - 2)
    py5.line(x - 2, y - 2, x - 2, y + corner)
    py5.stroke(*SCATTER_COLOR, 175)
    py5.line(x + w + 2, y + h + 2, x + w - corner, y + h + 2)
    py5.line(x + w + 2, y + h + 2, x + w + 2, y + h - corner)
    py5.no_stroke()


def _field_color(field):
    if field["phase"] == "echo":
        return ECHO_COLOR
    if field["scatter"] > field["coherence"]:
        return SCATTER_COLOR
    return COHERENT_COLOR


def _draw_vortex_marks(vortices):
    """仅画极淡的轨道缺口；主要形状仍由真实粒子给出。"""
    py5.no_fill()
    for field in vortices:
        if not field["active"] or field["position"] is None:
            continue
        x, y = field["position"]
        ring_radius = VORTEX_ORBIT_RADIUS * (1.0 + field["release"] * 0.9 + field["aperture"])
        color = _field_color(field)
        alpha = 8.0 + field["strength"] * (18.0 if field["observed"] else 7.0)
        angle = py5.frame_count * 0.012 * field["spin"]
        py5.stroke(*color, alpha)
        py5.stroke_weight(0.8)
        py5.arc(
            x,
            y,
            ring_radius * 2,
            ring_radius * 2,
            angle,
            angle + 0.72,
        )
        py5.arc(
            x,
            y,
            ring_radius * 2,
            ring_radius * 2,
            angle + 3.1416,
            angle + 3.86,
        )
        if field["observed"]:
            py5.no_stroke()
            py5.fill(*color, 70)
            py5.circle(x, y, 2.4)
            py5.no_fill()
    py5.no_stroke()


def _draw_debug_info(vortices):
    py5.fill(*UI_PRIMARY, 190)
    py5.text_size(12)
    py5.text_align(py5.LEFT, py5.BASELINE)
    active = sum(field["active"] for field in vortices)
    py5.text(f"VORTEX FIELD / {active}", 18, 26)
    py5.fill(*UI_MUTED, 180)
    py5.text(f"{PARTICLE_COUNT} DUST  ·  {py5.frame_rate:.0f} FPS", 18, 45)

    row = 66
    for field in vortices:
        if not field["active"]:
            continue
        py5.text(
            f"{field['slot'] + 1:02d}  {field['phase'].upper():10s}  "
            f"COH {field['coherence']:.2f}  "
            f"BREAK {field['scatter']:.2f}  "
            f"LIFE {field['strength']:.2f}",
            18,
            row,
        )
        row += 18


def _draw_idle_prompt():
    py5.fill(*UI_MUTED, 105)
    py5.text_align(py5.LEFT, py5.BASELINE)
    py5.text_size(12)
    py5.text("RAISE HANDS  ·  MOVE SLOWLY TO HOLD", 22, WINDOW_H - 24)
    py5.fill(*UI_MUTED, 70)
    py5.text("举手入场  ·  慢则成旋", 22, WINDOW_H - 8)


def _restart():
    global _tracker, _analyzer, _vortex_controller, _cloud, _cam_renderer
    global _last_time, _btn_flash, _source_frame_idx
    print("\nRestarting...", end=" ", flush=True)

    if _tracker:
        _tracker.release()
    _tracker = HandTracker(
        video_path=_video_path,
        mirror_video=bool(_video_path and _mirror_video),
    )
    _analyzer = MotionAnalyzer(WINDOW_W, WINDOW_H)
    _vortex_controller = VortexController()
    _cloud = CloudParticles(PARTICLE_COUNT, WINDOW_W, WINDOW_H)
    _cam_renderer = CameraRenderer(CAM_W, CAM_H, CAM_MARGIN)
    _cam_renderer.create_py5_image()
    _last_time = time.perf_counter()
    _source_frame_idx = 0
    _btn_flash = 1.0
    print("OK")


def _draw_restart_button():
    global _btn_cx, _btn_hover, _btn_flash
    _btn_cx = WINDOW_W - 34
    dx = py5.mouse_x - _btn_cx
    dy = py5.mouse_y - _btn_cy
    _btn_hover = dx * dx + dy * dy < _btn_r * _btn_r
    _btn_flash = max(0.0, _btn_flash * 0.84 - 0.002)

    alpha = 150 if _btn_hover else 70
    alpha += int(_btn_flash * 75)
    py5.no_fill()
    py5.stroke(*UI_BORDER, 220)
    py5.stroke_weight(1)
    py5.circle(_btn_cx, _btn_cy, _btn_r * 2)
    py5.no_stroke()
    py5.fill(*UI_PRIMARY, min(alpha, 230))
    py5.text_align(py5.CENTER, py5.CENTER)
    py5.text_size(15)
    py5.text("↻", _btn_cx, _btn_cy + 1)


def settings():
    global WINDOW_W, WINDOW_H
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        WINDOW_W = int(screen_w * 0.85)
        WINDOW_H = int(screen_h * 0.85)
    except Exception:
        pass

    import yan_gua.config as cfg

    cfg.WINDOW_W = WINDOW_W
    cfg.WINDOW_H = WINDOW_H
    py5.size(WINDOW_W, WINDOW_H, py5.P2D)


def setup():
    global _tracker, _analyzer, _vortex_controller, _cloud, _cam_renderer, _last_time
    global _source_fps, _source_frame_idx

    py5.window_title("演卦 · 双生涡场")
    _tracker = HandTracker(
        video_path=_video_path,
        mirror_video=bool(_video_path and _mirror_video),
    )
    _source_fps = _tracker.source_fps if _video_path else 60.0
    _source_frame_idx = 0
    py5.frame_rate(_source_fps)

    _analyzer = MotionAnalyzer(WINDOW_W, WINDOW_H)
    _vortex_controller = VortexController()
    _cloud = CloudParticles(PARTICLE_COUNT, WINDOW_W, WINDOW_H)
    _cam_renderer = CameraRenderer(CAM_W, CAM_H, CAM_MARGIN)
    _cam_renderer.create_py5_image()
    _last_time = time.perf_counter()
    py5.background(BG_R, BG_G, BG_B)

    _open_record_writer()
    print("=== 演卦 · 双生涡场 (py5 + Taichi GPU) ===")
    print("  慢则聚旋，快则解旋；失手留余涡，双手成流桥。")
    print("  [ESC] quit  [F] fullscreen  [D] field data  [click] restart")


def draw():
    global _last_time, _cam_renderer, _source_frame_idx

    frame, hands, pose_landmarks = _tracker.read()
    if frame is None:
        if _video_path:
            py5.exit_sketch()
        return

    if _video_path:
        dt = 1.0 / _source_fps
        now = _source_frame_idx * dt
    else:
        now = time.perf_counter()
        dt = now - _last_time
        _last_time = now

    hand_states = _analyzer.process(hands, now)
    vortices = _vortex_controller.update(hand_states, dt)
    observed = any(field["observed"] for field in vortices)
    any_field = any(field["active"] for field in vortices)

    camera_image, camera_x, camera_y = _cam_renderer.process(
        frame,
        hands,
        pose_landmarks,
    )

    py5.no_stroke()
    py5.fill(BG_R, BG_G, BG_B, TRAIL_ALPHA)
    py5.rect(0, 0, WINDOW_W, WINDOW_H)

    _cloud.update(dt, vortices)
    _cloud.draw(vortices)
    _draw_vortex_marks(vortices)

    if camera_image is not None:
        py5.blend_mode(py5.BLEND)
        _draw_camera_border(camera_x, camera_y, CAM_W, CAM_H)
        py5.image(camera_image, camera_x, camera_y)

    _draw_restart_button()
    if _show_debug:
        _draw_debug_info(vortices)
    if not observed and not any_field:
        _draw_idle_prompt()

    _record_current_frame()
    if _video_path:
        _source_frame_idx += 1


def key_pressed():
    global _show_debug
    if py5.key == py5.ESC:
        py5.exit_sketch()
    elif py5.key in ("f", "F"):
        py5.full_screen(not py5.is_full_screen)
    elif py5.key in ("d", "D"):
        _show_debug = not _show_debug


def mouse_pressed():
    dx = py5.mouse_x - _btn_cx
    dy = py5.mouse_y - _btn_cy
    if dx * dx + dy * dy < _btn_r * _btn_r:
        _restart()


def exiting():
    global _tracker, _record_writer
    if _tracker:
        _tracker.release()
    if _record_writer:
        _record_writer.release()
        _record_writer = None
        print(f"Recording saved: {_record_path}")
    print("\nYanGua - exited")
