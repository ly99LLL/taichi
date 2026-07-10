#!/usr/bin/env python3
"""视频离线渲染器：复用实时程序的追踪、涡场生命周期和 Taichi 物理。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import taichi as ti

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yan_gua.config import (
    CAM_H,
    CAM_MARGIN,
    CAM_W,
    PARTICLE_COUNT,
    UI_BORDER,
)

CANVAS_W, CANVAS_H = 1280, 720
UI_BORDER_BGR = (UI_BORDER[2], UI_BORDER[1], UI_BORDER[0])


def draw_camera_border(canvas, x, y):
    cv2.rectangle(
        canvas,
        (x - 1, y - 1),
        (x + CAM_W + 1, y + CAM_H + 1),
        UI_BORDER_BGR,
        1,
        cv2.LINE_AA,
    )
    gold = (148, 195, 218)
    cyan = (174, 145, 111)
    corner = 20
    cv2.line(canvas, (x - 2, y - 2), (x + corner, y - 2), gold, 1, cv2.LINE_AA)
    cv2.line(canvas, (x - 2, y - 2), (x - 2, y + corner), gold, 1, cv2.LINE_AA)
    cv2.line(
        canvas,
        (x + CAM_W + 2, y + CAM_H + 2),
        (x + CAM_W - corner, y + CAM_H + 2),
        cyan,
        1,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (x + CAM_W + 2, y + CAM_H + 2),
        (x + CAM_W + 2, y + CAM_H - corner),
        cyan,
        1,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从视频输入离线渲染演卦粒子效果。",
    )
    parser.add_argument("input", nargs="?", type=Path, default=Path("参考视频.mp4"))
    parser.add_argument("output", nargs="?", type=Path, default=Path("效果视频.mp4"))
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="不绘制右下角视频预览，仅输出粒子场。",
    )
    parser.add_argument(
        "--arch",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Taichi 后端（默认：auto）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    arch = {"auto": ti.gpu, "cuda": ti.cuda, "cpu": ti.cpu}[args.arch]
    ti.init(arch=arch, random_seed=42)

    from yan_gua.camera_renderer import CameraRenderer
    from yan_gua.motion import MotionAnalyzer
    from yan_gua.offline_renderer import ParticleFrameRenderer
    from yan_gua.physics import CloudParticles
    from yan_gua.tracking import HandTracker
    from yan_gua.vortex import VortexController, phase_label

    video_path = str(args.input)
    output_path = str(args.output)

    probe = cv2.VideoCapture(video_path)
    if not probe.isOpened():
        raise FileNotFoundError(f"无法打开输入视频：{args.input}")
    source_fps = probe.get(cv2.CAP_PROP_FPS)
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    if source_fps <= 0:
        source_fps = 30.0

    print("演卦 · 双生涡场 / 离线渲染")
    print(f"  {video_path} → {output_path}")
    print(f"  {CANVAS_W}x{CANVAS_H} @ {source_fps:.2f} fps · {PARTICLE_COUNT} dust")

    tracker = HandTracker(video_path=video_path)
    analyzer = MotionAnalyzer(CANVAS_W, CANVAS_H)
    vortex_controller = VortexController()
    cloud = CloudParticles(PARTICLE_COUNT, CANVAS_W, CANVAS_H, seed=42)
    camera_renderer = CameraRenderer(CAM_W, CAM_H, CAM_MARGIN)
    renderer = ParticleFrameRenderer(CANVAS_W, CANVAS_H, source_fps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (CANVAS_W, CANVAS_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {output_path}")

    frame_index = 0
    started = time.perf_counter()
    dt = 1.0 / source_fps

    while True:
        frame, hands, pose_landmarks = tracker.read()
        if frame is None:
            break

        hand_states = analyzer.process(hands, frame_index * dt)
        vortices = vortex_controller.update(hand_states, dt)
        cloud.update(dt, vortices)

        label = phase_label(vortices)
        final = renderer.render(cloud, vortices, phase_label=label)

        if not args.no_camera:
            camera_image, camera_x, camera_y = camera_renderer.process_bgr(
                frame,
                hands,
                pose_landmarks,
                CANVAS_W,
                CANVAS_H,
            )
            if camera_image is not None:
                draw_camera_border(final, camera_x, camera_y)
                final[
                    camera_y : camera_y + CAM_H,
                    camera_x : camera_x + CAM_W,
                ] = camera_image

        writer.write(final)
        frame_index += 1
        if frame_index % 60 == 0:
            elapsed = time.perf_counter() - started
            render_fps = frame_index / elapsed
            eta = (total_frames - frame_index) / max(render_fps, 0.001)
            print(f"  {frame_index}/{total_frames} · {render_fps:.1f} fps · ETA {eta:.0f}s")

    tracker.release()
    writer.release()
    elapsed = time.perf_counter() - started
    print(f"完成：{frame_index} 帧 / {elapsed:.1f}s")


if __name__ == "__main__":
    main()
