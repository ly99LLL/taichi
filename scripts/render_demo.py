#!/usr/bin/env python3
"""生成不读取摄像头、不包含真人画面的确定性演示视频与封面截图。"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import taichi as ti

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CANVAS_W, CANVAS_H = 1280, 720
DEFAULT_DURATION = 10.0
DEFAULT_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用合成双手轨迹生成无摄像头、无真人画面的演卦演示。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/yan-gua-demo.mp4"),
        help="演示视频路径（默认：artifacts/yan-gua-demo.mp4）",
    )
    parser.add_argument(
        "--poster",
        type=Path,
        default=Path("docs/assets/yan-gua-demo.png"),
        help="封面截图路径（默认：docs/assets/yan-gua-demo.png）",
    )
    parser.add_argument(
        "--poster-time",
        type=float,
        default=4.0,
        help="截取封面的时间点，单位为秒（默认：4.0）",
    )
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--arch",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Taichi 后端（默认：auto）",
    )
    return parser.parse_args()


def synthetic_hands(timestamp: float) -> list[dict]:
    """返回两只身份稳定的合成手；6.4 秒后撤场以展示余涡。"""
    if timestamp >= 6.4:
        return []

    if timestamp < 4.8:
        angle = timestamp * 0.72
        left_x = 0.365 + math.sin(angle) * 0.016
        right_x = 0.635 - math.sin(angle) * 0.016
        left_y = 0.50 + math.cos(angle * 0.83) * 0.025
        right_y = 0.50 - math.cos(angle * 0.83) * 0.025
        depth = math.sin(angle * 0.55) * 0.006
    else:
        burst = timestamp - 4.8
        travel = min(burst / 1.6, 1.0)
        oscillation = math.sin(burst * 11.0)
        left_x = 0.365 - travel * 0.16 + oscillation * 0.024
        right_x = 0.635 + travel * 0.16 - oscillation * 0.024
        left_y = 0.50 + math.sin(burst * 8.0) * 0.11
        right_y = 0.50 - math.sin(burst * 8.0) * 0.11
        depth = math.sin(burst * 7.0) * 0.025

    return [
        {
            "id_hint": "Left",
            "palm_center": {"x": left_x, "y": left_y, "z": depth},
            "landmarks": [],
        },
        {
            "id_hint": "Right",
            "palm_center": {"x": right_x, "y": right_y, "z": -depth},
            "landmarks": [],
        },
    ]


def phase_label(vortices: list[dict]) -> str:
    phases = {field["phase"] for field in vortices if field["active"]}
    if "dispersing" in phases:
        return "FAST / BREAK"
    if "echo" in phases:
        return "ECHO / RELEASE"
    if "holding" in phases:
        return "SLOW / HOLD"
    if "forming" in phases:
        return "FORMING"
    return "DORMANT"


def init_taichi(arch_name: str) -> None:
    arch = {"auto": ti.gpu, "cuda": ti.cuda, "cpu": ti.cpu}[arch_name]
    ti.init(arch=arch, random_seed=42)


def main() -> int:
    args = parse_args()
    if args.duration <= 0 or args.fps <= 0:
        raise SystemExit("--duration 和 --fps 必须大于 0")

    init_taichi(args.arch)

    from yan_gua.config import PARTICLE_COUNT
    from yan_gua.motion import MotionAnalyzer
    from yan_gua.offline_renderer import ParticleFrameRenderer
    from yan_gua.physics import CloudParticles
    from yan_gua.vortex import VortexController

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)

    total_frames = max(1, round(args.duration * args.fps))
    poster_frame = min(
        total_frames - 1,
        max(0, round(args.poster_time * args.fps)),
    )
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (CANVAS_W, CANVAS_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{args.output}")

    analyzer = MotionAnalyzer(CANVAS_W, CANVAS_H)
    controller = VortexController()
    cloud = CloudParticles(PARTICLE_COUNT, CANVAS_W, CANVAS_H, seed=42)
    renderer = ParticleFrameRenderer(CANVAS_W, CANVAS_H, args.fps)
    started = time.perf_counter()
    dt = 1.0 / args.fps

    print("演卦 · 合成演示 / 无摄像头输入")
    print(f"  {args.output} · {CANVAS_W}x{CANVAS_H} @ {args.fps:g} fps")

    try:
        for frame_index in range(total_frames):
            timestamp = frame_index * dt
            hands = synthetic_hands(timestamp)
            hand_states = analyzer.process(hands, timestamp)
            vortices = controller.update(hand_states, dt)
            cloud.update(dt, vortices)
            frame = renderer.render(
                cloud,
                vortices,
                phase_label=phase_label(vortices),
                synthetic=True,
            )
            writer.write(frame)

            if frame_index == poster_frame:
                if not cv2.imwrite(str(args.poster), frame):
                    raise RuntimeError(f"无法写入封面截图：{args.poster}")

            if (frame_index + 1) % round(args.fps * 2) == 0:
                elapsed = time.perf_counter() - started
                render_fps = (frame_index + 1) / max(elapsed, 0.001)
                print(f"  {frame_index + 1}/{total_frames} · {render_fps:.1f} fps")
    finally:
        writer.release()

    elapsed = time.perf_counter() - started
    print(f"完成：{total_frames} 帧 / {elapsed:.1f}s")
    print(f"封面：{args.poster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
