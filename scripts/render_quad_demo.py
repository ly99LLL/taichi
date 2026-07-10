#!/usr/bin/env python3
"""生成四象限涡场状态展示图 —— 不读取摄像头、不含真人画面。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import taichi as ti

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yan_gua.config import (
    BG_B,
    BG_G,
    BG_R,
    COHERENT_COLOR,
    ECHO_COLOR,
    PARTICLE_COUNT,
    SCATTER_COLOR,
    UI_MUTED,
    UI_PRIMARY,
)
from yan_gua.motion import MotionAnalyzer
from yan_gua.offline_renderer import ParticleFrameRenderer
from yan_gua.physics import CloudParticles
from yan_gua.runtime import TAICHI_ARCH_CHOICES, resolve_taichi_arch
from yan_gua.vortex import VortexController, phase_label

CANVAS_W, CANVAS_H = 640, 360
FPS = 30.0
SIM_SECONDS = 3.0  # 每个象限模拟时长
LABEL_H = 52  # 标签栏高度

BG_BGR = (BG_B, BG_G, BG_R)
MUTED_BGR = (UI_MUTED[2], UI_MUTED[1], UI_MUTED[0])
PRIMARY_BGR = (UI_PRIMARY[2], UI_PRIMARY[1], UI_PRIMARY[0])
COHERENT_BGR = (COHERENT_COLOR[2], COHERENT_COLOR[1], COHERENT_COLOR[0])
SCATTER_BGR = (SCATTER_COLOR[2], SCATTER_COLOR[1], SCATTER_COLOR[0])
ECHO_BGR = (ECHO_COLOR[2], ECHO_COLOR[1], ECHO_COLOR[0])

QUAD_NAMES = {
    "holding": ("SLOW / HOLD", "慢则聚旋", COHERENT_BGR),
    "dispersing": ("FAST / BREAK", "快则解旋", SCATTER_BGR),
    "echo": ("ECHO / RELEASE", "失手留余涡", ECHO_BGR),
    "dual": ("DUAL BRIDGE", "双手成流桥", PRIMARY_BGR),
}


def _quad_label_bar(label_en, label_cn, accent_bgr):
    """创建带双行标签的标题栏。"""
    bar = np.full((LABEL_H, CANVAS_W, 3), BG_BGR, dtype=np.uint8)
    cv2.line(bar, (0, LABEL_H - 1), (CANVAS_W, LABEL_H - 1), MUTED_BGR, 1)

    cv2.putText(
        bar,
        label_en,
        (22, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        accent_bgr,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bar,
        label_cn,
        (22, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        MUTED_BGR,
        1,
        cv2.LINE_AA,
    )
    # 角标色块
    cv2.rectangle(bar, (CANVAS_W - 18, 10), (CANVAS_W - 4, 22), accent_bgr, -1)
    return bar


def _simulate_and_render(hand_gen, label_key: str, arch, seed: int):
    """用合成手势驱动 Taichi 物理，返回最终帧。"""
    ti.init(arch=arch, random_seed=seed)

    analyzer = MotionAnalyzer(CANVAS_W, CANVAS_H)
    controller = VortexController()
    cloud = CloudParticles(PARTICLE_COUNT, CANVAS_W, CANVAS_H, seed=seed)
    renderer = ParticleFrameRenderer(CANVAS_W, CANVAS_H, FPS)

    dt = 1.0 / FPS
    total_frames = round(SIM_SECONDS * FPS)

    for fi in range(total_frames):
        timestamp = fi * dt
        hands = hand_gen(timestamp)
        hand_states = analyzer.process(hands, timestamp)
        vortices = controller.update(hand_states, dt)
        cloud.update(dt, vortices)

    label_en, label_cn, accent = QUAD_NAMES[label_key]
    frame = renderer.render(cloud, vortices, phase_label=phase_label(vortices))
    bar = _quad_label_bar(label_en, label_cn, accent)
    return np.vstack((bar, frame))


# ---- 四种合成手势 ----


def _hands_holding(timestamp: float) -> list[dict]:
    """慢速画圈 — 相干涡旋。"""
    angle = timestamp * 0.65
    return [
        {
            "id_hint": "Left",
            "palm_center": {
                "x": 0.42 + np.sin(angle) * 0.030,
                "y": 0.52 + np.cos(angle) * 0.025,
                "z": np.sin(angle * 0.5) * 0.004,
            },
            "landmarks": [],
        },
    ]


def _hands_dispersing(timestamp: float) -> list[dict]:
    """快速来回 — 解旋破碎。"""
    t = timestamp
    x = 0.38 + np.sin(t * 5.0) * 0.20 + np.cos(t * 3.7) * 0.06
    y = 0.50 + np.cos(t * 5.5) * 0.18 + np.sin(t * 4.2) * 0.05
    return [
        {
            "id_hint": "Right",
            "palm_center": {"x": x, "y": y, "z": np.sin(t * 7.0) * 0.025},
            "landmarks": [],
        },
    ]


def _hands_echo(timestamp: float) -> list[dict]:
    """前 1.8 秒有手，随后撤手展示余涡。"""
    if timestamp < 1.8:
        angle = timestamp * 0.55
        return [
            {
                "id_hint": "Left",
                "palm_center": {
                    "x": 0.40 + np.cos(angle) * 0.040,
                    "y": 0.50 + np.sin(angle) * 0.035,
                    "z": 0.0,
                },
                "landmarks": [],
            },
        ]
    return []


def _hands_dual(timestamp: float) -> list[dict]:
    """双手靠近互动 — 流桥效应。"""
    angle = timestamp * 0.48
    # 双手在画布中央靠近，绕小圈
    lx = 0.44 + np.sin(angle) * 0.040
    ly = 0.50 + np.cos(angle * 0.7) * 0.032
    rx = 0.56 - np.sin(angle) * 0.040
    ry = 0.50 - np.cos(angle * 0.7) * 0.032
    return [
        {"id_hint": "Left", "palm_center": {"x": lx, "y": ly, "z": 0.002}, "landmarks": []},
        {"id_hint": "Right", "palm_center": {"x": rx, "y": ry, "z": -0.002}, "landmarks": []},
    ]


QUAD_CONFIGS = [
    (_hands_holding, "holding", 1),
    (_hands_dispersing, "dispersing", 2),
    (_hands_echo, "echo", 3),
    (_hands_dual, "dual", 4),
]


def _assemble_grid(top_left, top_right, bottom_left, bottom_right):
    """拼合四个象限 + 标题条。"""
    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    grid = np.vstack((top, bottom))

    # 顶部标题
    title_h = 68
    header = np.full((title_h, CANVAS_W * 2, 3), BG_BGR, dtype=np.uint8)
    cv2.line(header, (0, title_h - 1), (CANVAS_W * 2, title_h - 1), MUTED_BGR, 1)
    cv2.putText(
        header,
        "YAN GUA  ·  TWIN VORTEX FIELD",
        (22, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        PRIMARY_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "演卦 · 双生涡场  —  四象限效果展示",
        (22, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        MUTED_BGR,
        1,
        cv2.LINE_AA,
    )

    return np.vstack((header, grid))


def parse_args():
    parser = argparse.ArgumentParser(description="生成演卦四象限效果展示图。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/yan-gua-quad-demo.png"),
        help="输出路径（默认：docs/assets/yan-gua-quad-demo.png）",
    )
    parser.add_argument(
        "--arch",
        choices=TAICHI_ARCH_CHOICES,
        default="auto",
        help="Taichi 后端（默认：auto；Apple Silicon 优先使用 Metal）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    arch = resolve_taichi_arch(args.arch, ti)

    frames = {}
    for hand_gen, key, seed in QUAD_CONFIGS:
        print(f"渲染 {QUAD_NAMES[key][0]} ...", end=" ", flush=True)
        frames[key] = _simulate_and_render(hand_gen, key, arch, seed)
        print("OK")

    result = _assemble_grid(
        frames["holding"],
        frames["dispersing"],
        frames["echo"],
        frames["dual"],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise RuntimeError(f"无法写入: {args.output}")
    print(f"\n四象限展示图: {args.output}")


if __name__ == "__main__":
    main()
