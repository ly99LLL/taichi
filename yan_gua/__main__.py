"""演卦 · 双生涡场 — 入口模块.

用法:
    python -m yan_gua
    python -m yan_gua --arch cpu
    python -m yan_gua --video 参考视频.mp4 --record 主程序直录.mp4
"""

import argparse
from pathlib import Path

from yan_gua.runtime import ensure_java_17

# py5 导入前验证环境；保留用户已有的 JAVA_HOME。
ensure_java_17()

import numpy as np
import py5
import taichi as ti

from yan_gua.physics import _particle_physics_kernel

# py5 在调用 run_sketch() 的命名空间中查找 settings/setup/draw 等函数
# 因此必须导入到模块级命名空间 (不能放在函数内部)
from yan_gua.sketch import (
    configure_input,
    draw,
    exiting,
    key_pressed,
    mouse_pressed,
    settings,
    setup,
)

_ARCH_MAP = {
    "auto": ti.gpu,
    "cuda": ti.cuda,
    "cpu": ti.cpu,
    "vulkan": ti.vulkan,
    "metal": ti.metal,
    "opengl": ti.opengl,
}


def _resolve_arch(name: str):
    """将 --arch 名称解析为 Taichi 后端；无法使用时给出明确错误。"""
    backend = _ARCH_MAP.get(name)
    if backend is None:
        choices = ", ".join(sorted(_ARCH_MAP))
        raise SystemExit(f"不支持的后端: {name}。可用选项: {choices}")

    label = name
    if name == "auto":
        try:
            backend = ti.gpu
            label = "gpu (auto)"
        except Exception:
            backend = ti.cpu
            label = "cpu (GPU 不可用，已自动退回)"

    print(f"Taichi 后端: {label}")
    return backend


def _warmup():
    """kernel 预热 — 必须在主线程、py5.run_sketch() 之前完成。

    Taichi kernel 编译发生在首次调用时, 必须在主线程完成。
    py5 的渲染线程不是主线程, 所以需要提前编译。
    """
    print("Kernel warmup...", end=" ", flush=True)
    _n_warm = 100
    _warm_kwargs = dict(
        px=np.random.uniform(0, 1280, _n_warm).astype(np.float32),
        py=np.random.uniform(0, 720, _n_warm).astype(np.float32),
        vx=np.zeros(_n_warm, dtype=np.float32),
        vy=np.zeros(_n_warm, dtype=np.float32),
        alpha=np.full(_n_warm, 5.0, dtype=np.float32),
        radius=np.full(_n_warm, 10.0, dtype=np.float32),
        ink_level=np.zeros(_n_warm, dtype=np.int32),
        base_alpha=np.full(_n_warm, 5.0, dtype=np.float32),
        base_radius=np.full(_n_warm, 10.0, dtype=np.float32),
        base_ink=np.zeros(_n_warm, dtype=np.int32),
        hx_arr=np.array([640.0, 0.0], dtype=np.float32),
        hy_arr=np.array([360.0, 0.0], dtype=np.float32),
        hvx_arr=np.zeros(2, dtype=np.float32),
        hvy_arr=np.zeros(2, dtype=np.float32),
        hstrength_arr=np.array([1.0, 0.0], dtype=np.float32),
        hcoherence_arr=np.array([1.0, 0.0], dtype=np.float32),
        hscatter_arr=np.zeros(2, dtype=np.float32),
        hrelease_arr=np.zeros(2, dtype=np.float32),
        hmaturity_arr=np.array([1.0, 0.0], dtype=np.float32),
        haperture_arr=np.zeros(2, dtype=np.float32),
        hsplash_arr=np.zeros(2, dtype=np.float32),
        hspin_arr=np.array([1.0, -1.0], dtype=np.float32),
        hactive_arr=np.array([1, 0], dtype=np.int32),
        dt=0.016,
        win_w=1280.0,
        win_h=720.0,
        infl_r=320.0,
        orbit_r=92.0,
        base_damp=0.973,
    )
    _particle_physics_kernel(**_warm_kwargs)
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="演卦 · 双生涡场")
    parser.add_argument(
        "--arch",
        choices=sorted(_ARCH_MAP),
        default="auto",
        help="Taichi 计算后端（默认：auto 自动选择 GPU，不可用时退回 CPU）",
    )
    parser.add_argument(
        "--video",
        metavar="PATH",
        help="用视频文件替代摄像头输入，并按视频原始帧率播放",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="把 py5 主窗口逐帧录制为 MP4（不含音频）",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="视频已经是镜像画面时，不再模拟摄像头的水平镜像",
    )
    args = parser.parse_args()

    if args.video and not Path(args.video).is_file():
        parser.error(f"找不到输入视频: {args.video}")
    if args.record and not args.video:
        parser.error("--record 目前需要与 --video 一起使用")

    # 解析并初始化 Taichi 后端（必须早于 kernel 调用和 py5 渲染线程）。
    arch = _resolve_arch(args.arch)
    ti.init(arch=arch, random_seed=42)

    configure_input(args.video, args.record, mirror_video=not args.no_mirror)
    _warmup()
    # py5.run_sketch() 必须在模块级调用 (不是函数内),
    # 因为它检查调用帧的命名空间来查找 sketch 函数
    py5.run_sketch()
