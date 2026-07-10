# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**演卦 (YanGua / Twin Vortex Field)** — 双手驱动的实时星尘涡旋系统。
慢手维持相干涡环，快手使涡环破碎；检测丢失后保留连续衰减的余涡。

完整架构与约束记录在 [`AGENTS.md`](AGENTS.md)，用户文档见 [`README.md`](README.md)。

## 常用命令

```bash
# 实时运行
python -m yan_gua                         # 自动 GPU
python -m yan_gua --arch cpu              # CPU 模式
python -m yan_gua --arch auto             # 自动检测

# 离线渲染
python scripts/render_video.py 输入.mp4 输出.mp4
python scripts/render_demo.py --arch cpu

# 测试（默认跳过 CUDA）
python -m pytest tests/ -m "not cuda" -v
python -m pytest tests/test_physics.py -v
python -m pytest tests/test_vortex.py -v

# 静态检查
ruff check .
ruff format --check .
```

## 不可破坏的语义

1. 粒子常驻。手出现/消失只改变粒子的组织度和可见度，不能批量生成粒子。
2. `MotionAnalyzer.process()` 始终返回两个身份槽；`observed` 表示真实观测，`hand_detected` 只用于 UI 迟滞。
3. 涡旋生命周期：缺手必须经历 `echo`，不得直接关闭场。
4. 两个槽位旋向相反：slot 0 = `+1`，slot 1 = `-1`。
5. 实时与离线必须复用 `MotionAnalyzer`、`VortexController` 和 `CloudParticles`。

## Taichi 约束

- Kernel 必须在主线程预热后再调用 `py5.run_sketch()`。
- `physics.py` 不能使用 `from __future__ import annotations`。
- `ti.static` 循环内不使用 `continue`；用 `if hactive_arr[h] == 1` 包裹。
- `@ti.func` 不写 early return。
- 修改 kernel 参数时同步更新 `CloudParticles.update()`、`__main__._warmup()` 和测试。

## py5 约束

- settings/setup/draw 等回调必须在调用 `py5.run_sketch()` 的模块级命名空间。
- `physics.py` 中延迟导入 py5，保证 CPU 测试不启动 JVM。
- 窗口尺寸由 `sketch.settings()` 自适应并回写 `config`。
