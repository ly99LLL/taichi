# AGENTS.md

This file provides guidance to Codex when working in this repository.

## 项目概述

**演卦 (YanGua / Twin Vortex Field)** — 双手驱动的实时星尘涡旋系统。慢手维持相干
涡环，快手使涡环破碎；检测丢失后保留连续衰减的余涡。用户文档见 `README.md`。

## 常用命令

```bash
python -m yan_gua
python -m yan_gua --arch cpu
python scripts/render_video.py
python scripts/render_demo.py --arch cpu

python -m pytest tests/ -m "not cuda" -v
python -m pytest tests/test_physics.py -v
python -m pytest tests/test_vortex.py -v
ruff check .
ruff format --check .

python -c "from yan_gua.vortex import VortexController"
python -c "import taichi as ti; ti.init(arch=ti.cuda); from yan_gua.physics import CloudParticles"
```

Git push 在部分网络环境需要：

```bash
git config --global http.version HTTP/1.1
git push origin master
```

## 架构

```text
摄像头 → CLAHE → MediaPipe Hands / Pose
                         ↓
               MotionAnalyzer
          固定身份槽 + 速度/曲率/纵深
                         ↓
               VortexController
       forming / holding / dispersing / echo
                         ↓
          Taichi Kernel (7200 粒子)
     常驻尘场 / 双涡环 / 解束 / 弹性碰撞
                         ↓
          py5 或 OpenCV 离线渲染
```

模块依赖：

```text
config
  ├── motion
  ├── vortex
  ├── tracking
  ├── camera_renderer
  ├── physics (taichi；py5 仅在 draw() 延迟导入)
  └── sketch (上述模块 + py5)
        └── __main__
```

## 不可破坏的语义

1. 粒子常驻。手出现/消失只改变粒子的组织度和可见度，不能在手的位置批量生成粒子。
2. `MotionAnalyzer.process()` 始终返回两个身份槽；`observed` 表示当前帧真实观测，
   `hand_detected` 只用于短 UI 迟滞。
3. 涡旋生命周期由 `VortexController` 管理。缺手必须经历 `echo`，不得直接关闭场。
4. 两个槽位旋向相反，slot 0 为 `+1`，slot 1 为 `-1`。
5. 实时与离线渲染必须复用 `MotionAnalyzer`、`VortexController` 和 `CloudParticles`。

## Taichi 约束

1. Kernel 必须在主线程预热后再调用 `py5.run_sketch()`。
2. `physics.py` 不能使用 `from __future__ import annotations`，Taichi 需要读取真实类型注解。
3. `ti.static` 循环内不使用 `continue`；用 `if hactive_arr[h] == 1` 包裹逻辑。
4. `@ti.func` 不写 early return。
5. NumPy 数组通过 `ti.types.ndarray()` 传入；修改 kernel 参数时同步更新
   `CloudParticles.update()`、`__main__._warmup()` 和 `tests/test_physics.py`。

## py5 约束

1. `settings/setup/draw` 等回调必须导入到调用 `py5.run_sketch()` 的模块级命名空间。
2. 自适应窗口尺寸在 `settings()` 中设置。
3. `physics.py` 中继续延迟导入 py5，保证 CPU 测试不启动 JVM。
4. `WINDOW_W/WINDOW_H` 运行时由 `sketch.settings()` 覆盖并回写 `config`。

## 视觉约束

- 背景近黑，稳定涡旋旧金，破碎冷蓝，余涡蓝灰。
- UI 采用单线、低对比、无阴影；摄像头保留原彩，只允许在手部附近叠加识别光效。
- 左右手识别使用旧金/月青双色、指尖光点和掌心断环，不得对整幅视频去色。
- 掌心保持安静，不恢复大面积手部光晕。
- 主要调参：`VORTEX_ORBIT_RADIUS`、`VORTEX_SLOW_SPEED`、
  `VORTEX_BREAK_SPEED`、`VORTEX_ECHO_SECONDS`、`TRAIL_ALPHA`。
