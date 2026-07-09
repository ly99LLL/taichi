# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

**演卦 (YanGua / Stardust Taichi)** — 太极手势驱动的星尘粒子实时交互系统。读取手势"运动气质"（手速/曲直/纵深）翻译为粒子物理响应。设计理念详见 `项目简介.txt`，用户文档见 `README.md`。

## 常用命令

```bash
# 运行
python -m yan_gua                        # 主程序 (需要 JDK 17)
python scripts/render_video.py           # 视频离线渲染

# 测试
python -m pytest tests/ -m "not cuda" -v # 默认 CPU 测试
python -m pytest tests/test_physics.py -v # 单个测试文件
python -m pytest tests/ -k "test_cloud"  # 按名称筛选
ruff check .                             # 静态检查
ruff format --check .                    # 格式检查

# 单模块导入验证
python -c "from yan_gua.config import INFLUENCE_RADIUS"
python -c "import taichi as ti; ti.init(arch=ti.cuda); from yan_gua.physics import CloudParticles"

# Git (push 需要 HTTP/1.1)
git config --global http.version HTTP/1.1
git push origin master
```

## 架构

```
摄像头(1280×720) → CLAHE(LAB-L通道) → MediaPipe Hands → MotionAnalyzer
                     ↓                    ↓ (降级补位)        ↓
                  增强帧              MediaPipe Pose    speed/curvature
                                                       /z_velocity (EMA)
                                                            ↓
                                            Taichi CUDA Kernel (6000粒子并行)
                                                            ↓
                                            py5 OpenGL (alpha排序 + 拖尾)
                                                            ↓
                                            CameraRenderer (右下角水墨小窗)
```

### 模块依赖关系

```
config (无依赖)
  ├── motion (config)
  ├── tracking (config)
  ├── camera_renderer (config, py5)
  ├── physics (config, taichi)     ← py5 延迟导入, 仅 draw() 方法需要
  └── sketch (全部上述模块, py5)
        └── __main__ (physics, sketch, py5)
```

### 检测策略

1. **MediaPipe Hands** — 优先。`model_complexity=1`, detection/tracking confidence=0.4
2. **MediaPipe Pose** — 降级。手太小时腕关节 (landmark 15/16) 补位，visibility>0.4 生效
3. `HandTracker.read()` → `(bgr_frame, hands_list_or_None, pose_landmarks)`

## 关键设计约束 (只读代码看不出来)

### Taichi Kernel

1. **必须在主线程编译** — `__main__.py` 用伪数据预热 kernel，之后再 `py5.run_sketch()`
2. **`@ti.func` 和 `@ti.kernel` 不可拆分到不同文件** — Taichi 编译单元必须在同一 .py
3. **`ti.static` 循环内不能用 `continue`** — 不能用 `if hactive_arr[h] == 0: continue`，必须反转为 `if hactive_arr[h] == 1: {所有逻辑}`
4. **`@ti.func` 内不能 early return** — 不能 `if dist >= infl_r: return`，必须包成 `if dist < infl_r: {所有逻辑}`
5. **`ti.types.ndarray()` 零拷贝** — 自动 CPU↔GPU 传输，kernel 直接读写 NumPy 数组

### py5

1. **`run_sketch()` 检查调用帧的命名空间** — settings/setup/draw 等函数必须在调用 `py5.run_sketch()` 的模块级命名空间中（不能包在函数里），`__main__.py` 通过 `from yan_gua.sketch import ...` 导入到模块级
2. **`settings()` 先于 `setup()` 调用** — 自适应窗口大小的逻辑放在 `settings()` 中，用 `py5.size()`
3. **py5 用 JOGL (Java OpenGL)** — 与 Taichi CUDA 无冲突
4. **JVM 必须先启动** — `physics.py` 对 py5 做了延迟导入（`draw()` 方法内 `import py5`），这样测试环境不触发 JVM，避免 crash

### 全局状态

- `WINDOW_W`/`WINDOW_H` 默认值在 `config.py`，运行时由 `sketch.settings()` 覆盖并回写 `config`
- `MotionAnalyzer` 和 `CloudParticles` 在 `setup()` 中创建，传入运行时的窗口尺寸

## 粒子物理简记

三层空间：0-25% 中空推力 / 25-40% 环壁拉力 / 40-100% 外层（拖拽+飞溅+漩涡+呼吸）

七大力的公式见 `physics.py:_apply_hand_force`。核心：`viscosity = 1.0 - speed/MAX_SPEED`（慢=粘稠，快=稀薄）。

所有可调参数在 `yan_gua/config.py`。影响视觉最明显的：`TRAIL_ALPHA`（拖尾长度）、`INFLUENCE_RADIUS`（影响范围）、`PARTICLE_COUNT`（粒子数）。

## 依赖

```
py5>=0.10 (需 JDK 17)  opencv-python  mediapipe  numpy  taichi>=1.7
```
