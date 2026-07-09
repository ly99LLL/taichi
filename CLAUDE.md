# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**演卦 (YanGua / Stardust Taichi)** — 太极手势驱动的星尘粒子实时交互系统。不识别具体招式，而是读取手势"运动气质"（手速/曲直/纵深）并翻译为粒子物理响应。美学方向：暗宇宙星尘场 + 琥珀搅动 + 旋转星云 + 劲断意不断。

设计理念详见 `项目简介.txt`。

## 文件地图

```
CLAUDE.md           # 本文件
yan_gua.py          # ★ 唯一主程序: py5实时交互 + Taichi GPU物理 + MediaPipe
render_video.py     # 辅助工具: 视频→效果视频 (OpenCV CPU, 离线渲染)
main.py             # 旧版: Pygame + moderngl OpenGL 3D (已废弃)
run.bat             # 启动脚本 (设置JAVA_HOME + 运行yan_gua.py)
requirements.txt    # Python依赖
项目简介.txt         # 项目设计理念 (中文)
```

## 运行方式

```bash
pip install -r requirements.txt

# 启动 (需要 JDK 17)
run.bat                    # Windows双击, 自动设JAVA_HOME
# 或: python yan_gua.py

# 视频离线渲染 (辅助)
python render_video.py
```

快捷键: `ESC` 退出 | `F` 全屏 | `D` 调试

## 架构总览

```
摄像头(1280×720) → CLAHE增强 → MediaPipe Hands + Pose → MotionAnalyzer (双手)
                                                              ↓
                                              Taichi CUDA Kernel (GPU粒子物理)
                                                              ↓
                                              py5 OpenGL (粒子渲染 + 拖尾)
                                                              ↓
                                              CameraRenderer (右下角水墨小窗)
```

### 检测策略 (双轨)

1. **MediaPipe Hands** (21点手指) — 优先。手够大时用完整手指关键点
2. **MediaPipe Pose** (33点全身骨架) — 降级。手太小时用腕关节(landmark 15/16)作为手掌位置
3. `HandTracker.read()` 返回 `(frame, hands_list, pose_landmarks)` — 双手优先从Hands取，检测不到时从Pose腕关节补

### CLAHE 预处理

摄像头帧在送MediaPipe前经过CLAHE(对比度受限自适应直方图均衡化)：LAB色彩空间L通道均衡 → 提升远距离手部检出率约20-27%。

### 核心模块

| 模块 | 位置 | 职责 |
|------|------|------|
| `HandTracker` | yan_gua.py | 1280×720摄像头 + CLAHE + MediaPipe Hands+Pose |
| `MotionAnalyzer` | yan_gua.py | 双手独立追踪，提取 speed/curvature/z_velocity (EMA平滑) |
| `_ti_particle_physics` | yan_gua.py | **Taichi CUDA kernel** — GPU并行6000粒子物理 |
| `CloudParticles` | yan_gua.py | 粒子数据管理 + py5渲染 |
| `CameraRenderer` | yan_gua.py | 右下角小窗：水墨滤镜 + Pose骨架笔触 + 手部关键点 |

### 粒子物理 — 三层空间结构

```
手掌中心
  ← 0-25% 半径: 中空区 (粒子被猛推向外, 不回吸)
  ← 25-40% 半径: 环壁 (轻吸维持边界, 粒子堆积成可见漩涡环)
  ← 40-100% 半径: 外层 (粘性拖拽 + 旋涡 + 飞溅 + 呼吸)
  ← >100%: 不受影响 (缓慢回归基态)
```

### 六大物理力 (GPU kernel中按顺序施加)

| 力 | 方向 | 公式 | 太极慢速效果 |
|----|------|------|-------------|
| **中空推力** | 径向向外 | 8.0 + visc×6.0 | 强推 (≈13.5) |
| **环壁拉力** | 径向向内 | 0.8 + visc×1.5 | 轻吸 (≈2.2) |
| **粘性拖拽** | 跟随手速 | hv × visc × 0.08 | 粒子跟随手流动 |
| **飞溅** | 径向向外 | norm_spd × 2.5 | 快=迸裂 |
| **基线漩涡** | 切向 | visc × 3.0 | 手在就有旋转 |
| **曲率漩涡** | 切向 | curv × (3.0 + visc×10.0) | 画弧时叠加 |
| **呼吸** | 径向 | z_vel × 0.6 | 前推膨胀/后拉收缩 |

关键公式: `viscosity = 1.0 - speed/MAX_SPEED`, `norm_speed = speed/MAX_SPEED`

### 调色板 (深空星尘)

- 背景: RGB(16,12,8) 深空暗墨
- 粒子六阶: (180,160,130) → (140,120,95) → (105,85,65) → (75,58,42) → (48,35,24) → (28,18,12)
- 暖金: WARM_ACCENT=(200,155,90), WARM_LIGHT=(240,210,150)

### 关键常量

| 常量 | 值 | 说明 |
|------|-----|------|
| PARTICLE_COUNT | 6000 | Taichi GPU加速后提升 |
| INFLUENCE_RADIUS | 240 | 手掌影响范围 (1.5x扩大) |
| MAX_SPEED | 800 | |
| CURVATURE_REF | 400 | 曲率灵敏度 |
| SMOOTH_ALPHA | 0.35 | EMA平滑系数 |
| HISTORY_SIZE | 45 | |
| BASE_DAMPING | 0.985 | |
| TRAIL_ALPHA | 7 | 拖尾消退速度 (值越小拖尾越长) |
| PARTICLE_ALPHA_MAX | 22 | |
| PARTICLE_SIZE_MAX | 32 | |
| Hands model_complexity | 1 | 精度/速度平衡最优 |
| Hands tracking_confidence | 0.4 | 减少抖动 |
| Pose tracking_confidence | 0.4 | 减少抖动 |
| 摄像头分辨率 | 1280×720 | 远距离手部检出关键 |

## 依赖

```
opencv-python, mediapipe, numpy, taichi>=1.7    (所有版本)
py5>=0.10, JDK 17                               (主程序)
```
