# 演卦 · 双生涡场

[![CI](https://github.com/ly99LLL/taichi/actions/workflows/ci.yml/badge.svg)](https://github.com/ly99LLL/taichi/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11%E2%80%933.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-C9A96E.svg)](LICENSE)

**演卦（YanGua / Twin Vortex Field）** 是一个由双手驱动的实时星尘涡旋系统。
它将手部速度、轨迹、曲率与纵深变化映射为具有生命周期的双生流场：慢手维持相干
涡环，快手使涡环解旋破碎，手离开后仍会留下连续衰减的余涡。

![无摄像头输入的双生涡场合成演示](docs/assets/demo.png)

> 演示图由内置的确定性合成轨迹生成，不读取摄像头，也不包含真人影像。

## 功能概览

- **稳定双手身份**：固定维护两个身份槽，不因检测顺序变化而交换涡旋。
- **连续生命周期**：统一管理成核、保持、解旋、余涡和重新成核。
- **常驻粒子系统**：7200 个粒子始终存在，手势只改变其组织度、速度和可见度。
- **双涡场互动**：两个槽位旋向相反，接近时产生排斥、擦边和粒子甩散。
- **多种输入模式**：支持实时摄像头、视频文件和无真人合成轨迹。
- **统一物理实现**：实时与离线渲染复用相同的分析、控制和 Taichi 粒子逻辑。
- **本地隐私处理**：摄像头帧默认只在本机内存中处理，不包含上传或遥测功能。

## 交互模型

| 输入行为 | 粒子响应 |
|---|---|
| 缓慢移动或停留 | 尘埃逐渐锁定到掌心外缘的空心轨道，形成稳定涡环 |
| 快速移动 | 相干性下降，环流转为径向剪切，粒子向外解束 |
| 突然停止 | 粒子短暂继承手部动量，产生惯性泼洒 |
| 手暂时丢失 | 最后位置保留余涡，沿惯性滑行、扩张并衰减 |
| 手重新出现 | 低亮度尘场渐进成核，不瞬间生成粒子团 |
| 双手靠近 | 两个反向涡场在接触区排斥并擦边滑移 |

## 环境要求

| 组件 | 要求 |
|---|---|
| Python | 64 位 CPython 3.11 或 3.12 |
| Java | JDK 17 或更高版本；推荐 JDK 21 |
| GPU | 可选；支持 Metal、CUDA、Vulkan、OpenGL 和 CPU 后端 |
| 摄像头 | 可选；视频文件与合成演示模式不需要摄像头 |

Python 与 Java 必须使用相同的处理器架构。程序启动时会自动检查版本与架构，避免混用
arm64 和 x86_64 运行时。

### 计算后端

| 运行环境 | 推荐后端 | 参数 |
|---|---|---|
| macOS / arm64 | Metal | `--arch auto` 或 `--arch metal` |
| macOS / x86_64 | 自动或 CPU | `--arch auto` 或 `--arch cpu` |
| Windows / Linux + NVIDIA GPU | CUDA | `--arch auto` 或 `--arch cuda` |
| Windows / Linux + Intel / AMD GPU | Vulkan | `--arch auto` 或 `--arch vulkan` |
| 无可用 GPU 或虚拟机 | CPU | `--arch cpu` |

`--arch auto` 是默认选项：Taichi 会选择可用的 GPU 后端，并在 GPU 不可用时回退到
CPU。实时入口和全部离线脚本支持相同的 `auto`、`cpu`、`cuda`、`metal`、`vulkan`、
`opengl` 选项。

兼容性参考：[Taichi 支持的系统与后端](https://docs.taichi-lang.org/docs/hello_world)、
[MediaPipe 支持的平台](https://ai.google.dev/edge/mediapipe/framework/getting_started/troubleshooting?hl=zh-cn)和
[py5 安装要求](https://py5coding.org/content/install.html)。

## 安装

克隆项目并创建虚拟环境：

```bash
git clone https://github.com/ly99LLL/taichi.git
cd taichi
python -m venv .venv
```

激活虚拟环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

如果 macOS 中 `python -c "import platform; print(platform.machine())"` 输出 `x86_64`，
而系统使用 arm64，请安装原生 arm64 Python 后重新创建虚拟环境。Java 也应安装对应架构的
JDK。

## 使用

### 实时摄像头

```bash
python -m yan_gua
```

首次运行时，系统可能要求允许终端或 IDE 访问摄像头。Windows 也可双击 `run.bat`；
macOS / Linux 可运行 `bash run.sh`。

### 视频输入与录制

```bash
# 使用视频文件替代摄像头
python -m yan_gua --video "输入视频.mp4"

# 输入视频已经水平镜像
python -m yan_gua --video "输入视频.mp4" --no-mirror

# 录制主窗口；输出不包含音频
python -m yan_gua --video "输入视频.mp4" --record "输出视频.mp4"
```

### 离线渲染

```bash
# 输出粒子场并隐藏原视频小窗
python scripts/render_video.py "输入视频.mp4" "效果视频.mp4" --no-camera

# 生成不读取摄像头、不包含真人的合成演示
python scripts/render_demo.py
```

所有入口都可使用 `--arch` 指定后端，例如：

```bash
python -m yan_gua --arch metal
python scripts/render_demo.py --arch cpu
```

使用 `--help` 查看完整参数。

### 操作

| 按键 / 控件 | 功能 |
|---|---|
| `ESC` | 退出 |
| `F` | 切换全屏 |
| `D` | 显示相干性、破碎度和生命周期 |
| 右上角按钮 | 重置粒子场 |

## 系统架构

```text
摄像头 / 视频 / 合成轨迹
          │
          ├─ 图像输入 → CLAHE → MediaPipe Hands / Pose
          ▼
   MotionAnalyzer
   固定身份槽 + 速度 / 曲率 / 纵深
          ▼
   VortexController
   forming / holding / dispersing / echo
          ▼
   Taichi Kernel（7200 个常驻粒子）
          ├─ py5：实时窗口
          └─ OpenCV：离线与合成渲染
```

`MotionAnalyzer.process()` 始终返回两个身份槽；`observed` 表示当前帧存在真实观测，
`hand_detected` 仅用于短暂的 UI 迟滞。缺手状态必须经过 `echo`，不会直接关闭物理场。

## 项目结构

```text
yan_gua/        追踪、运动分析、涡场生命周期、物理与渲染
scripts/        视频离线渲染和无真人合成演示
tests/          CPU 测试与可选 CUDA 集成测试
docs/assets/    已确认可公开、无人物的 README 素材
pyproject.toml  包元数据、依赖和开发工具配置
```

## 配置

主要参数集中在 `yan_gua/config.py`：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `PARTICLE_COUNT` | 7200 | 常驻粒子数量 |
| `VORTEX_ORBIT_RADIUS` | 92 | 稳定涡环半径 |
| `VORTEX_SLOW_SPEED` | 135 | 开始降低相干性的速度 |
| `VORTEX_BREAK_SPEED` | 480 | 完全解旋的速度 |
| `VORTEX_ECHO_SECONDS` | 2.4 | 余涡衰减时间尺度 |
| `TRAIL_ALPHA` | 24 | 帧缓冲拖尾衰减 |

两个身份槽的旋向固定相反：slot 0 为 `+1`，slot 1 为 `-1`。

## 开发与测试

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest tests/ -m "not cuda" -v
```

CUDA 硬件测试默认关闭。显式设置 `YANGUA_RUN_CUDA_TESTS=1` 后，可运行：

```bash
python -m pytest tests/test_physics_cuda.py -m cuda -v
```

CI 在 Python 3.11 和 3.12 上执行静态检查、格式检查、非 CUDA 测试和覆盖率统计。

## 隐私与媒体文件

- 摄像头帧默认仅在本机进程内处理，只有显式使用 `--record` 才会写入视频。
- 原始视频、录制结果、逐帧调试图、`artifacts/` 和本地 `presentation/` 均由
  `.gitignore` 排除。
- `docs/assets/` 只应存放已确认可公开、无人物、无个人信息的演示素材。
- 提交前应检查文件元数据、绝对路径、邮箱、访问令牌、密钥和第三方肖像。

## 已知限制

- CPU 模式功能完整，但实时帧率通常低于 GPU 模式。
- GPU 后端的可用性和性能取决于 Taichi、操作系统及显卡驱动。
- MediaPipe 识别会受到遮挡、逆光、手部尺寸和摄像头帧率影响。
- MP4 录制不保留输入音频，需要在本地后期合并。

## 参与项目

提交改动前请阅读[贡献指南](CONTRIBUTING.md)和[行为准则](CODE_OF_CONDUCT.md)。安全问题请按
[安全政策](SECURITY.md)私下报告；版本变化见[更新日志](CHANGELOG.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。
