# 演卦 · 双生涡场

[![CI](https://github.com/ly99LLL/taichi/actions/workflows/ci.yml/badge.svg)](https://github.com/ly99LLL/taichi/actions/workflows/ci.yml)
[![Python 3.11–3.12](https://img.shields.io/badge/Python-3.11%E2%80%933.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M5%2B-C9A96E?logo=apple&logoColor=white)](#apple-silicon-与-m5)
[![License: MIT](https://img.shields.io/badge/License-MIT-C9A96E.svg)](LICENSE)

**演卦（YanGua / Twin Vortex Field）** 是一个由双手驱动的实时星尘涡旋系统。
慢手让尘埃聚成相干涡环，快手使涡环破碎；手离开后，场中仍会留下连续衰减的余涡。

![无摄像头输入的双生涡场合成演示](docs/assets/demo.png)

> 图片由内置合成轨迹生成，不读取摄像头，也不含真人影像。粒子始终常驻，手只改变
> 它们的组织度、速度和可见度。

## 核心特性

- 双手身份槽稳定，不因 MediaPipe 返回顺序变化而交换涡旋。
- 统一管理 `forming → holding → dispersing → echo` 生命周期。
- 7200 个粒子常驻，不在手掌位置批量生成或销毁。
- 两个槽位反向旋转，接近时产生排斥、擦边和粒子甩散。
- 实时、视频和合成模式复用同一套分析、控制与 Taichi 物理逻辑。
- 摄像头默认只在本机内存中处理，不包含上传或遥测功能。

## 系统兼容性

| 平台 | 推荐后端 | 启动参数 | 说明 |
|---|---|---|---|
| Apple Silicon Mac（M1–M5 及更新） | Metal | `--arch auto` 或 `--arch metal` | 原生 arm64，推荐 |
| Intel Mac | 自动 / CPU | `--arch auto` 或 `--arch cpu` | 性能取决于机型与驱动 |
| Windows / Linux + NVIDIA | CUDA | `--arch auto` 或 `--arch cuda` | 推荐 |
| Windows / Linux + Intel / AMD GPU | Vulkan | `--arch auto` 或 `--arch vulkan` | 取决于显卡驱动 |
| 无可用 GPU / 虚拟机 | CPU | `--arch cpu` | 功能完整，帧率较低 |

运行要求：

- 64 位 CPython 3.11 或 3.12；
- JDK 17 或更高版本，推荐 JDK 21；
- Python 与 Java 必须使用相同处理器架构；
- 摄像头可选，视频输入和合成演示不依赖摄像头。

项目支持 `auto`、`cpu`、`cuda`、`metal`、`vulkan` 和 `opengl`，所有实时与离线入口
使用相同的后端解析规则。`auto` 会选择可用 GPU，失败时由 Taichi 回退到 CPU。

兼容性依据见 [Taichi 支持的系统与后端](https://docs.taichi-lang.org/docs/hello_world)和
[MediaPipe macOS arm64 支持说明](https://ai.google.dev/edge/mediapipe/framework/getting_started/troubleshooting?hl=zh-cn)。

### Apple Silicon 与 M5

**M5 可以使用。** 项目按 `macOS + arm64 + Metal` 判断兼容性，不按 M1、M2、M3 等
营销型号建立白名单，因此后续写成 M5 也不需要单独增加芯片判断。旧 README 的“M1–M3”
只是过时的文档范围，不是代码限制。

在 M5 上建议先确认原生架构：

```bash
uname -m
python3 -c "import platform; print(platform.machine())"
```

两条命令都应显示 `arm64`。如果 Python 显示 `x86_64`，说明正在使用 Rosetta 版本；请安装
arm64 Python 后重新创建虚拟环境。程序启动时还会检查 Java 与 Python 架构，避免混用
x86_64 JDK 和 arm64 Python。

## 快速开始

```bash
git clone https://github.com/ly99LLL/taichi.git
cd taichi
python3 -m venv .venv
```

激活环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

安装并启动：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m yan_gua
```

Apple Silicon 也可以显式指定 Metal：

```bash
python -m yan_gua --arch metal
```

Windows 可双击 `run.bat`；macOS / Linux 也可运行 `bash run.sh`。首次使用摄像头时，
请允许终端或 IDE 访问摄像头。

## 使用方式

```bash
# 实时摄像头
python -m yan_gua

# 使用视频文件；默认按摄像头视角水平镜像
python -m yan_gua --video "输入视频.mp4"

# 输入已经镜像
python -m yan_gua --video "输入视频.mp4" --no-mirror

# 录制主窗口（不含音频）
python -m yan_gua --video "输入视频.mp4" --record "输出视频.mp4"

# 离线渲染，不在结果中显示原视频小窗
python scripts/render_video.py "输入视频.mp4" "效果视频.mp4" --no-camera

# 生成不含真人的合成演示
python scripts/render_demo.py
```

以上命令均可附加 `--arch metal`、`--arch cuda`、`--arch vulkan` 或 `--arch cpu`。
使用 `--help` 查看每个入口的完整参数。

### 操作

| 按键 / 控件 | 功能 |
|---|---|
| `ESC` | 退出 |
| `F` | 切换全屏 |
| `D` | 显示相干性、破碎度和生命周期 |
| 右上角按钮 | 重置粒子场 |

## 工作流程

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
   Taichi Kernel（7200 常驻粒子）
          ├─ py5：实时窗口
          └─ OpenCV：离线与合成渲染
```

`MotionAnalyzer.process()` 始终返回两个身份槽；`observed` 表示当前帧真实观测，
`hand_detected` 仅用于短暂 UI 迟滞。缺手必须经过 `echo`，不会直接关闭物理场。

## 项目结构

```text
yan_gua/        追踪、运动分析、涡场生命周期、物理与渲染
scripts/        视频离线渲染和无真人合成演示
tests/          CPU 测试与可选 CUDA 集成测试
docs/assets/    已确认可公开、无人物的 README 素材
pyproject.toml  包元数据、依赖和开发工具配置
```

主要视觉参数位于 `yan_gua/config.py`，常用项包括 `VORTEX_ORBIT_RADIUS`、
`VORTEX_SLOW_SPEED`、`VORTEX_BREAK_SPEED`、`VORTEX_ECHO_SECONDS` 和 `TRAIL_ALPHA`。

## 开发与测试

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest tests/ -m "not cuda" -v
```

CUDA 硬件测试需显式设置 `YANGUA_RUN_CUDA_TESTS=1` 后运行
`tests/test_physics_cuda.py`。CI 在 Python 3.11 和 3.12 上运行静态检查、格式检查和
非 CUDA 测试。

## 隐私与本地文件

- 摄像头帧默认仅在本机进程内处理，只有显式使用 `--record` 才会写入视频。
- 原始视频、录制结果、`artifacts/` 和本地 `presentation/` 展示网页均被 Git 忽略。
- `docs/assets/` 只应存放已确认可公开、无人物、无个人信息的演示素材。
- 提交前仍应检查文件元数据、绝对路径、邮箱、令牌、密钥和第三方肖像。

## 已知限制

- CPU 模式功能完整，但 7200 粒子的实时帧率明显低于 GPU 模式。
- Metal、Vulkan 和 CUDA 的可用性最终取决于 Taichi、操作系统和显卡驱动。
- MediaPipe 识别会受到遮挡、逆光、手部尺寸和摄像头帧率影响。
- MP4 录制不保留输入音频，需要在本地后期合并。

参与贡献前请阅读[贡献指南](CONTRIBUTING.md)和[行为准则](CODE_OF_CONDUCT.md)。
安全问题请按[安全政策](SECURITY.md)私下报告，版本变化见[更新日志](CHANGELOG.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。
