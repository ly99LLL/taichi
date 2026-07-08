# 墨·境 / Taichi Bagua

一个用 Python 实现的太极动作粒子视觉实验。项目通过鼠标、摄像头或离线视频读取动作轨迹，把速度、稳定性和停顿映射成水墨粒子的排斥、吸引、环绕与绽放。

> 隐私说明：仓库只保存源码、姿态模板和公开文档。原始视频、渲染成片、临时输出和本地工作记录都已被 `.gitignore` 排除，不应该提交到 GitHub。

## 功能

- 鼠标模式：无需摄像头即可演示粒子力场。
- 摄像头模式：使用 MediaPipe Pose 追踪左右手腕，摄像头不可用时自动回退到鼠标模式。
- 预览模式：离屏生成四阶段预览图，方便快速确认可运行。
- 离线渲染：读取本地视频，叠加水墨粒子与招式触发效果。
- 姿态模板：内置 24 式太极参考姿态 JSON 与裁剪图，用于动作匹配实验。

## 环境要求

- Python 3.11 推荐
- Windows、macOS 或 Linux 桌面环境
- 摄像头模式需要本机摄像头
- 离线视频保留音频时需要安装 FFmpeg，并确保 `ffmpeg` / `ffprobe` 在 PATH 中

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

生成一张预览图：

```powershell
python main.py --preview --particles 800 --frames 120 --output preview.png
```

运行鼠标演示：

```powershell
python main.py --input mouse --particles 3600
```

运行摄像头演示：

```powershell
python main.py --input camera --particles 3600
```

常用快捷键：

- `D`：开关调试面板
- `B`：手动触发一次绽放
- `M`：开关声音
- `Shift + 鼠标移动`：模拟左手辅助力场
- `Esc`：退出

## 离线视频渲染

视频素材请放在本地私有目录，例如 `private/`。该目录和所有常见视频格式已经被 Git 忽略。

```powershell
python render_ink_video.py --input private/input.mp4 --output outputs/ink_demo.mp4 --particles 3200
```

如果没有 FFmpeg，或不想合成原视频音频：

```powershell
python render_ink_video.py --input private/input.mp4 --output outputs/ink_demo.mp4 --particles 3200 --no-audio
```

## 项目结构

```text
.
├── main.py                    # 交互演示入口：mouse / camera / preview
├── render_ink_video.py        # 本地视频离线渲染工具
├── bagua/                     # 粒子、输入、渲染、音频与姿态匹配模块
├── reference_poses/           # 24 式参考姿态模板与裁剪图
├── requirements.txt           # 运行依赖
├── pyproject.toml             # 项目元数据与开发工具配置
├── PRIVACY.md                 # 隐私与发布检查
├── CONTRIBUTING.md            # 开发约定
└── LICENSE                    # MIT License
```

## 发布前检查

提交前建议执行：

```powershell
git status --short
git ls-files | rg -i "\.(mp4|mov|avi|mkv|webm|m4v)$"
rg -n -uu -i "api[_-]?key|secret|token|password|authorization|bearer|private key|client_secret" .
```

第二条命令应该没有任何输出；如果出现视频文件，先从 Git 暂存区移除再提交。

## 开发

运行轻量测试：

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

仅检查导入和模板加载：

```powershell
python -m pytest tests/test_smoke.py
```

## License

MIT License. See [LICENSE](LICENSE).
