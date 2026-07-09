# 演卦 · 星尘太极

太极手势驱动的星尘粒子实时交互系统。不识别具体招式，读取手势"运动气质"（手速/曲直/纵深）并翻译为粒子物理响应。

美学方向：暗宇宙星尘场 + 琥珀搅动 + 旋转星云 + 劲断意不断。

## 快速开始

```bash
pip install -r requirements.txt
python -m yan_gua     # 实时交互
# 或双击 run.bat
```

**快捷键**: `ESC` 退出 | `F` 全屏 | `D` 调试

## 项目结构

```
yan_gua/                 # 主包
├── __main__.py          # 入口
├── config.py            # 常量配置
├── physics.py           # Taichi GPU 粒子物理
├── tracking.py          # 摄像头 + MediaPipe
├── motion.py            # 运动特征分析
├── camera_renderer.py   # 水墨滤镜
└── sketch.py            # py5 渲染循环

scripts/
└── render_video.py      # 视频离线渲染

tests/                   # 单元测试
```

## 原理

摄像头 → MediaPipe 手部检测 → 提取速度/曲率/纵深 → Taichi GPU 并行计算 6000 粒子物理 → py5 OpenGL 渲染

## 依赖

py5, opencv-python, mediapipe, numpy, taichi, JDK 17
