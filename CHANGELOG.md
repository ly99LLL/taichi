# 更新日志

本项目遵循[语义化版本](https://semver.org/lang/zh-CN/)，从 v3.0.1 起在此记录重要变化。

## [未发布]

### 新增

- Python 包元数据、开发依赖和统一工具配置。
- Python 3.11/3.12 CPU 测试、Ruff 与覆盖率 GitHub Actions。
- 可选 CUDA 集成测试、pre-commit 和 Dependabot。
- 许可证、贡献指南、安全政策、行为准则和 Issue/PR 模板。
- 视频输入与主窗口录制能力。

### 修复

- 使用真实 MediaPipe Z 坐标计算纵深速度，不再误用屏幕 Y 坐标。
- 启动时尊重现有 `JAVA_HOME`，移除作者电脑上的 JDK 绝对路径。

## [3.0.0] - 2026-07-07

- 将单文件原型重构为 `yan_gua` 包。
- 引入 Taichi GPU 粒子物理、MediaPipe Hands/Pose 双轨检测和 py5 渲染。

[未发布]: https://github.com/ly99LLL/taichi/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/ly99LLL/taichi/releases/tag/v3.0.0
