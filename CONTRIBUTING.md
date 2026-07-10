# 参与贡献

感谢你愿意改进演卦。提交改动前，请先搜索现有 Issue；较大的功能或交互变化建议先开
Discussion/Issue 对齐设计方向。

## 本地开发

需要 Python 3.11 或 3.12、JDK 17 或更高版本。实时渲染建议使用 Metal、CUDA 或 Vulkan
后端；CPU 模式和默认测试不需要独立显卡。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
pre-commit install
pytest -m "not cuda"
```

常用质量检查：

```bash
ruff check .
ruff format --check .
pytest -m "not cuda" --cov=yan_gua
```

有 NVIDIA CUDA 环境时，可额外运行：

```bash
set YANGUA_RUN_CUDA_TESTS=1
pytest -m cuda
```

macOS/Linux 请将 `set` 换成 `export`。

## Pull Request 约定

- 每个 PR 聚焦一个问题，并说明动机、实现和验证方式。
- 行为变化必须补测试；视觉变化建议附截图或短视频。
- 不提交大体积原始视频、虚拟环境、构建产物或本机配置。
- 保持 Taichi kernel 与其调用的 `@ti.func` 在同一文件，并遵守
  [AGENTS.md](AGENTS.md) 中记录的 Taichi/py5 约束。

参与本项目即表示同意遵守 [行为准则](CODE_OF_CONDUCT.md)。
