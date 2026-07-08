# Contributing

感谢关注这个项目。当前仓库以可运行、可复现和隐私安全为优先级。

## 本地开发

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

## 提交约定

- 保持改动聚焦，一次提交只解决一个明确问题。
- 不提交原始视频、渲染视频、音频、临时帧、环境文件或本地助手记录。
- 新增依赖时同时更新 `pyproject.toml` 和 `requirements.txt`。
- 改动交互、渲染或姿态逻辑后，至少运行一次 preview smoke test。

## 推荐检查

```powershell
python main.py --preview --particles 200 --frames 30 --output preview_smoke.png
python -m pytest
git status --short
git ls-files | rg -i "\.(mp4|mov|avi|mkv|webm|m4v)$"
```

最后一条命令应无输出。
