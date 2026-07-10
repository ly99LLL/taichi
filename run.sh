#!/usr/bin/env bash
# 演卦 · 双生涡场 — macOS / Linux 启动器
#
# 用法:
#   bash run.sh                 # 自动选择 GPU 后端
#   bash run.sh --arch cpu      # 强制 CPU 模式
#   bash run.sh --video 输入.mp4 --record 输出.mp4
#
# 和 Windows 的 run.bat 一样，这个脚本不会覆盖已有的 JAVA_HOME。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 检查并激活虚拟环境 ----
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

# ---- Java 环境 ----
if [ -z "${JAVA_HOME:-}" ]; then
  if command -v java >/dev/null 2>&1; then
    echo "JAVA_HOME 未设置，使用 PATH 中的 java: $(command -v java)"
  else
    echo "错误: 未找到 Java。请安装 JDK 17+ 并设置 JAVA_HOME 或将 java 加入 PATH。"
    exit 1
  fi
else
  echo "JAVA_HOME: ${JAVA_HOME}"
fi

# ---- 启动 ----
echo "=== 演卦 · 双生涡场 ==="
python -m yan_gua "$@"
