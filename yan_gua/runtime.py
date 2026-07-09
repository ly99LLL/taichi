"""运行环境检查，保持入口模块可移植。"""

import os
import re
import shutil
import subprocess


def ensure_java_17() -> None:
    """确认 PATH 或现有 JAVA_HOME 中可用的是 JDK 17+。

    不修改用户环境变量，也不猜测某个发行版的安装目录。
    """
    java_home = os.environ.get("JAVA_HOME")
    java_name = "java.exe" if os.name == "nt" else "java"
    java = os.path.join(java_home, "bin", java_name) if java_home else shutil.which("java")
    if not java or not os.path.isfile(java):
        raise RuntimeError("未找到 Java。请安装 JDK 17，并设置 JAVA_HOME 或将 java 加入 PATH。")

    try:
        result = subprocess.run(
            [java, "-version"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"无法执行 Java: {java}") from exc

    version_output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r'version "(?:1\.)?(\d+)', version_output)
    if not match:
        raise RuntimeError("无法识别 Java 版本，请确认安装的是 JDK 17 或更高版本。")
    if int(match.group(1)) < 17:
        raise RuntimeError(f"当前 Java 主版本为 {match.group(1)}；py5 需要 JDK 17 或更高版本。")
