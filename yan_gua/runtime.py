"""运行环境检查，保持入口模块可移植。"""

import os
import platform
import re
import shutil
import subprocess

TAICHI_ARCH_CHOICES = ("auto", "cpu", "cuda", "metal", "vulkan", "opengl")


def _normalize_machine(machine: str) -> str:
    """把 Python/Java 常见的处理器名称归一为可比较的值。"""
    normalized = machine.strip().lower().replace("-", "_")
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    if normalized in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    return normalized


def _java_machine(version_output: str) -> str | None:
    match = re.search(r"^\s*os\.arch\s*=\s*(\S+)\s*$", version_output, re.MULTILINE)
    return _normalize_machine(match.group(1)) if match else None


def ensure_java_17() -> None:
    """确认 Java 版本可用，并检查它与 Python 的处理器架构是否一致。

    不修改用户环境变量，也不猜测某个发行版的安装目录。
    """
    java_home = os.environ.get("JAVA_HOME")
    java_name = "java.exe" if os.name == "nt" else "java"
    java = os.path.join(java_home, "bin", java_name) if java_home else shutil.which("java")
    if not java or not os.path.isfile(java):
        raise RuntimeError("未找到 Java。请安装 JDK 17，并设置 JAVA_HOME 或将 java 加入 PATH。")

    try:
        result = subprocess.run(
            [java, "-XshowSettings:properties", "-version"],
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

    python_machine = _normalize_machine(platform.machine())
    java_machine = _java_machine(version_output)
    known_machines = {"arm64", "x86_64"}
    if (
        python_machine in known_machines
        and java_machine in known_machines
        and python_machine != java_machine
    ):
        raise RuntimeError(
            "Python 与 Java 架构不一致："
            f"Python={python_machine}，Java={java_machine}。"
            "请安装与 Python 同架构的 JDK；Apple Silicon 应同时使用 arm64 版本。"
        )


def describe_taichi_arch(name: str) -> str:
    """返回面向用户的后端说明，不依赖具体 Apple 芯片代际。"""
    system = platform.system()
    machine = _normalize_machine(platform.machine())
    if name == "auto" and system == "Darwin" and machine == "arm64":
        return "auto（Apple Silicon / arm64：优先 Metal，必要时回退 CPU）"
    if name == "auto":
        return "auto（自动选择可用 GPU，必要时回退 CPU）"
    return name


def resolve_taichi_arch(name: str, taichi_module):
    """把统一的命令行后端名称映射到 Taichi 常量。"""
    if name not in TAICHI_ARCH_CHOICES:
        choices = ", ".join(TAICHI_ARCH_CHOICES)
        raise ValueError(f"不支持的 Taichi 后端：{name}。可用选项：{choices}")

    system = platform.system()
    if name == "metal" and system != "Darwin":
        raise RuntimeError("Metal 后端仅适用于 macOS；当前系统请使用 --arch auto。")
    if name in {"cuda", "opengl"} and system == "Darwin":
        raise RuntimeError(f"macOS 不支持 {name} 后端；请使用 --arch auto 或 --arch metal。")

    attribute = "gpu" if name == "auto" else name
    return getattr(taichi_module, attribute)


def initialize_taichi(name: str, taichi_module, **kwargs) -> None:
    """以一致的提示和错误信息初始化实时与离线 Taichi 后端。"""
    backend = resolve_taichi_arch(name, taichi_module)
    print(f"Taichi 后端：{describe_taichi_arch(name)}")
    try:
        taichi_module.init(arch=backend, **kwargs)
    except Exception as exc:
        hint = "可改用 --arch cpu 验证环境。"
        if platform.system() == "Darwin":
            hint = "请确认 Python 为 arm64 原生版本；也可改用 --arch cpu 验证环境。"
        raise RuntimeError(f"Taichi 后端 {name} 初始化失败。{hint}") from exc
