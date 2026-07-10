"""运行环境检查测试。"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from yan_gua.runtime import (
    TAICHI_ARCH_CHOICES,
    describe_taichi_arch,
    ensure_java_17,
    resolve_taichi_arch,
)


def test_uses_existing_java_home(monkeypatch, tmp_path):
    java = tmp_path / "bin" / ("java.exe" if __import__("os").name == "nt" else "java")
    java.parent.mkdir()
    java.touch()
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    run = Mock(return_value=Mock(stdout="", stderr='openjdk version "17.0.12"'))
    monkeypatch.setattr("yan_gua.runtime.subprocess.run", run)

    ensure_java_17()

    assert Path(run.call_args.args[0][0]) == java


def test_rejects_old_java(monkeypatch, tmp_path):
    java = tmp_path / "bin" / ("java.exe" if __import__("os").name == "nt" else "java")
    java.parent.mkdir()
    java.touch()
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    monkeypatch.setattr(
        "yan_gua.runtime.subprocess.run",
        Mock(return_value=Mock(stdout="", stderr='openjdk version "11.0.2"')),
    )

    with pytest.raises(RuntimeError, match="JDK 17"):
        ensure_java_17()


def test_rejects_java_with_different_processor_architecture(monkeypatch, tmp_path):
    java = tmp_path / "bin" / ("java.exe" if __import__("os").name == "nt" else "java")
    java.parent.mkdir()
    java.touch()
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    monkeypatch.setattr("yan_gua.runtime.platform.machine", Mock(return_value="arm64"))
    monkeypatch.setattr(
        "yan_gua.runtime.subprocess.run",
        Mock(
            return_value=Mock(
                stdout="",
                stderr='openjdk version "21.0.2"\n    os.arch = x86_64',
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Python 与 Java 架构不一致"):
        ensure_java_17()


class FakeTaichi:
    cpu = "cpu-backend"
    cuda = "cuda-backend"
    gpu = "gpu-backend"
    metal = "metal-backend"
    opengl = "opengl-backend"
    vulkan = "vulkan-backend"


def test_all_entrypoints_share_the_same_taichi_backends():
    assert TAICHI_ARCH_CHOICES == ("auto", "cpu", "cuda", "metal", "vulkan", "opengl")
    assert resolve_taichi_arch("auto", FakeTaichi) == "gpu-backend"
    assert resolve_taichi_arch("cpu", FakeTaichi) == "cpu-backend"


def test_apple_silicon_auto_description_is_generation_independent(monkeypatch):
    monkeypatch.setattr("yan_gua.runtime.platform.system", Mock(return_value="Darwin"))
    monkeypatch.setattr("yan_gua.runtime.platform.machine", Mock(return_value="arm64"))

    description = describe_taichi_arch("auto")

    assert "Apple Silicon" in description
    assert "Metal" in description
