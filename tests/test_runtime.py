"""运行环境检查测试。"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from yan_gua.runtime import ensure_java_17


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
