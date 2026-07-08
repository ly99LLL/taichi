from pathlib import Path

from bagua.config import Config
from bagua.pose_matcher import PoseMatcher


def test_config_defaults_are_usable() -> None:
    config = Config()

    assert config.width > 0
    assert config.height > 0
    assert config.dark_particle_count > 0


def test_reference_templates_load() -> None:
    root = Path(__file__).resolve().parents[1]
    matcher = PoseMatcher(root / "reference_poses")

    assert len(matcher.known_moves) >= 24
    assert "第十五式 转身左蹬脚" in matcher.known_moves
