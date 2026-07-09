"""无摄像头合成演示的确定性输入测试。"""

from scripts.render_demo import phase_label, synthetic_hands


def test_synthetic_demo_uses_stable_hand_slots():
    hands = synthetic_hands(2.0)

    assert [hand["id_hint"] for hand in hands] == ["Left", "Right"]
    assert all("palm_center" in hand for hand in hands)


def test_synthetic_demo_releases_both_hands_for_echo():
    assert synthetic_hands(6.4) == []


def test_phase_label_prioritises_visible_lifecycle_state():
    vortices = [
        {"active": True, "phase": "holding"},
        {"active": True, "phase": "dispersing"},
    ]

    assert phase_label(vortices) == "FAST / BREAK"
    assert phase_label([{"active": True, "phase": "echo"}]) == "ECHO / RELEASE"
    assert phase_label([{"active": False, "phase": "dormant"}]) == "DORMANT"
