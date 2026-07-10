"""涡场生命周期测试。"""

import numpy as np

from yan_gua.vortex import VortexController


def hand_state(speed=0.0, observed=True, position=(640.0, 360.0)):
    return {
        "observed": observed,
        "hand_world_pos": np.array(position, dtype=np.float32),
        "hand_velocity": np.array([speed, 0.0], dtype=np.float32),
        "speed": speed,
        "z_velocity": 0.0,
    }


def test_slow_hand_forms_coherent_vortex():
    controller = VortexController()
    fields = None
    for _ in range(60):
        fields = controller.update([hand_state(speed=30.0)], 1 / 60)

    assert fields[0]["active"] is True
    assert fields[0]["phase"] == "holding"
    assert fields[0]["coherence"] > 0.9
    assert fields[0]["scatter"] < 0.1


def test_fast_hand_breaks_vortex_apart():
    controller = VortexController()
    for _ in range(30):
        controller.update([hand_state(speed=20.0)], 1 / 60)
    for _ in range(30):
        fields = controller.update([hand_state(speed=700.0)], 1 / 60)

    assert fields[0]["phase"] == "dispersing"
    assert fields[0]["scatter"] > 0.9
    assert fields[0]["coherence"] < 0.1


def test_missing_hand_leaves_echo_then_returns_to_dust():
    controller = VortexController()
    for _ in range(45):
        controller.update([hand_state()], 1 / 60)

    fields = controller.update([], 1 / 60)
    assert fields[0]["phase"] == "echo"
    strength_after_loss = fields[0]["strength"]

    for _ in range(60):
        fields = controller.update([], 1 / 60)
    assert fields[0]["active"] is True
    assert fields[0]["strength"] < strength_after_loss
    assert fields[0]["release"] > 0.3

    for _ in range(700):
        fields = controller.update([], 1 / 60)
    assert fields[0]["active"] is False
    assert fields[0]["position"] is None


def test_two_hands_have_opposite_spin():
    controller = VortexController()
    fields = controller.update(
        [
            hand_state(position=(420.0, 360.0)),
            hand_state(position=(860.0, 360.0)),
        ],
        1 / 60,
    )
    assert fields[0]["spin"] == -fields[1]["spin"]


def test_vortex_center_follows_slow_hand_without_sluggish_lag():
    controller = VortexController()
    for _ in range(30):
        controller.update([hand_state(position=(400.0, 300.0))], 1 / 60)

    fields = controller.update(
        [hand_state(speed=60.0, position=(500.0, 300.0))],
        1 / 60,
    )

    assert 430.0 < fields[0]["position"][0] < 500.0
    assert fields[0]["coherence"] > 0.9


def test_sudden_stop_keeps_short_splash_impulse():
    controller = VortexController()
    for _ in range(12):
        controller.update([hand_state(speed=260.0)], 1 / 60)

    fields = controller.update([hand_state(speed=0.0)], 1 / 60)
    assert fields[0]["splash"] > 0.4

    for _ in range(40):
        fields = controller.update([hand_state(speed=0.0)], 1 / 60)
    assert fields[0]["splash"] < 0.2
