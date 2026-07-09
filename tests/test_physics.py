"""常驻尘场与 Taichi 涡旋 kernel 测试。"""

import numpy as np
import pytest
import taichi as ti

ti.init(arch=ti.cpu, random_seed=42)

pytestmark = pytest.mark.physics


def vortex(
    slot=0,
    position=(640.0, 360.0),
    coherence=1.0,
    scatter=0.0,
):
    return {
        "slot": slot,
        "position": np.array(position, dtype=np.float32),
        "velocity": np.zeros(2, dtype=np.float32),
        "strength": 1.0,
        "coherence": coherence,
        "scatter": scatter,
        "release": 0.0,
        "maturity": 1.0,
        "aperture": 0.0,
        "spin": 1.0 if slot == 0 else -1.0,
        "active": True,
        "observed": True,
        "phase": "holding",
    }


def test_cloud_particles_init():
    from yan_gua.physics import CloudParticles

    cloud = CloudParticles(count=500, win_w=1280, win_h=720, seed=1)
    assert cloud.c == 500
    assert len(cloud.px) == 500
    assert np.all((cloud.px >= 0) & (cloud.px <= 1280))
    assert np.all((cloud.py >= 0) & (cloud.py <= 720))
    assert np.all(cloud.base_alpha > 0)


def test_cloud_particles_update_no_vortices():
    from yan_gua.physics import CloudParticles

    cloud = CloudParticles(count=500, win_w=1280, win_h=720, seed=2)
    before = np.column_stack((cloud.px.copy(), cloud.py.copy()))
    cloud.update(0.016, [])
    after = np.column_stack((cloud.px, cloud.py))

    assert not np.allclose(after, before)
    assert np.isfinite(after).all()


def test_coherent_vortex_organises_and_brightens_nearby_dust():
    from yan_gua.physics import CloudParticles

    cloud = CloudParticles(count=500, win_w=1280, win_h=720, seed=3)
    # 把一部分尘埃放在目标轨道附近，使测试关注确定性的场响应。
    angles = np.linspace(0, np.pi * 2, 120, endpoint=False)
    cloud.px[:120] = 640 + np.cos(angles) * 92
    cloud.py[:120] = 360 + np.sin(angles) * 92
    alpha_before = float(cloud.alpha[:120].mean())

    for _ in range(20):
        cloud.update(1 / 60, [vortex()])

    assert float(cloud.alpha[:120].mean()) > alpha_before + 5
    assert np.isfinite(cloud.vx).all()
    assert np.isfinite(cloud.vy).all()


def test_two_vortices_pack_into_fixed_identity_slots():
    from yan_gua.physics import CloudParticles

    cloud = CloudParticles(count=200, win_w=1280, win_h=720, seed=4)
    fields = [
        vortex(slot=0, position=(420.0, 360.0)),
        vortex(slot=1, position=(860.0, 360.0)),
    ]
    cloud.update(0.016, fields)

    assert cloud._hactive.tolist() == [1, 1]
    assert cloud._hspin.tolist() == [1.0, -1.0]
    assert np.isfinite(cloud.px).all()


def test_physics_kernel_warmup():
    from yan_gua.physics import _particle_physics_kernel

    count = 100
    _particle_physics_kernel(
        px=np.random.uniform(0, 1280, count).astype(np.float32),
        py=np.random.uniform(0, 720, count).astype(np.float32),
        vx=np.zeros(count, dtype=np.float32),
        vy=np.zeros(count, dtype=np.float32),
        alpha=np.full(count, 5.0, dtype=np.float32),
        radius=np.full(count, 2.0, dtype=np.float32),
        ink_level=np.zeros(count, dtype=np.int32),
        base_alpha=np.full(count, 5.0, dtype=np.float32),
        base_radius=np.full(count, 2.0, dtype=np.float32),
        base_ink=np.zeros(count, dtype=np.int32),
        hx_arr=np.array([640.0, 0.0], dtype=np.float32),
        hy_arr=np.array([360.0, 0.0], dtype=np.float32),
        hvx_arr=np.zeros(2, dtype=np.float32),
        hvy_arr=np.zeros(2, dtype=np.float32),
        hstrength_arr=np.array([1.0, 0.0], dtype=np.float32),
        hcoherence_arr=np.array([1.0, 0.0], dtype=np.float32),
        hscatter_arr=np.zeros(2, dtype=np.float32),
        hrelease_arr=np.zeros(2, dtype=np.float32),
        hmaturity_arr=np.array([1.0, 0.0], dtype=np.float32),
        haperture_arr=np.zeros(2, dtype=np.float32),
        hspin_arr=np.array([1.0, -1.0], dtype=np.float32),
        hactive_arr=np.array([1, 0], dtype=np.int32),
        dt=0.016,
        win_w=1280.0,
        win_h=720.0,
        infl_r=320.0,
        orbit_r=92.0,
        base_damp=0.973,
    )
    assert True
