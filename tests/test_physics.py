"""测试粒子物理模块 — 验证 CloudParticles + Taichi kernel。"""

import numpy as np
import taichi as ti

# Taichi 必须在导入 physics 前初始化
ti.init(arch=ti.cuda, random_seed=42)


def test_cloud_particles_init():
    """CloudParticles 初始化应创建正确数量的粒子。"""
    from yan_gua.physics import CloudParticles

    cp = CloudParticles(count=500, win_w=1280, win_h=720)
    assert cp.c == 500
    assert len(cp.px) == 500
    assert len(cp.py) == 500
    assert len(cp.vx) == 500
    assert len(cp.alpha) == 500
    assert len(cp.ink_level) == 500
    # 所有粒子应在屏幕范围内
    assert np.all(cp.px >= 0) and np.all(cp.px <= 1280)
    assert np.all(cp.py >= 0) and np.all(cp.py <= 720)


def test_cloud_particles_update_no_hands():
    """无手时粒子物理更新不应崩溃。"""
    from yan_gua.physics import CloudParticles

    cp = CloudParticles(count=500, win_w=1280, win_h=720)
    px_before = cp.px.copy()
    py_before = cp.py.copy()

    cp.update(0.016, [])

    # 粒子应有漂移 (位置变化)
    assert not np.allclose(cp.px, px_before)
    assert not np.allclose(cp.py, py_before)


def test_cloud_particles_update_with_hand():
    """有手时粒子应对手势产生响应。"""
    from yan_gua.physics import CloudParticles

    cp = CloudParticles(count=500, win_w=1280, win_h=720)

    # 单手, 居中, 快速移动
    hand_pos = np.array([640.0, 360.0], dtype=np.float32)
    feat = {
        'speed': 200.0,
        'curvature': 0.1,
        'z_velocity': 10.0,
        'hand_velocity': np.array([50.0, 20.0], dtype=np.float32),
        'hand_detected': True,
    }
    hands = [(hand_pos, feat)]

    cp.update(0.016, hands)
    # 不应崩溃, 粒子应有合理的值
    assert not np.any(np.isnan(cp.px))
    assert not np.any(np.isnan(cp.py))
    assert not np.any(np.isnan(cp.vx))


def test_cloud_particles_two_hands():
    """双手叠加不应崩溃。"""
    from yan_gua.physics import CloudParticles

    cp = CloudParticles(count=500, win_w=1280, win_h=720)

    feat = {
        'speed': 100.0,
        'curvature': 0.05,
        'z_velocity': 5.0,
        'hand_velocity': np.array([10.0, 5.0], dtype=np.float32),
        'hand_detected': True,
    }
    hands = [
        (np.array([400.0, 360.0], dtype=np.float32), feat),
        (np.array([880.0, 360.0], dtype=np.float32), feat),
    ]

    cp.update(0.016, hands)
    assert not np.any(np.isnan(cp.px))
    assert not np.any(np.isnan(cp.py))


def test_physics_kernel_warmup():
    """GPU kernel 预热应成功编译。"""
    from yan_gua.physics import _particle_physics_kernel

    n_warm = 100
    _particle_physics_kernel(
        px=np.random.uniform(0, 1280, n_warm).astype(np.float32),
        py=np.random.uniform(0, 720, n_warm).astype(np.float32),
        vx=np.zeros(n_warm, dtype=np.float32),
        vy=np.zeros(n_warm, dtype=np.float32),
        alpha=np.full(n_warm, 5.0, dtype=np.float32),
        radius=np.full(n_warm, 10.0, dtype=np.float32),
        ink_level=np.zeros(n_warm, dtype=np.int32),
        base_alpha=np.full(n_warm, 5.0, dtype=np.float32),
        base_radius=np.full(n_warm, 10.0, dtype=np.float32),
        base_ink=np.zeros(n_warm, dtype=np.int32),
        hx_arr=np.array([640.0, 0.0], dtype=np.float32),
        hy_arr=np.array([360.0, 0.0], dtype=np.float32),
        hvx_arr=np.zeros(2, dtype=np.float32),
        hvy_arr=np.zeros(2, dtype=np.float32),
        hspd_arr=np.array([50.0, 0.0], dtype=np.float32),
        hcurv_arr=np.zeros(2, dtype=np.float32),
        hzvel_arr=np.zeros(2, dtype=np.float32),
        hactive_arr=np.array([1, 0], dtype=np.int32),
        dt=0.016, win_w=1280.0, win_h=720.0,
        infl_r=240.0, max_spd=800.0, curv_ref=400.0, base_damp=0.985,
    )
    # 如果成功返回 (无异常), kernel 已编译
    assert True
