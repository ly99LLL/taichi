"""双生涡场粒子引擎 — Taichi 并行物理 + py5 渲染。

慢手把背景尘埃组织成稳定轨道；快手降低相干性并产生径向剪切。检测丢失
时涡场继续以余涡形态衰减，因此粒子的去向始终连续。
"""

import numpy as np
import taichi as ti

from yan_gua.config import (
    AMBIENT_DRIFT,
    BASE_DAMPING,
    COHERENT_COLOR,
    ECHO_COLOR,
    INK_COLORS,
    NUM_INK_LEVELS,
    PARTICLE_ALPHA_MAX,
    PARTICLE_ALPHA_MIN,
    PARTICLE_COUNT,
    PARTICLE_SIZE_MAX,
    PARTICLE_SIZE_MIN,
    SCATTER_COLOR,
    VORTEX_INFLUENCE_RADIUS,
    VORTEX_ORBIT_RADIUS,
    VORTEX_ORBIT_SPEED,
    VORTEX_PAIR_DISTANCE,
    WINDOW_H,
    WINDOW_W,
)


@ti.kernel
def _particle_physics_kernel(
    px: ti.types.ndarray(dtype=ti.f32, ndim=1),
    py: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vx: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ink_level: ti.types.ndarray(dtype=ti.i32, ndim=1),
    base_alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_ink: ti.types.ndarray(dtype=ti.i32, ndim=1),
    hx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hstrength_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hcoherence_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hscatter_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hrelease_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hmaturity_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    haperture_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hspin_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hactive_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    dt: ti.f32,
    win_w: ti.f32,
    win_h: ti.f32,
    infl_r: ti.f32,
    orbit_r: ti.f32,
    base_damp: ti.f32,
):
    """推进背景尘场、两个涡旋和双手之间的流桥。"""
    for i in range(px.shape[0]):
        # 背景不是静止的库存，而是一层几乎不可见的缓慢尘流。
        vx[i] += (ti.random(dtype=ti.f32) * 2.0 - 1.0) * AMBIENT_DRIFT * dt
        vy[i] += (ti.random(dtype=ti.f32) * 2.0 - 1.0) * AMBIENT_DRIFT * dt

        visual_signal = 0.0
        visual_coherence = 0.0
        visual_scatter = 0.0
        visual_release = 0.0

        for h in ti.static(range(2)):
            if hactive_arr[h] == 1:
                dx = px[i] - hx_arr[h]
                dy = py[i] - hy_arr[h]
                dist = ti.sqrt(dx * dx + dy * dy) + 0.001
                reach = infl_r * (1.0 + hrelease_arr[h] * 0.35)

                if dist < reach:
                    normal_x = dx / dist
                    normal_y = dy / dist
                    tangent_x = -normal_y * hspin_arr[h]
                    tangent_y = normal_x * hspin_arr[h]
                    unit_distance = 1.0 - dist / reach
                    falloff = unit_distance * unit_distance * (3.0 - 2.0 * unit_distance)

                    ring_radius = orbit_r * (1.0 + hrelease_arr[h] * 0.9 + haperture_arr[h])
                    ring_width = ring_radius * (0.32 + hscatter_arr[h] * 0.55)
                    ring_band = ti.exp(-ti.abs(dist - ring_radius) / (ring_width + 0.001))

                    maturity_gain = 0.28 + hmaturity_arr[h] * 0.72
                    coherent = hstrength_arr[h] * hcoherence_arr[h] * maturity_gain
                    scattered = hstrength_arr[h] * hscatter_arr[h]

                    # 相干涡旋：将速度连续牵引到切向轨道，而不是直接推开粒子。
                    orbit_speed = (
                        VORTEX_ORBIT_SPEED
                        * (0.58 + ring_band * 0.42)
                        * (1.0 + haperture_arr[h] * 0.18)
                    )
                    target_vx = tangent_x * orbit_speed + hvx_arr[h] * 0.08
                    target_vy = tangent_y * orbit_speed + hvy_arr[h] * 0.08
                    steering = coherent * falloff * (1.8 + ring_band * 2.4)
                    vx[i] += (target_vx - vx[i]) * steering * dt
                    vy[i] += (target_vy - vy[i]) * steering * dt

                    # 空心轨道：环外向内收，环内向外托住，掌心保持安静。
                    ring_error = ti.math.clamp(
                        (dist - ring_radius) * 3.2,
                        -230.0,
                        330.0,
                    )
                    radial_gain = hstrength_arr[h] * (0.22 + hcoherence_arr[h] * 0.78)
                    vx[i] -= normal_x * ring_error * radial_gain * falloff * dt
                    vy[i] -= normal_y * ring_error * radial_gain * falloff * dt

                    # 快手不制造新涡旋，而是把现有轨道剪碎、吹散。
                    burst = scattered * falloff * (0.35 + ring_band * 0.65)
                    turbulence_x = ti.random(dtype=ti.f32) * 2.0 - 1.0
                    turbulence_y = ti.random(dtype=ti.f32) * 2.0 - 1.0
                    vx[i] += (normal_x * 520.0 * burst + turbulence_x * 145.0 * burst) * dt
                    vy[i] += (normal_y * 520.0 * burst + turbulence_y * 145.0 * burst) * dt

                    # 余涡解束时轨道扩大并轻轻向外释放。
                    release_force = hstrength_arr[h] * hrelease_arr[h] * falloff * 95.0
                    vx[i] += normal_x * release_force * dt
                    vy[i] += normal_y * release_force * dt

                    signal = (
                        hstrength_arr[h]
                        * falloff
                        * (
                            hcoherence_arr[h] * ring_band
                            + hscatter_arr[h] * 0.42
                            + (1.0 - hrelease_arr[h]) * 0.08
                        )
                    )
                    if signal > visual_signal:
                        visual_signal = signal
                        visual_coherence = hcoherence_arr[h]
                        visual_scatter = hscatter_arr[h]
                        visual_release = hrelease_arr[h]

        # 两只相干涡旋靠近时，粒子沿中轴形成一条轻微的 ∞ 形流桥。
        if hactive_arr[0] == 1 and hactive_arr[1] == 1:
            pair_dx = hx_arr[1] - hx_arr[0]
            pair_dy = hy_arr[1] - hy_arr[0]
            pair_dist = ti.sqrt(pair_dx * pair_dx + pair_dy * pair_dy) + 0.001
            if pair_dist > orbit_r * 1.5 and pair_dist < VORTEX_PAIR_DISTANCE:
                axis_x = pair_dx / pair_dist
                axis_y = pair_dy / pair_dist
                local_x = px[i] - hx_arr[0]
                local_y = py[i] - hy_arr[0]
                projection = ti.math.clamp(
                    local_x * axis_x + local_y * axis_y,
                    0.0,
                    pair_dist,
                )
                nearest_x = hx_arr[0] + axis_x * projection
                nearest_y = hy_arr[0] + axis_y * projection
                bridge_dx = px[i] - nearest_x
                bridge_dy = py[i] - nearest_y
                bridge_dist = ti.sqrt(bridge_dx * bridge_dx + bridge_dy * bridge_dy) + 0.001
                bridge_width = orbit_r * 0.72

                if bridge_dist < bridge_width:
                    center_weight = ti.sin(3.14159265 * projection / pair_dist)
                    bridge_weight = (
                        ti.min(hstrength_arr[0], hstrength_arr[1])
                        * ti.min(hcoherence_arr[0], hcoherence_arr[1])
                        * center_weight
                        * (1.0 - bridge_dist / bridge_width)
                    )
                    bridge_nx = bridge_dx / bridge_dist
                    bridge_ny = bridge_dy / bridge_dist
                    vx[i] -= bridge_nx * bridge_dist * 5.5 * bridge_weight * dt
                    vy[i] -= bridge_ny * bridge_dist * 5.5 * bridge_weight * dt
                    vx[i] += -axis_y * hspin_arr[0] * 75.0 * bridge_weight * dt
                    vy[i] += axis_x * hspin_arr[0] * 75.0 * bridge_weight * dt
                    if bridge_weight * 0.7 > visual_signal:
                        visual_signal = bridge_weight * 0.7
                        visual_coherence = 1.0
                        visual_scatter = 0.0
                        visual_release = 0.0

        # 可见度来自“被组织起来”，不是粒子的出生。
        if visual_signal > 0.001:
            target_alpha = base_alpha[i] + visual_signal * (30.0 + visual_coherence * 54.0)
            target_radius = base_radius[i] + visual_signal * (0.7 + visual_coherence * 2.7)
            alpha[i] += (target_alpha - alpha[i]) * ti.min(dt * 8.0, 1.0)
            radius[i] += (target_radius - radius[i]) * ti.min(dt * 7.0, 1.0)

            if visual_release > 0.35:
                ink_level[i] = 0 + i % 3
            elif visual_scatter > visual_coherence:
                ink_level[i] = 3 + i % 2
            elif visual_coherence > 0.55:
                ink_level[i] = 5 + i % 2
        else:
            alpha[i] += (base_alpha[i] - alpha[i]) * ti.min(dt * 2.4, 1.0)
            radius[i] += (base_radius[i] - radius[i]) * ti.min(dt * 2.1, 1.0)
            ink_level[i] = base_ink[i]

        speed_sq = vx[i] * vx[i] + vy[i] * vy[i]
        if speed_sq > 700.0 * 700.0:
            speed_scale = 700.0 / ti.sqrt(speed_sq)
            vx[i] *= speed_scale
            vy[i] *= speed_scale

        px[i] += vx[i] * dt
        py[i] += vy[i] * dt
        damping = ti.pow(base_damp, dt * 60.0)
        vx[i] *= damping
        vy[i] *= damping

        margin = 42.0
        if px[i] < -margin:
            px[i] = win_w + margin
        if px[i] > win_w + margin:
            px[i] = -margin
        if py[i] < -margin:
            py[i] = win_h + margin
        if py[i] > win_h + margin:
            py[i] = -margin


class CloudParticles:
    """常驻尘场；涡旋只组织粒子，不负责生成粒子。"""

    def __init__(
        self,
        count: int = PARTICLE_COUNT,
        win_w: int = WINDOW_W,
        win_h: int = WINDOW_H,
        seed: int | None = None,
    ):
        self.c = count
        self.win_w = win_w
        self.win_h = win_h
        rng = np.random.default_rng(seed)

        self.px = rng.uniform(0, win_w, count).astype(np.float32)
        self.py = rng.uniform(0, win_h, count).astype(np.float32)
        self.vx = rng.uniform(-12.0, 12.0, count).astype(np.float32)
        self.vy = rng.uniform(-12.0, 12.0, count).astype(np.float32)
        self.alpha = rng.uniform(
            PARTICLE_ALPHA_MIN,
            PARTICLE_ALPHA_MAX,
            count,
        ).astype(np.float32)
        self.radius = rng.uniform(
            PARTICLE_SIZE_MIN,
            PARTICLE_SIZE_MAX,
            count,
        ).astype(np.float32)
        self.ink_level = rng.choice(
            NUM_INK_LEVELS,
            count,
            p=[0.20, 0.18, 0.17, 0.12, 0.08, 0.07, 0.10, 0.08],
        ).astype(np.int32)

        self.base_alpha = self.alpha.copy()
        self.base_radius = self.radius.copy()
        self.base_ink = self.ink_level.copy()

        self._hx = np.zeros(2, dtype=np.float32)
        self._hy = np.zeros(2, dtype=np.float32)
        self._hvx = np.zeros(2, dtype=np.float32)
        self._hvy = np.zeros(2, dtype=np.float32)
        self._hstrength = np.zeros(2, dtype=np.float32)
        self._hcoherence = np.zeros(2, dtype=np.float32)
        self._hscatter = np.zeros(2, dtype=np.float32)
        self._hrelease = np.ones(2, dtype=np.float32)
        self._hmaturity = np.zeros(2, dtype=np.float32)
        self._haperture = np.zeros(2, dtype=np.float32)
        self._hspin = np.array([1.0, -1.0], dtype=np.float32)
        self._hactive = np.zeros(2, dtype=np.int32)

    def update(self, dt: float, vortices: list[dict]) -> None:
        dt = min(max(float(dt), 0.0), 0.1)
        self._pack_vortices(vortices)
        _particle_physics_kernel(
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.alpha,
            self.radius,
            self.ink_level,
            self.base_alpha,
            self.base_radius,
            self.base_ink,
            self._hx,
            self._hy,
            self._hvx,
            self._hvy,
            self._hstrength,
            self._hcoherence,
            self._hscatter,
            self._hrelease,
            self._hmaturity,
            self._haperture,
            self._hspin,
            self._hactive,
            dt,
            float(self.win_w),
            float(self.win_h),
            float(VORTEX_INFLUENCE_RADIUS),
            float(VORTEX_ORBIT_RADIUS),
            float(BASE_DAMPING),
        )

    def draw(self, vortices: list[dict]) -> None:
        """按亮度排序渲染；高能粒子有一层克制的光晕。"""
        import py5

        py5.no_stroke()
        py5.blend_mode(py5.BLEND)
        order = np.argsort(self.alpha)

        for idx in order:
            particle_alpha = float(self.alpha[idx])
            if particle_alpha < 1.0:
                continue

            ink = min(int(self.ink_level[idx]), NUM_INK_LEVELS - 1)
            color = INK_COLORS[ink]
            color = self.tint_color(
                color,
                float(self.px[idx]),
                float(self.py[idx]),
                vortices,
            )
            particle_radius = float(self.radius[idx])

            if particle_alpha > 42.0:
                py5.fill(*color, min(particle_alpha * 0.12, 16.0))
                py5.circle(
                    self.px[idx],
                    self.py[idx],
                    particle_radius * 3.4,
                )

            py5.fill(*color, min(particle_alpha, 150.0))
            py5.circle(
                self.px[idx],
                self.py[idx],
                particle_radius * 2.0,
            )

    def _pack_vortices(self, vortices: list[dict]) -> None:
        self._hactive.fill(0)
        for field in vortices:
            slot = int(field.get("slot", 0))
            if slot < 0 or slot >= 2 or not field.get("active"):
                continue
            position = field.get("position")
            if position is None:
                continue

            velocity = field.get("velocity", (0.0, 0.0))
            self._hx[slot] = position[0]
            self._hy[slot] = position[1]
            self._hvx[slot] = velocity[0]
            self._hvy[slot] = velocity[1]
            self._hstrength[slot] = field.get("strength", 0.0)
            self._hcoherence[slot] = field.get("coherence", 0.0)
            self._hscatter[slot] = field.get("scatter", 0.0)
            self._hrelease[slot] = field.get("release", 0.0)
            self._hmaturity[slot] = field.get("maturity", 0.0)
            self._haperture[slot] = field.get("aperture", 0.0)
            self._hspin[slot] = field.get("spin", 1.0 if slot == 0 else -1.0)
            self._hactive[slot] = 1

    @staticmethod
    def tint_color(
        color: tuple[int, int, int],
        px: float,
        py: float,
        vortices: list[dict],
    ) -> tuple[int, int, int]:
        """按最近涡场的阶段做轻量色温偏移，保留粒子自身色阶。"""
        best_weight = 0.0
        target = color
        for field in vortices:
            if not field.get("active") or field.get("position") is None:
                continue
            position = field["position"]
            distance = float(np.hypot(px - position[0], py - position[1]))
            reach = VORTEX_INFLUENCE_RADIUS * (1.0 + field.get("release", 0.0) * 0.35)
            if distance >= reach:
                continue
            weight = (1.0 - distance / reach) * field.get("strength", 0.0)
            if weight <= best_weight:
                continue
            best_weight = weight
            if field.get("release", 0.0) > 0.3:
                target = ECHO_COLOR
            elif field.get("scatter", 0.0) > field.get("coherence", 0.0):
                target = SCATTER_COLOR
            else:
                target = COHERENT_COLOR

        blend = min(best_weight * 0.28, 0.28)
        return tuple(
            int(channel + (target_channel - channel) * blend)
            for channel, target_channel in zip(color, target, strict=True)
        )
