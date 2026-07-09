"""(GPU) 粒子物理引擎 — Taichi CUDA kernel + CloudParticles 管理类.

六大物理力: 中空推力 / 环壁拉力 / 粘性拖拽 / 飞溅 / 漩涡 / 呼吸。
"""

import numpy as np
import taichi as ti

from yan_gua.config import (
    BASE_DAMPING,
    CENTER_GRAVITY,
    CURVATURE_REF,
    HAND_FORCE_MULTIPLIER,
    INFLUENCE_RADIUS,
    INK_COLORS,
    MAX_SPEED,
    NUM_INK_LEVELS,
    PARTICLE_COUNT,
    WARM_LIGHT,
    WINDOW_H,
    WINDOW_W,
)

# ============================================================
# Taichi GPU Kernel — 粒子物理
# ============================================================


@ti.kernel
def _particle_physics_kernel(
    # 粒子状态 (原地修改)
    px: ti.types.ndarray(dtype=ti.f32, ndim=1),
    py: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vx: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ink_level: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # 基准状态 (只读)
    base_alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_ink: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # 手部数据 (最多 2 只手, 未使用的填 0)
    hx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hspd_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hcurv_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hzvel_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hactive_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # 全局参数
    dt: ti.f32,
    win_w: ti.f32,
    win_h: ti.f32,
    infl_r: ti.f32,
    max_spd: ti.f32,
    curv_ref: ti.f32,
    base_damp: ti.f32,
):
    """GPU 并行粒子物理: 环境力 + 双手手势力 + 视觉响应 + 位置积分.

    每个 GPU 线程处理一个粒子, 6000 粒子完全并行。
    """
    for i in range(px.shape[0]):
        # ---- 环境: 有机漂移 ----
        vx[i] += (ti.random(dtype=ti.f32) * 1.2 - 0.6) * dt
        vy[i] += (ti.random(dtype=ti.f32) * 1.2 - 0.6) * dt

        # 弱中心引力
        cx_m = win_w * 0.5
        cy_m = win_h * 0.5
        vx[i] -= (px[i] - cx_m) * CENTER_GRAVITY
        vy[i] -= (py[i] - cy_m) * CENTER_GRAVITY

        influenced = ti.i32(0)

        # ---- 双手手势影响 (叠加) ----
        for h in ti.static(range(2)):
            if hactive_arr[h] == 1:
                _apply_hand_force(
                    i,
                    h,
                    px,
                    py,
                    vx,
                    vy,
                    alpha,
                    radius,
                    ink_level,
                    base_alpha,
                    base_radius,
                    hx_arr,
                    hy_arr,
                    hvx_arr,
                    hvy_arr,
                    hspd_arr,
                    hcurv_arr,
                    hzvel_arr,
                    infl_r,
                    max_spd,
                )
                # 检查该粒子是否在手的影响范围内
                dx = px[i] - hx_arr[h]
                dy = py[i] - hy_arr[h]
                dist = ti.sqrt(dx * dx + dy * dy)
                if dist < infl_r:
                    influenced = 1

        # ---- 不受影响 → 回归基态 ----
        if influenced == 0:
            alpha[i] += (base_alpha[i] - alpha[i]) * 0.03
            radius[i] += (base_radius[i] - radius[i]) * 0.02
            ink_level[i] = base_ink[i]

        # ---- 位置积分 ----
        px[i] += vx[i] * dt * 80.0
        py[i] += vy[i] * dt * 80.0

        # ---- 阻尼 ----
        vx[i] *= base_damp
        vy[i] *= base_damp

        # ---- 屏幕环绕 ----
        if px[i] < -50.0:
            px[i] = win_w + 50.0
        if px[i] > win_w + 50.0:
            px[i] = -50.0
        if py[i] < -50.0:
            py[i] = win_h + 50.0
        if py[i] > win_h + 50.0:
            py[i] = -50.0


@ti.func
def _apply_hand_force(
    i: ti.i32,
    h: ti.i32,
    px: ti.types.ndarray(dtype=ti.f32, ndim=1),
    py: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vx: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ink_level: ti.types.ndarray(dtype=ti.i32, ndim=1),
    base_alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hspd_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hcurv_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hzvel_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    infl_r: ti.f32,
    max_spd: ti.f32,
):
    """对单个粒子施加一只手的所有力: 中空推力/环壁拉力/粘性/飞溅/漩涡/呼吸 + 视觉响应。"""
    hx = hx_arr[h]
    hy = hy_arr[h]
    dx = px[i] - hx
    dy = py[i] - hy
    dist = ti.sqrt(dx * dx + dy * dy)

    if dist < infl_r:
        # ---- 距离衰减 (smoothstep) ----
        t_val = 1.0 - dist / infl_r
        falloff = t_val * t_val * (3.0 - 2.0 * t_val)

        # 径向单位向量
        rnx = dx / (dist + 0.001)
        rny = dy / (dist + 0.001)

        # 运动特征
        nspd = ti.math.clamp(hspd_arr[h] / max_spd, 0.0, 1.0)
        visc = 1.0 - nspd  # 慢 = 粘稠, 快 = 稀薄
        hvx = hvx_arr[h]
        hvy = hvy_arr[h]

        # ---- 六大物理力 ----

        # 1. 中空推力 (0-25%): 粒子向外猛推 → 掌心虚空
        if dist < infl_r * 0.25:
            push = falloff * (8.0 + visc * 6.0) * HAND_FORCE_MULTIPLIER
            vx[i] += rnx * push
            vy[i] += rny * push

        # 2. 环壁拉力 (25-40%): 轻吸维持边界 → 漩涡环
        if dist >= infl_r * 0.25 and dist < infl_r * 0.4:
            attract = falloff * (0.8 + visc * 1.5) * HAND_FORCE_MULTIPLIER
            vx[i] -= rnx * attract
            vy[i] -= rny * attract

        # 3. 粘性拖拽: 手慢 → 粒子跟随手流动
        vx[i] += hvx * falloff * visc * 0.08 * HAND_FORCE_MULTIPLIER
        vy[i] += hvy * falloff * visc * 0.08 * HAND_FORCE_MULTIPLIER

        # 4. 飞溅: 手快 → 粒子向外迸裂
        vx[i] += rnx * falloff * nspd * 2.5 * HAND_FORCE_MULTIPLIER
        vy[i] += rny * falloff * nspd * 2.5 * HAND_FORCE_MULTIPLIER

        # 5. 漩涡: 基线旋转 + 曲率叠加 → 画弧时更猛
        tx = -rny
        ty = rnx
        vortex_f = visc * 3.0 + hcurv_arr[h] * (3.0 + visc * 10.0)
        vx[i] += tx * falloff * vortex_f * HAND_FORCE_MULTIPLIER
        vy[i] += ty * falloff * vortex_f * HAND_FORCE_MULTIPLIER

        # 6. 呼吸: 纵深位移 → 膨胀/收缩
        breath = hzvel_arr[h] * 0.6
        vx[i] += rnx * falloff * breath * HAND_FORCE_MULTIPLIER
        vy[i] += rny * falloff * breath * HAND_FORCE_MULTIPLIER

        # ---- 视觉响应 ----
        activity = ti.math.clamp(
            nspd * 0.4 + hcurv_arr[h] * 2.0 + ti.abs(hzvel_arr[h]) * 0.003,
            0.0,
            1.0,
        )

        # 透明度: 在基准亮度上增亮，避免扰动时反而变暗
        if activity > 0.15:
            target_alpha = base_alpha[i] + 42.0 * activity * falloff
            alpha[i] += (target_alpha - alpha[i]) * 0.2
        else:
            alpha[i] += (base_alpha[i] - alpha[i]) * 0.2

        # 半径: 限制膨胀，避免大圆叠加成混沌色块
        if activity > 0.1:
            target_radius = base_radius[i] + falloff * 6.0 * activity
            radius[i] += (target_radius - radius[i]) * 0.2
        else:
            radius[i] += (base_radius[i] - radius[i]) * 0.2

        # 环带粒子额外增亮 → 漩涡环可见
        in_ring = dist >= infl_r * 0.25 and dist < infl_r * 0.4
        if in_ring and activity > 0.15:
            alpha[i] = ti.min(alpha[i] + 5.0, 72.0)
            radius[i] = ti.min(radius[i] + 1.0, 11.0)


# ============================================================
# 粒子云气 (Taichi GPU 物理 + py5 渲染)
# ============================================================


class CloudParticles:
    """6000 粒子系统。Taichi CUDA kernel 处理物理, py5 OpenGL 渲染。"""

    def __init__(self, count=PARTICLE_COUNT, win_w=WINDOW_W, win_h=WINDOW_H):
        self.c = count
        self.win_w = win_w
        self.win_h = win_h
        rng = np.random.default_rng()

        # 位置 (屏幕空间均匀分布)
        self.px = rng.uniform(0, win_w, count).astype(np.float32)
        self.py = rng.uniform(0, win_h, count).astype(np.float32)

        # 速度
        self.vx = rng.uniform(-0.5, 0.5, count).astype(np.float32)
        self.vy = rng.uniform(-0.5, 0.5, count).astype(np.float32)

        # 视觉属性
        self.alpha = rng.uniform(10, 24, count).astype(np.float32)
        self.radius = rng.uniform(1.8, 5.2, count).astype(np.float32)
        self.ink_level = rng.choice(
            NUM_INK_LEVELS,
            count,
            p=[0.16, 0.14, 0.13, 0.12, 0.11, 0.13, 0.12, 0.09],
        ).astype(np.int32)

        # 基准状态 (不受手影响时的回归目标)
        self.base_alpha = self.alpha.copy()
        self.base_radius = self.radius.copy()
        self.base_ink = self.ink_level.copy()

        # 手部数据打包缓冲 (GPU 传输用, 2 手 x 7 特征)
        self._hx = np.zeros(2, dtype=np.float32)
        self._hy = np.zeros(2, dtype=np.float32)
        self._hvx = np.zeros(2, dtype=np.float32)
        self._hvy = np.zeros(2, dtype=np.float32)
        self._hspd = np.zeros(2, dtype=np.float32)
        self._hcurv = np.zeros(2, dtype=np.float32)
        self._hzvel = np.zeros(2, dtype=np.float32)
        self._hactive = np.zeros(2, dtype=np.int32)

    def update(self, dt, hands):
        """打包手部数据 → GPU kernel → 原地更新粒子状态。

        Args:
            dt: 帧间隔时间 (秒), 自动 clamp 到 0.1s。
            hands: [(pos_2d, features_dict), ...] — 双手独立施加影响力。
        """
        dt = min(dt, 0.1)
        self._pack_hand_data(hands)

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
            self._hspd,
            self._hcurv,
            self._hzvel,
            self._hactive,
            dt,
            float(self.win_w),
            float(self.win_h),
            float(INFLUENCE_RADIUS),
            float(MAX_SPEED),
            float(CURVATURE_REF),
            float(BASE_DAMPING),
        )

    def draw(self, has_hand, hand_x, hand_y):
        """逐粒子渲染: 墨韵软圆 + 多层叠加。

        py5 GPU 渲染, 按 alpha 排序实现深度感 (暗粒子在下, 亮粒子在上)。
        """
        import py5  # 延迟导入: 仅在渲染时需要 py5/JVM

        py5.no_stroke()
        py5.blend_mode(py5.BLEND)

        # 按透明度排序: 远/暗粒子先画, 近/亮粒子后画
        order = np.argsort(self.alpha)

        for j in range(self.c):
            idx = order[j]
            a = self.alpha[idx]
            if a < 1.0:
                continue

            r = self.radius[idx]
            ink = self.ink_level[idx]
            cr, cg, cb = INK_COLORS[min(ink, NUM_INK_LEVELS - 1)]

            # 手部附近混入暖金色
            if has_hand and hand_x is not None:
                cr, cg, cb = self._blend_warm(
                    cr,
                    cg,
                    cb,
                    self.px[idx],
                    self.py[idx],
                    hand_x,
                    hand_y,
                )

            py5.fill(cr, cg, cb, min(a, 255))
            py5.circle(self.px[idx], self.py[idx], r * 2)

    # ---- 内部方法 ----

    def _pack_hand_data(self, hands):
        """将活跃手部数据打包到固定大小数组 (最多 2 手)。"""
        for i in range(2):
            if i < len(hands) and hands[i][0] is not None:
                hp, feat = hands[i]
                self._hx[i] = hp[0]
                self._hy[i] = hp[1]
                self._hvx[i] = feat["hand_velocity"][0]
                self._hvy[i] = feat["hand_velocity"][1]
                self._hspd[i] = feat["speed"]
                self._hcurv[i] = feat["curvature"]
                self._hzvel[i] = feat["z_velocity"]
                self._hactive[i] = 1
            else:
                self._hactive[i] = 0

    @staticmethod
    def _blend_warm(cr, cg, cb, px, py, hand_x, hand_y):
        """手部附近的粒子混入暖金色。"""
        dx = px - hand_x
        dy = py - hand_y
        d = np.sqrt(dx * dx + dy * dy)
        if d < INFLUENCE_RADIUS * 0.7:
            t_val = 1.0 - d / (INFLUENCE_RADIUS * 0.7)
            # 仅轻微注入暖光，保留粒子原本的色相
            cr = int(cr + (WARM_LIGHT[0] - cr) * t_val * 0.25)
            cg = int(cg + (WARM_LIGHT[1] - cg) * t_val * 0.25)
            cb = int(cb + (WARM_LIGHT[2] - cb) * t_val * 0.25)
        return cr, cg, cb
