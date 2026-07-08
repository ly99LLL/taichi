from __future__ import annotations

import colorsys
import math

import numpy as np

from .audio import AudioEngine
from .config import Config
from .input_pose import FieldState


class ParticleSystem:
    def __init__(self, config: Config, seed: int = 7) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.bloom_cooldown = 0.0
        self.gold_accumulator = 0.0
        self.trail_cursor = 0
        self.gold_cursor = 0
        self.bloom_cursor = 0

        n = config.dark_particle_count
        self.pos = np.empty((n, 3), dtype=np.float32)
        self.pos[:, 0] = self.rng.uniform(-config.world_x, config.world_x, n)
        self.pos[:, 1] = self.rng.uniform(-config.world_y, config.world_y, n)
        self.pos[:, 2] = self.rng.uniform(config.world_z_far, config.world_z_near, n)
        self.vel = self.rng.normal(0.0, 0.035, (n, 3)).astype(np.float32)
        self.mass = self.rng.uniform(0.65, 1.75, n).astype(np.float32)
        self.seed = self.rng.uniform(0.0, 1000.0, n).astype(np.float32)
        self.brightness = self.rng.uniform(0.0, 0.08, n).astype(np.float32)
        self.base_size = self.rng.uniform(2.0, 5.8, n).astype(np.float32)
        self.base_color = self._make_base_colors(n)

        self.trail_pos = np.zeros((config.max_trail_particles, 3), dtype=np.float32)
        self.trail_vel = np.zeros_like(self.trail_pos)
        self.trail_color = np.zeros_like(self.trail_pos)
        self.trail_size = np.zeros(config.max_trail_particles, dtype=np.float32)
        self.trail_life = np.zeros(config.max_trail_particles, dtype=np.float32)
        self.trail_max_life = np.ones(config.max_trail_particles, dtype=np.float32)

        self.gold_pos = np.zeros((config.max_golden_particles, 3), dtype=np.float32)
        self.gold_vel = np.zeros_like(self.gold_pos)
        self.gold_size = np.zeros(config.max_golden_particles, dtype=np.float32)
        self.gold_life = np.zeros(config.max_golden_particles, dtype=np.float32)
        self.gold_max_life = np.ones(config.max_golden_particles, dtype=np.float32)

        self.bloom_pos = np.zeros((config.max_bloom_particles, 3), dtype=np.float32)
        self.bloom_vel = np.zeros_like(self.bloom_pos)
        self.bloom_size = np.zeros(config.max_bloom_particles, dtype=np.float32)
        self.bloom_life = np.zeros(config.max_bloom_particles, dtype=np.float32)
        self.bloom_max_life = np.ones(config.max_bloom_particles, dtype=np.float32)

        # FX particles: special move effects (kick bursts, etc.)
        self.fx_max = 500
        self.fx_pos = np.zeros((self.fx_max, 3), dtype=np.float32)
        self.fx_vel = np.zeros_like(self.fx_pos)
        self.fx_color = np.zeros_like(self.fx_pos)
        self.fx_size = np.zeros(self.fx_max, dtype=np.float32)
        self.fx_life = np.zeros(self.fx_max, dtype=np.float32)
        self.fx_max_life = np.ones(self.fx_max, dtype=np.float32)
        self.fx_cursor = 0
        self.fx_cooldown = 0.0

        # Shockwave rings: screen-space expanding circles for dramatic moments
        self.shockwaves: list[dict] = []

        # Jet reversal state for move explosions (delayed outward blast)
        self._jet_pending: bool = False
        self._jet_indices: np.ndarray = np.array([], dtype=np.int64)
        self._jet_direction: np.ndarray = np.zeros(3, dtype=np.float32)
        self._jet_timer: float = 0.0
        self._jet_force: float = 0.0

        # Performance auto-degradation
        self.perf_mode: int = 0  # 0=full, 1=reduced, 2=minimal
        self._perf_fps: float = 60.0
        self._perf_frame_count: int = 0

        self.last_right_trail = None
        self.last_left_trail = None

    @property
    def total_count(self) -> int:
        return (
            len(self.pos)
            + int(np.count_nonzero(self.trail_life > 0.0))
            + int(np.count_nonzero(self.gold_life > 0.0))
            + int(np.count_nonzero(self.bloom_life > 0.0))
            + int(np.count_nonzero(self.fx_life > 0.0))
        )

    def update(self, dt: float, field: FieldState, audio: AudioEngine) -> bool:
        dt = min(max(dt, 1.0 / 240.0), 1.0 / 20.0)
        self.time += dt
        self.bloom_cooldown = max(0.0, self.bloom_cooldown - dt)

        # ── Performance auto-degradation ──────────────────────────
        self._perf_frame_count += 1
        if self._perf_frame_count % 30 == 0:
            instant_fps = 1.0 / max(dt, 1e-5)
            a = self.config.perf_fps_ema_alpha
            self._perf_fps = self._perf_fps * (1.0 - a) + instant_fps * a
            if self._perf_fps < self.config.perf_fps_critical:
                self.perf_mode = 2
            elif self._perf_fps < self.config.perf_fps_warn:
                self.perf_mode = 1
            else:
                self.perf_mode = 0

        # Drift force: skip in degraded modes
        if self.perf_mode < 1:
            force = self._drift_force()
        else:
            force = np.zeros_like(self.pos)

        min_dist = np.full(len(self.pos), 999.0, dtype=np.float32)
        if field.right_active:
            min_dist = np.minimum(
                min_dist,
                self._apply_hand(force, field.right_pos, field.right_vel, field, 1.0),
            )
        if field.left_active:
            min_dist = np.minimum(
                min_dist,
                self._apply_hand(
                    force,
                    field.left_pos,
                    field.left_vel,
                    field,
                    self.config.secondary_hand_strength,
                ),
            )

        force -= self.vel * self.config.drag_coefficient
        self._apply_bounds(force)
        self.vel += (force / self.mass[:, None]) * dt
        speed = np.linalg.norm(self.vel, axis=1)
        too_fast = speed > 10.5
        if np.any(too_fast):
            self.vel[too_fast] *= (10.5 / speed[too_fast])[:, None]
        self.pos += self.vel * dt

        activation_mask = (min_dist < 0.55) & (field.ema_speed < self.config.attract_threshold)
        self.brightness[activation_mask] = np.minimum(
            1.0,
            self.brightness[activation_mask] + self.config.activation_rate * dt,
        )
        self.brightness[~activation_mask] = np.maximum(
            0.0,
            self.brightness[~activation_mask] - self.config.deactivation_rate * dt,
        )

        self._spawn_trails(field, audio)
        self._update_trails(dt)
        self._spawn_gold(field)
        self._update_gold(dt)
        self._update_bloom_particles(dt)
        self._update_fx_particles(dt)
        self.fx_cooldown = max(0.0, self.fx_cooldown - dt)

        # ── Jet reversal: delayed outward blast from move explosion ──
        if self._jet_pending:
            self._jet_timer -= dt
            if self._jet_timer <= 0.0 and len(self._jet_indices) > 0:
                jet_count = len(self._jet_indices)
                spread = self.rng.normal(0.0, 0.35, (jet_count, 3)).astype(np.float32)
                dirs = self._jet_direction[None, :] + spread
                dirs /= np.linalg.norm(dirs, axis=1)[:, None] + 1e-5
                speeds = self.rng.uniform(
                    self._jet_force * 0.6, self._jet_force * 1.4, jet_count,
                ).astype(np.float32)
                self.vel[self._jet_indices] = dirs * speeds[:, None]
                self.brightness[self._jet_indices] = 1.0
                self._jet_pending = False

        did_bloom = False
        if field.bloom_ready and self.bloom_cooldown <= 0.0:
            self.trigger_bloom(field, audio)
            did_bloom = True
        return did_bloom

    def trigger_bloom(self, field: FieldState, audio: AudioEngine | None = None) -> None:
        center = field.right_pos
        delta = self.pos - center[None, :]
        dist = np.linalg.norm(delta, axis=1) + 1e-5
        mask = dist < 2.35
        if np.any(mask):
            direction = delta[mask] / dist[mask, None]
            swirl = self.rng.normal(0.0, 0.70, (np.count_nonzero(mask), 3)).astype(np.float32)
            bias = np.array([0.0, 1.35, 0.78], dtype=np.float32)
            blast = (
                direction
                * self.rng.uniform(
                    self.config.bloom_explosion_force * 0.55,
                    self.config.bloom_explosion_force * 1.08,
                    (np.count_nonzero(mask), 1),
                ).astype(np.float32)
                + swirl
                + bias
            )
            falloff = np.clip(1.0 - dist[mask] / 2.35, 0.0, 1.0)
            self.vel[mask] += blast * (0.72 + falloff[:, None] * 0.75)
            self.brightness[mask] = 1.0

        self._add_bloom_particles(center)
        self.bloom_cooldown = self.config.bloom_cooldown_s
        if audio is not None:
            audio.bloom(float(center[1]))

    # ── Move explosion: reuse existing dark particles ──────────────

    def trigger_move_explosion(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        radius: float = 2.5,
        max_particles: int = 400,
        splatter_fraction: float = 0.6,
        force: float = 16.0,
        converge_delay: float = 0.12,
    ) -> None:
        """Repurpose existing dark particles near *origin* for a move explosion.

        Instead of spawning new particles, this blasts the existing ink
        cloud outward — 万法归宗, 气场爆发.
        """
        # Select nearby dark particles
        delta = self.pos - origin[None, :]
        dist = np.linalg.norm(delta, axis=1)
        nearby = np.where(dist < radius)[0]
        if len(nearby) == 0:
            return

        if len(nearby) > max_particles:
            nearby = self.rng.choice(nearby, max_particles, replace=False)

        base_dir = direction / (np.linalg.norm(direction) + 1e-5)

        # Split: splatter (immediate outward) vs jet (converge then blast)
        split = int(len(nearby) * splatter_fraction)
        splatter_idx = nearby[:split]
        jet_idx = nearby[split:]

        if len(splatter_idx) > 0:
            spread = self.rng.normal(0.0, 0.45, (len(splatter_idx), 3)).astype(np.float32)
            dirs = base_dir[None, :] + spread
            dirs /= np.linalg.norm(dirs, axis=1)[:, None] + 1e-5
            speeds = self.rng.uniform(
                force * 0.5, force * 1.2, len(splatter_idx),
            ).astype(np.float32)
            self.vel[splatter_idx] = dirs * speeds[:, None]
            self.brightness[splatter_idx] = 1.0

        if len(jet_idx) > 0:
            # Converge inward rapidly — will reverse after converge_delay
            to_origin = origin[None, :] - self.pos[jet_idx]
            to_origin /= np.linalg.norm(to_origin, axis=1)[:, None] + 1e-5
            converge_speed = self.rng.uniform(
                force * 1.1, force * 2.0, len(jet_idx),
            ).astype(np.float32)
            self.vel[jet_idx] = to_origin * converge_speed[:, None]
            self.brightness[jet_idx] = 1.0
            self._jet_pending = True
            self._jet_indices = jet_idx
            self._jet_direction = base_dir
            self._jet_timer = converge_delay
            self._jet_force = force

    # ── Draw data accessors ────────────────────────────────────────

    def dark_draw_data(self, ga: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        breath = 0.0
        if ga > 0.75:
            phase = (math.sin(self.time * math.tau / self.config.breath_interval - math.pi / 2.0) + 1.0) * 0.5
            breath = (phase**4) * self.config.breath_brightness_delta
        bright = np.clip(self.brightness + breath, 0.0, 1.0)
        ink = np.array([22.0, 24.0, 27.0], dtype=np.float32)
        warm_ink = np.array([64.0, 45.0, 29.0], dtype=np.float32)
        target = ink * (1.0 - min(ga, 0.65) * 0.25) + warm_ink * min(ga, 0.65) * 0.25
        colors = self.base_color * (1.0 - bright[:, None]) + target[None, :] * bright[:, None]
        alpha = 28.0 + bright * 155.0 + ga * 12.0
        size = self.base_size * (0.68 + bright * 1.22)
        return self.pos, colors, size, alpha

    def trail_draw_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = self.trail_life > 0.0
        if not np.any(mask):
            return self._empty()
        ratio = np.clip(self.trail_life[mask] / self.trail_max_life[mask], 0.0, 1.0)
        return self.trail_pos[mask], self.trail_color[mask], self.trail_size[mask] * (0.35 + ratio * 0.82), 108.0 * ratio

    def gold_draw_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = self.gold_life > 0.0
        if not np.any(mask):
            return self._empty()
        ratio = np.clip(self.gold_life[mask] / self.gold_max_life[mask], 0.0, 1.0)
        colors = np.zeros((np.count_nonzero(mask), 3), dtype=np.float32)
        colors[:, 0] = 152.0
        colors[:, 1] = 112.0 + 18.0 * ratio
        colors[:, 2] = 48.0
        return self.gold_pos[mask], colors, self.gold_size[mask], 130.0 * np.minimum(1.0, ratio * 2.0)

    def bloom_draw_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = self.bloom_life > 0.0
        if not np.any(mask):
            return self._empty()
        ratio = np.clip(self.bloom_life[mask] / self.bloom_max_life[mask], 0.0, 1.0)
        colors = np.zeros((np.count_nonzero(mask), 3), dtype=np.float32)
        colors[:, 0] = 35.0
        colors[:, 1] = 32.0
        colors[:, 2] = 28.0
        return self.bloom_pos[mask], colors, self.bloom_size[mask] * (0.45 + ratio), 118.0 * ratio

    def _make_base_colors(self, n: int) -> np.ndarray:
        colors = np.zeros((n, 3), dtype=np.float32)
        warm = self.rng.random(n) < 0.18
        hues = np.where(warm, self.rng.uniform(25.0, 42.0, n), self.rng.uniform(210.0, 245.0, n))
        sats = self.rng.uniform(0.04, 0.16, n)
        vals = self.rng.uniform(0.20, 0.38, n)
        for i in range(n):
            r, g, b = colorsys.hls_to_rgb(hues[i] / 360.0, vals[i], sats[i])
            colors[i] = (r * 255.0, g * 255.0, b * 255.0)
        return colors

    def _drift_force(self) -> np.ndarray:
        p = self.pos
        t = self.time
        s = self.seed
        amp = self.config.drift_amplitude
        force = np.empty_like(p)
        force[:, 0] = np.sin(p[:, 1] * 1.7 + p[:, 2] * 0.45 + s + t * 0.31)
        force[:, 1] = np.sin(p[:, 2] * 1.1 + p[:, 0] * 0.55 + s * 1.3 + t * 0.23)
        force[:, 2] = np.cos(p[:, 0] * 1.2 + p[:, 1] * 0.65 + s * 0.7 + t * 0.19)
        return force * amp

    def _apply_hand(
        self,
        force: np.ndarray,
        hand_pos: np.ndarray,
        hand_vel: np.ndarray,
        field: FieldState,
        strength_scale: float,
    ) -> np.ndarray:
        delta = hand_pos[None, :] - self.pos
        dist = np.linalg.norm(delta, axis=1) + 1e-5
        mask = dist < field.radius
        if not np.any(mask):
            return dist

        direction = delta[mask] / dist[mask, None]
        falloff = np.clip(1.0 - dist[mask] / field.radius, 0.0, 1.0)
        falloff = falloff * falloff
        strength_scale *= 0.70 + field.stability * 0.42

        if field.level == 0:
            force[mask] += -direction * (self.config.repulsion_strength * strength_scale * falloff[:, None])
        elif field.level == 1:
            force[mask] += -direction * (self.config.weak_repulsion_strength * strength_scale * falloff[:, None])
        else:
            if field.level == 2:
                attract = self.config.attraction_strength
                vortex = self.config.vortex_strength * 0.55
            elif field.level == 3:
                attract = self.config.attraction_strength * 1.45
                vortex = self.config.vortex_strength * 1.45
            else:
                attract = self.config.converge_strength
                vortex = self.config.vortex_strength * 0.34
            force[mask] += direction * (attract * strength_scale * falloff[:, None])
            # Vortex / tangential force: skip in minimal perf mode
            if self.perf_mode < 2:
                hv = hand_vel.astype(np.float32)
                if float(np.linalg.norm(hv)) < 0.02:
                    hv = np.array([0.0, 0.25, 0.10], dtype=np.float32)
                tangent = np.cross(direction, hv[None, :])
                tangent_norm = np.linalg.norm(tangent, axis=1) + 1e-5
                tangent = tangent / tangent_norm[:, None]
                force[mask] += tangent * (vortex * field.stability * strength_scale * falloff[:, None])
        return dist

    def _apply_bounds(self, force: np.ndarray) -> None:
        c = self.config
        force[:, 0] += np.where(self.pos[:, 0] < -c.world_x, (-c.world_x - self.pos[:, 0]) * 5.0, 0.0)
        force[:, 0] += np.where(self.pos[:, 0] > c.world_x, (c.world_x - self.pos[:, 0]) * 5.0, 0.0)
        force[:, 1] += np.where(self.pos[:, 1] < -c.world_y, (-c.world_y - self.pos[:, 1]) * 5.0, 0.0)
        force[:, 1] += np.where(self.pos[:, 1] > c.world_y, (c.world_y - self.pos[:, 1]) * 5.0, 0.0)
        force[:, 2] += np.where(self.pos[:, 2] < c.world_z_far, (c.world_z_far - self.pos[:, 2]) * 3.8, 0.0)
        force[:, 2] += np.where(self.pos[:, 2] > 0.65, (0.65 - self.pos[:, 2]) * 4.5, 0.0)

    def _spawn_trails(self, field: FieldState, audio: AudioEngine) -> None:
        if field.ema_speed > self.config.entry_threshold:
            return
        if field.right_active:
            self._spawn_trail_for_hand(field.right_pos, field.right_vel, field, "right", audio)
        if field.left_active:
            self._spawn_trail_for_hand(field.left_pos, field.left_vel, field, "left", audio)

    def _spawn_trail_for_hand(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        field: FieldState,
        side: str,
        audio: AudioEngine,
    ) -> None:
        last_name = "last_right_trail" if side == "right" else "last_left_trail"
        last = getattr(self, last_name)
        if last is None:
            setattr(self, last_name, pos.copy())
            return
        moved = float(np.linalg.norm(pos - last))
        if moved < self.config.trail_trigger_dist:
            return
        setattr(self, last_name, pos.copy())
        count = self.config.trail_particles_per_trigger + (2 if field.level >= 3 else 0)
        if side == "left":
            count = max(2, int(count * self.config.secondary_hand_strength))
        self._add_trails(pos, vel, count, field)
        audio.trail(float(pos[1]), field.ema_speed, field.ga)

    def _add_trails(self, pos: np.ndarray, vel: np.ndarray, count: int, field: FieldState) -> None:
        idx = (np.arange(count) + self.trail_cursor) % self.config.max_trail_particles
        self.trail_cursor = int((self.trail_cursor + count) % self.config.max_trail_particles)
        life = self.rng.uniform(1.35, self.config.trail_particle_lifetime, count).astype(np.float32)
        self.trail_life[idx] = life
        self.trail_max_life[idx] = life
        self.trail_pos[idx] = pos[None, :] + self.rng.normal(0.0, 0.11, (count, 3)).astype(np.float32)
        self.trail_vel[idx] = vel[None, :] * 0.18 + self.rng.normal(0.0, 0.075, (count, 3)).astype(np.float32)
        ink = self._ink_color(field.ema_speed)
        self.trail_color[idx] = ink[None, :] * (0.80 + self.rng.random((count, 1)).astype(np.float32) * 0.35)
        self.trail_size[idx] = self.rng.uniform(5.0, 10.5, count).astype(np.float32)

    def _ink_color(self, speed: float) -> np.ndarray:
        slow = np.clip(1.0 - speed / max(self.config.entry_threshold, 1e-5), 0.0, 1.0)
        cold = np.array([56.0, 57.0, 57.0], dtype=np.float32)
        warm = np.array([78.0, 58.0, 38.0], dtype=np.float32)
        return cold * (1.0 - slow) + warm * slow

    def _update_trails(self, dt: float) -> None:
        mask = self.trail_life > 0.0
        if not np.any(mask):
            return
        self.trail_life[mask] -= dt
        drift = np.array([0.0, 0.018, -0.022], dtype=np.float32)
        self.trail_pos[mask] += (self.trail_vel[mask] + drift[None, :]) * dt

    def _spawn_gold(self, field: FieldState) -> None:
        if field.ga < 0.50 or field.level < 2:
            return
        rate = 2.0 + (field.ga - 0.50) * 14.0
        self.gold_accumulator += rate / 60.0
        while self.gold_accumulator >= 1.0:
            self.gold_accumulator -= 1.0
            self._add_gold(field.right_pos, field.ga)

    def _add_gold(self, pos: np.ndarray, ga: float) -> None:
        idx = self.gold_cursor
        self.gold_cursor = (self.gold_cursor + 1) % self.config.max_golden_particles
        life = float(self.rng.uniform(2.4, self.config.golden_particle_lifetime))
        self.gold_life[idx] = life
        self.gold_max_life[idx] = life
        self.gold_pos[idx] = pos + self.rng.normal(0.0, 0.18, 3).astype(np.float32)
        self.gold_vel[idx] = np.array(
            [
                self.rng.uniform(-0.18, 0.18),
                self.rng.uniform(0.48, 1.35),
                self.rng.uniform(-0.12, 0.18),
            ],
            dtype=np.float32,
        )
        self.gold_size[idx] = float(self.rng.uniform(3.0, 7.0) * (0.7 + ga * 0.55))

    def _update_gold(self, dt: float) -> None:
        mask = self.gold_life > 0.0
        if not np.any(mask):
            return
        self.gold_life[mask] -= dt
        wobble = np.sin(self.time * 2.2 + self.gold_pos[mask, 1])[:, None]
        self.gold_pos[mask] += (self.gold_vel[mask] + np.c_[wobble[:, 0] * 0.045, wobble[:, 0] * 0.0, wobble[:, 0] * 0.025]) * dt

    def _add_bloom_particles(self, center: np.ndarray) -> None:
        count = min(self.config.bloom_particle_count, self.config.max_bloom_particles)
        idx = (np.arange(count) + self.bloom_cursor) % self.config.max_bloom_particles
        self.bloom_cursor = int((self.bloom_cursor + count) % self.config.max_bloom_particles)
        direction = self.rng.normal(0.0, 1.0, (count, 3)).astype(np.float32)
        direction[:, 1] += 0.9
        direction[:, 2] += 0.45
        direction /= np.linalg.norm(direction, axis=1)[:, None] + 1e-5
        life = self.rng.uniform(0.75, 1.55, count).astype(np.float32)
        self.bloom_life[idx] = life
        self.bloom_max_life[idx] = life
        self.bloom_pos[idx] = center[None, :] + direction * self.rng.uniform(0.03, 0.30, (count, 1)).astype(np.float32)
        self.bloom_vel[idx] = direction * self.rng.uniform(3.8, 8.4, (count, 1)).astype(np.float32)
        self.bloom_size[idx] = self.rng.uniform(4.0, 9.0, count).astype(np.float32)

    def _update_bloom_particles(self, dt: float) -> None:
        mask = self.bloom_life > 0.0
        if not np.any(mask):
            return
        self.bloom_life[mask] -= dt
        self.bloom_vel[mask] *= 0.955
        self.bloom_pos[mask] += self.bloom_vel[mask] * dt

    # ── FX particles (special move effects) ──────────────────────────

    def add_fx_burst(
        self, origin: np.ndarray, direction: np.ndarray, count: int = 120,
        force: float = 18.0, lifetime: float = 2.0,
        color_rgb: tuple[float, float, float] = (220.0, 140.0, 35.0),
        screen_pos: tuple[int, int] | None = None,
    ) -> None:
        """Spawn a dramatic directional burst of FX particles + shockwave ring."""
        if self.fx_cooldown > 0.0:
            return
        count = min(count, self.fx_max)
        idx = (np.arange(count) + self.fx_cursor) % self.fx_max
        self.fx_cursor = int((self.fx_cursor + count) % self.fx_max)
        life = self.rng.uniform(lifetime * 0.4, lifetime, count).astype(np.float32)
        self.fx_life[idx] = life
        self.fx_max_life[idx] = life
        # Direction with wide spread for dramatic explosion look
        base_dir = direction / (np.linalg.norm(direction) + 1e-5)
        spread = self.rng.normal(0.0, 0.45, (count, 3)).astype(np.float32)
        dirs = base_dir[None, :] + spread
        dirs /= np.linalg.norm(dirs, axis=1)[:, None] + 1e-5
        self.fx_pos[idx] = origin[None, :] + self.rng.normal(0.0, 0.10, (count, 3)).astype(np.float32)
        speeds = self.rng.uniform(force * 0.4, force * 1.3, count).astype(np.float32)
        self.fx_vel[idx] = dirs * speeds[:, None]
        # Large particles for visibility
        self.fx_size[idx] = self.rng.uniform(9.0, 22.0, count).astype(np.float32)
        self.fx_color[idx] = np.array(color_rgb, dtype=np.float32)[None, :] * (
            0.70 + self.rng.random((count, 1)).astype(np.float32) * 0.40
        )
        self.fx_cooldown = 0.45  # shorter cooldown for more bursts

        # Shockwave ring
        if screen_pos is not None:
            self.shockwaves.append({
                "sx": screen_pos[0], "sy": screen_pos[1],
                "radius": 8.0, "max_radius": 180.0,
                "alpha": 220.0, "life": 0.55,
            })

    def _update_fx_particles(self, dt: float) -> None:
        mask = self.fx_life > 0.0
        if np.any(mask):
            self.fx_life[mask] -= dt
            self.fx_vel[mask] *= 0.93
            self.fx_vel[mask, 1] += 0.55 * dt  # stronger upward drift
            self.fx_pos[mask] += self.fx_vel[mask] * dt

        # Update shockwave rings
        survived = []
        for sw in self.shockwaves:
            sw["life"] -= dt
            sw["radius"] += 320.0 * dt  # expand speed
            sw["alpha"] = max(0.0, sw["alpha"] - 420.0 * dt)
            if sw["life"] > 0.0 and sw["alpha"] > 0.0:
                survived.append(sw)
        self.shockwaves = survived

    def fx_draw_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = self.fx_life > 0.0
        if not np.any(mask):
            return self._empty()
        ratio = np.clip(self.fx_life[mask] / self.fx_max_life[mask], 0.0, 1.0)
        # Warm gold → fade to dark brown
        colors = self.fx_color[mask] * ratio[:, None]
        alpha = 155.0 * ratio
        return self.fx_pos[mask], colors, self.fx_size[mask] * (0.35 + ratio * 0.78), alpha

    def _empty(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
