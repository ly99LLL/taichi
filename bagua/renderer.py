from __future__ import annotations

import numpy as np
import pygame

from .config import Config
from .input_pose import FieldState
from .particles import ParticleSystem


LEVEL_COLORS = (
    (178, 58, 45),
    (128, 92, 70),
    (58, 96, 126),
    (122, 93, 42),
    (74, 67, 56),
)


class Renderer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.layer = pygame.Surface((config.width, config.height), pygame.SRCALPHA)
        self.glow = pygame.Surface((config.width, config.height), pygame.SRCALPHA)
        self.ring = pygame.Surface((config.width, config.height), pygame.SRCALPHA)

    def render(self, screen: pygame.Surface, particles: ParticleSystem, field: FieldState) -> None:
        screen.fill(self.config.space_color)
        self._draw_backwash(screen, field.ga)

        self.layer.fill((0, 0, 0, 0))
        self.glow.fill((0, 0, 0, 0))

        self._draw_cloud(*particles.dark_draw_data(field.ga), glow_gain=0.25 + field.ga * 0.24)
        self._draw_cloud(*particles.trail_draw_data(), glow_gain=1.10)
        self._draw_cloud(*particles.gold_draw_data(), glow_gain=1.55)
        self._draw_cloud(*particles.bloom_draw_data(), glow_gain=2.20)
        self._draw_cloud(*particles.fx_draw_data(), glow_gain=2.80)

        if self.config.bloom_enabled:
            small = pygame.transform.smoothscale(
                self.glow,
                (max(1, self.config.width // 3), max(1, self.config.height // 3)),
            )
            blurred = pygame.transform.smoothscale(small, (self.config.width, self.config.height))
            screen.blit(blurred, (0, 0))

        screen.blit(self.layer, (0, 0))
        self._draw_field_rings(screen, field)
        self._draw_ga_lamp(screen, field.ga)

    def _draw_cloud(
        self,
        pos: np.ndarray,
        colors: np.ndarray,
        sizes: np.ndarray,
        alpha: np.ndarray,
        glow_gain: float,
    ) -> None:
        if len(pos) == 0:
            return
        sx, sy, scale, depth, mask = self._project(pos)
        if not np.any(mask):
            return
        indices = np.where(mask)[0]
        order = indices[np.argsort(pos[indices, 2])]
        for i in order:
            x = int(sx[i])
            y = int(sy[i])
            radius = int(max(1.0, sizes[i] * scale[i]))
            a = int(np.clip(alpha[i] * depth[i], 0, 245))
            if a <= 0:
                continue
            color = colors[i]
            rgba = (
                int(np.clip(color[0], 0, 255)),
                int(np.clip(color[1], 0, 255)),
                int(np.clip(color[2], 0, 255)),
                a,
            )
            pygame.draw.circle(self.layer, rgba, (x, y), radius)
            if glow_gain > 0.0 and a > 42:
                ga = int(np.clip(a * 0.08 * glow_gain, 0, 58))
                if ga > 0:
                    pygame.draw.circle(
                        self.glow,
                        (rgba[0], rgba[1], rgba[2], ga),
                        (x, y),
                        max(radius + 2, int(radius * (2.4 + glow_gain))),
                    )

    def _project(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dist = np.maximum(1.0, self.config.camera_z - pos[:, 2])
        world_scale = self.config.projection_scale / dist
        sx = self.config.width * 0.5 + pos[:, 0] * world_scale
        sy = self.config.height * 0.53 - pos[:, 1] * world_scale
        point_scale = 10.5 / dist
        depth = np.clip((pos[:, 2] - self.config.world_z_far) / abs(self.config.world_z_far), 0.0, 1.0)
        depth = 0.28 + depth * 0.82
        margin = 80
        mask = (
            (sx > -margin)
            & (sx < self.config.width + margin)
            & (sy > -margin)
            & (sy < self.config.height + margin)
        )
        return sx, sy, point_scale, depth, mask

    def _draw_field_rings(self, screen: pygame.Surface, field: FieldState) -> None:
        self.ring.fill((0, 0, 0, 0))
        for pos, active, strength in (
            (field.right_pos, field.right_active, 1.0),
            (field.left_pos, field.left_active, self.config.secondary_hand_strength),
        ):
            if not active:
                continue
            sx, sy, scale, _, mask = self._project(pos[None, :])
            if not bool(mask[0]):
                continue
            color = LEVEL_COLORS[field.level]
            radius = max(8, int(field.radius * self.config.projection_scale / max(1.0, self.config.camera_z - pos[2])))
            alpha = int((42 + field.ga * 22) * strength)
            pygame.draw.circle(self.ring, (*color, alpha), (int(sx[0]), int(sy[0])), radius, 1)
            pygame.draw.circle(self.ring, (*color, min(190, alpha + 70)), (int(sx[0]), int(sy[0])), 5)
            pygame.draw.circle(self.ring, (20, 22, 24, 210), (int(sx[0]), int(sy[0])), 2)
            pygame.draw.line(
                self.ring,
                (20, 22, 24, 130),
                (int(sx[0]) - 8, int(sy[0])),
                (int(sx[0]) + 8, int(sy[0])),
                1,
            )
            pygame.draw.line(
                self.ring,
                (20, 22, 24, 130),
                (int(sx[0]), int(sy[0]) - 8),
                (int(sx[0]), int(sy[0]) + 8),
                1,
            )
        screen.blit(self.ring, (0, 0))

    def _draw_ga_lamp(self, screen: pygame.Surface, ga: float) -> None:
        x, y = 44, self.config.height - 42
        alpha = int(42 + ga * 60)
        color = (104 + int(ga * 58), 82 + int(ga * 40), 42)
        pygame.draw.circle(screen, (*color, alpha), (x, y), 10, 1)
        if ga > 0.03:
            pygame.draw.circle(screen, (*color, int(alpha * 0.18)), (x, y), 20)

    def draw_shockwaves(
        self, screen: pygame.Surface, shockwaves: list[dict], ga: float = 0.0,
    ) -> None:
        """Draw expanding shockwave rings on screen (screen-space).

        Color follows the unified palette: warm muted gold when GA is high,
        deep ink otherwise. No vermillion red — that is reserved for the seal stamp.
        """
        for sw in shockwaves:
            r = int(sw["radius"])
            a = int(np.clip(sw["alpha"], 0, 200))
            if r <= 0 or a <= 0:
                continue
            # Outer ring: muted gold when qi is high, deep ink otherwise
            if ga > 0.5:
                ring_color = (105, 82, 40)  # warm muted gold
            else:
                ring_color = (25, 22, 18)   # deep ink
            pygame.draw.circle(
                screen, (*ring_color, a), (sw["sx"], sw["sy"]), r, max(2, r // 30),
            )
            # Inner ring: deep ink
            inner_r = max(1, r - 10)
            if inner_r > 0:
                pygame.draw.circle(
                    screen, (28, 24, 18, min(a, 130)),
                    (sw["sx"], sw["sy"]), inner_r, max(1, inner_r // 40),
                )

    def _draw_backwash(self, screen: pygame.Surface, ga: float) -> None:
        if ga <= 0.01:
            return
        overlay = pygame.Surface((self.config.width, self.config.height), pygame.SRCALPHA)
        overlay.fill((218, 200, 170, int(ga * 10)))
        screen.blit(overlay, (0, 0))
