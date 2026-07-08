from __future__ import annotations

import pygame

from .input_pose import FieldState, LEVEL_NAMES


class DebugOverlay:
    def __init__(self) -> None:
        self.enabled = True
        self.font: pygame.font.Font | None = None

    def ensure_font(self) -> None:
        if self.font is None:
            self.font = pygame.font.Font(None, 18)

    def draw(self, screen: pygame.Surface, field: FieldState, fps: float, particle_count: int, audio_on: bool, perf_tier: int = 0) -> None:
        if not self.enabled:
            return
        self.ensure_font()
        assert self.font is not None
        tier_names = {0: "full", 1: "reduced", 2: "minimal"}
        lines = [
            f"FPS {fps:5.1f}  perf {tier_names.get(perf_tier, '?')}",
            f"mode {LEVEL_NAMES[field.level]}  speed {field.ema_speed:.4f}",
            f"stability {field.stability:.2f}  GA {field.ga:.2f}",
            f"still {field.stillness_frames:03d}  particles {particle_count}",
            f"audio {'on' if audio_on else 'off'}",
            "mouse: right hand | Shift+mouse: left hand | D: debug | B: bloom | M: mute",
        ]
        y = 14
        for line in lines:
            surf = self.font.render(line, True, (205, 211, 220))
            bg = pygame.Surface((surf.get_width() + 10, surf.get_height() + 4), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 90))
            screen.blit(bg, (12, y - 2))
            screen.blit(surf, (17, y))
            y += 20
