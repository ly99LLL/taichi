"""Move announcement system — minimalist ink-wash UI overlay.

Design principle: 大道至简 / 留白.
A single calligraphic move name, a subtle vignette, and a ghosted
vermillion seal stamp branded into the negative space. No borders,
no shadows, no lines, no bilingual clutter.
"""

from __future__ import annotations

import numpy as np
import pygame


# Ink-wash palette — vermillion ONLY on the seal stamp
INK_DEEP = (18, 16, 14)
INK_MID = (42, 38, 32)
INK_LIGHT = (78, 72, 62)
VERMILLION = (180, 45, 35)  # 朱砂红 — seal stamp ONLY, nowhere else
PAPER_CREAM = (252, 248, 240)


class MoveAnnouncer:
    """Manages move-callout UI state and rendering."""

    def __init__(self) -> None:
        self.active: list[Announcement] = []
        self._font_large: pygame.font.Font | None = None
        self._font_seal: pygame.font.Font | None = None

    def announce(
        self, move_name: str, subtitle: str = "",
        screen_width: int = 816, screen_height: int = 1450,
    ) -> None:
        """Trigger a move announcement."""
        if len(self.active) >= 2:
            self.active.pop(0)
        self.active.append(Announcement(
            name=move_name,
            life=1.5,
            max_life=1.5,
            screen_w=screen_width,
            screen_h=screen_height,
        ))

    def update(self, dt: float) -> None:
        survived = []
        for a in self.active:
            a.life -= dt
            if a.life > 0:
                survived.append(a)
        self.active = survived

    def draw(self, screen: pygame.Surface) -> None:
        """Draw all active announcements on screen."""
        if not self.active:
            return

        w, h = screen.get_width(), screen.get_height()

        for a in self.active:
            # ── Clean fade: 0.3s in, 0.5s out ──────────────────
            if a.life > a.max_life - 0.3:
                alpha_mul = (a.max_life - a.life) / 0.3
            elif a.life < 0.5:
                alpha_mul = a.life / 0.5
            else:
                alpha_mul = 1.0
            alpha_mul = float(np.clip(alpha_mul, 0.0, 1.0))

            # ── Subtle vignette ────────────────────────────────
            vignette = pygame.Surface((w, h), pygame.SRCALPHA)
            vignette_alpha = int(30 * alpha_mul)
            for radius_pct in np.linspace(1.0, 0.25, 4):
                r_alpha = int(vignette_alpha * (1.0 - radius_pct) * 0.45)
                if r_alpha <= 0:
                    continue
                rx = int(w * radius_pct)
                ry = int(h * radius_pct)
                temp = pygame.Surface((w, h), pygame.SRCALPHA)
                pygame.draw.ellipse(
                    temp, (*INK_DEEP, r_alpha),
                    (w // 2 - rx // 2, h // 2 - ry // 2, rx, ry),
                )
                vignette.blit(temp, (0, 0))
            screen.blit(vignette, (0, 0))

            # ── Move name — pure calligraphy, no shadow ────────
            self._ensure_fonts()
            name_alpha = int(210 * alpha_mul)
            if name_alpha > 0 and self._font_large is not None:
                display_name = a.name.split(" ", 1)[-1]
                name_surf = self._font_large.render(
                    display_name, True, (*INK_DEEP, name_alpha),
                )
                name_rect = name_surf.get_rect(
                    center=(w // 2, int(h * 0.40)),
                )
                screen.blit(name_surf, name_rect)

            # ── Seal stamp — branded, no border ────────────────
            seal_alpha = int(80 * alpha_mul)
            if seal_alpha > 10 and self._font_seal is not None:
                seal_x = int(w * 0.82)
                seal_y = int(h * 0.57)
                seal_char = self._font_seal.render(
                    "武", True, (*VERMILLION, seal_alpha),
                )
                sc_rect = seal_char.get_rect(center=(seal_x, seal_y))
                screen.blit(seal_char, sc_rect)

    def _ensure_fonts(self) -> None:
        if self._font_large is not None:
            return
        try:
            candidates = [
                "C:/Windows/Fonts/STKAITI.TTF",  # 华文楷体
                "C:/Windows/Fonts/simkai.ttf",    # 楷体
                "C:/Windows/Fonts/simhei.ttf",    # 黑体
                "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            ]
            font_path = None
            for c in candidates:
                try:
                    pygame.font.Font(c, 12)
                    font_path = c
                    break
                except Exception:
                    continue

            if font_path:
                self._font_large = pygame.font.Font(font_path, 72)
                self._font_seal = pygame.font.Font(font_path, 28)
            else:
                self._font_large = pygame.font.Font(None, 72)
                self._font_seal = pygame.font.Font(None, 28)
        except Exception:
            self._font_large = pygame.font.Font(None, 72)
            self._font_seal = pygame.font.Font(None, 28)


class Announcement:
    __slots__ = ("name", "life", "max_life", "screen_w", "screen_h")

    def __init__(
        self, name: str, life: float, max_life: float,
        screen_w: int, screen_h: int,
    ) -> None:
        self.name = name
        self.life = life
        self.max_life = max_life
        self.screen_w = screen_w
        self.screen_h = screen_h
