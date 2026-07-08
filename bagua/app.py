from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import pygame

from .audio import AudioEngine
from .config import Config
from .debug import DebugOverlay
from .input_pose import CameraInput, FieldState, MouseInput, ScriptInput
from .particles import ParticleSystem
from .renderer import Renderer


class BaguaApp:
    def __init__(self, config: Config, input_mode: str = "mouse") -> None:
        self.config = config
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((config.width, config.height))
        pygame.display.set_caption("演卦 - Python Particle Demo")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(config)
        self.particles = ParticleSystem(config)
        self.audio = AudioEngine(config)
        self.debug = DebugOverlay()
        self.input = self._make_input(input_mode)
        self.running = True
        self.last_field: FieldState | None = None

    def _make_input(self, input_mode: str):
        if input_mode == "script":
            return ScriptInput(self.config)
        if input_mode == "camera":
            camera = CameraInput(self.config)
            print(camera.message)
            return camera
        return MouseInput(self.config)

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(self.config.fps) / 1000.0
            self._handle_events()
            field = self.input.update(dt)
            self.last_field = field
            self.particles.update(dt, field, self.audio)
            self.renderer.render(self.screen, self.particles, field)
            self.debug.draw(
                self.screen,
                field,
                self.clock.get_fps(),
                self.particles.total_count,
                self.audio.enabled,
                self.particles.perf_mode,
            )
            pygame.display.flip()
        self._close_input()
        pygame.quit()

    def render_preview(self, output: str, frames: int) -> None:
        captures: list[tuple[str, Image.Image]] = []
        points = {
            max(1, int(frames * 0.12)): "fast / repel",
            max(2, int(frames * 0.38)): "slow / attract",
            max(3, int(frames * 0.66)): "orbit / gather",
        }
        bloom_capture_at: int | None = None
        dt = 1.0 / self.config.fps
        for frame in range(frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            field = self.input.update(dt)
            self.last_field = field
            did_bloom = self.particles.update(dt, field, self.audio)
            self.renderer.render(self.screen, self.particles, field)
            if did_bloom and bloom_capture_at is None:
                bloom_capture_at = min(frames - 1, frame + 5)
            if frame in points:
                captures.append((points[frame], self._surface_to_image(self.screen)))
            if bloom_capture_at is not None and frame == bloom_capture_at:
                captures.append(("fajin / bloom", self._surface_to_image(self.screen)))
                bloom_capture_at = None
        if len(captures) < 4:
            captures.append(("settled after bloom", self._surface_to_image(self.screen)))
        grid = self._make_preview_grid(captures)
        out = Path(output)
        if not out.is_absolute():
            out = Path.cwd() / out
        grid.save(out)
        print(f"preview saved: {out}")
        self._close_input()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_d:
                    self.debug.enabled = not self.debug.enabled
                elif event.key == pygame.K_m:
                    self.audio.enabled = not self.audio.enabled
                elif event.key == pygame.K_b and self.last_field is not None:
                    self.particles.trigger_bloom(self.last_field, self.audio)

    def _surface_to_image(self, surface: pygame.Surface) -> Image.Image:
        raw = pygame.image.tostring(surface, "RGB")
        return Image.frombytes("RGB", surface.get_size(), raw)

    def _make_preview_grid(self, captures: list[tuple[str, Image.Image]]) -> Image.Image:
        if not captures:
            return self._surface_to_image(self.screen)
        thumb_w = self.config.width // 2
        thumb_h = self.config.height // 2
        grid = Image.new("RGB", (thumb_w * 2, thumb_h * 2), self.config.space_color)
        draw = ImageDraw.Draw(grid)
        for idx, (label, image) in enumerate(captures[:4]):
            thumb = image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = (idx % 2) * thumb_w
            y = (idx // 2) * thumb_h
            grid.paste(thumb, (x, y))
            draw.rectangle((x + 12, y + 12, x + 172, y + 38), fill=(0, 0, 0))
            draw.text((x + 20, y + 18), label, fill=(220, 218, 206))
        return grid

    def _close_input(self) -> None:
        close = getattr(self.input, "close", None)
        if callable(close):
            close()
