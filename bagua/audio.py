from __future__ import annotations

import math
import time

import numpy as np
import pygame

from .config import Config


class AudioEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.enabled = config.audio_enabled
        self.last_trail = 0.0
        self.last_bloom = 0.0
        if not self.enabled:
            return
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
            pygame.mixer.init()
        except pygame.error:
            self.enabled = False

    def trail(self, y: float, speed: float, ga: float) -> None:
        now = time.perf_counter()
        if not self.enabled or now - self.last_trail < 0.055:
            return
        self.last_trail = now
        freq = self._pentatonic_freq(y)
        duration = 0.035 + max(0.0, 0.05 - speed * 0.03)
        self._play_tone(freq, duration, 0.08, harmonic=ga > 0.25)

    def bloom(self, y: float) -> None:
        now = time.perf_counter()
        if not self.enabled or now - self.last_bloom < 1.0:
            return
        self.last_bloom = now
        base = self._pentatonic_freq(y) * 0.5
        for idx, ratio in enumerate((1, 2, 3, 4, 5, 6)):
            self._play_tone(base * ratio, 0.75 + idx * 0.14, 0.06 / (idx + 1))

    def _pentatonic_freq(self, y: float) -> float:
        notes = np.array([261.63, 293.66, 329.63, 392.0, 440.0], dtype=np.float32)
        octave = int(np.clip((y + 3.0) / 6.0 * 3.0, 0.0, 2.99))
        degree = int(np.clip((y + 3.0) / 6.0 * len(notes), 0, len(notes) - 1))
        return float(notes[degree] * (2 ** octave))

    def _play_tone(self, freq: float, duration: float, volume: float, harmonic: bool = False) -> None:
        sample_rate = 44100
        count = max(1, int(sample_rate * duration))
        t = np.linspace(0.0, duration, count, endpoint=False, dtype=np.float32)
        wave = np.sin(t * freq * math.tau)
        if harmonic:
            wave += 0.18 * np.sin(t * freq * 2.0 * math.tau)
        attack = min(count, int(sample_rate * 0.006))
        release = min(count, int(sample_rate * duration * 0.75))
        envelope = np.ones(count, dtype=np.float32)
        if attack > 1:
            envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release > 1:
            envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
        mono = np.clip(wave * envelope * volume * 32767.0, -32767, 32767).astype(np.int16)
        stereo = np.column_stack([mono, mono])
        try:
            pygame.sndarray.make_sound(stereo).play()
        except pygame.error:
            self.enabled = False
