from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import pygame

from .config import Config


LEVEL_NAMES = ("repel", "escape", "attract", "orbit", "converge")


@dataclass
class FieldState:
    right_pos: np.ndarray
    left_pos: np.ndarray
    right_vel: np.ndarray
    left_vel: np.ndarray
    right_active: bool
    left_active: bool
    ema_speed: float
    stability: float
    level: int
    radius: float
    ga: float
    stillness_frames: int
    bloom_ready: bool


class MotionState:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.right_pos = np.array([0.0, 0.0, -2.0], dtype=np.float32)
        self.left_pos = np.array([-1.5, 0.0, -2.0], dtype=np.float32)
        self.right_prev = self.right_pos.copy()
        self.left_prev = self.left_pos.copy()
        self.right_vel = np.zeros(3, dtype=np.float32)
        self.left_vel = np.zeros(3, dtype=np.float32)
        self.right_active = True
        self.left_active = False
        self.ema_speed = 0.0
        self.ga = 0.0
        self.stillness_frames = 0
        self.speed_history: deque[float] = deque(maxlen=config.stability_window)

    def update(
        self,
        right_pos: np.ndarray | None,
        left_pos: np.ndarray | None,
        dt: float,
    ) -> FieldState:
        dt = max(dt, 1e-4)
        if right_pos is not None:
            self.right_prev = self.right_pos.copy()
            self.right_pos = right_pos.astype(np.float32)
            self.right_active = True
        if left_pos is not None:
            self.left_prev = self.left_pos.copy()
            self.left_pos = left_pos.astype(np.float32)
            self.left_active = True

        self.right_vel = (self.right_pos - self.right_prev) / dt
        self.left_vel = (self.left_pos - self.left_prev) / dt
        dominant_speed = float(np.linalg.norm(self.right_vel) / self.config.speed_normalizer)
        if self.left_active:
            left_speed = float(np.linalg.norm(self.left_vel) / self.config.speed_normalizer)
            dominant_speed = max(dominant_speed, left_speed * self.config.secondary_hand_strength)

        a = self.config.ema_alpha
        self.ema_speed = a * self.ema_speed + (1.0 - a) * dominant_speed
        self.speed_history.append(self.ema_speed)
        stability = self._stability()

        if self.ema_speed > self.config.entry_threshold:
            level = 0
            radius = self.config.influence_radius_base
        elif self.ema_speed > self.config.attract_threshold:
            level = 1
            radius = self.config.influence_radius_escape
        elif self.ema_speed > self.config.vortex_threshold:
            level = 2
            radius = self.config.influence_radius_attract
        elif self.ema_speed > self.config.converge_threshold:
            level = 3
            radius = self.config.influence_radius_vortex
        else:
            level = 4
            radius = self.config.influence_radius_max

        if self.ema_speed <= self.config.converge_threshold * 1.55:
            self.stillness_frames += 1
        else:
            self.stillness_frames = 0

        if stability > self.config.ga_stability_threshold and level >= 2:
            self.ga = min(1.0, self.ga + self.config.ga_fill_rate * dt)
        else:
            self.ga = max(0.0, self.ga - self.config.ga_drain_rate * dt)

        return FieldState(
            right_pos=self.right_pos.copy(),
            left_pos=self.left_pos.copy(),
            right_vel=self.right_vel.copy(),
            left_vel=self.left_vel.copy(),
            right_active=self.right_active,
            left_active=self.left_active,
            ema_speed=self.ema_speed,
            stability=stability,
            level=level,
            radius=radius,
            ga=self.ga,
            stillness_frames=self.stillness_frames,
            bloom_ready=self.stillness_frames >= self.config.stillness_frames,
        )

    def _stability(self) -> float:
        if len(self.speed_history) < 8:
            return 0.75
        values = np.array(self.speed_history, dtype=np.float32)
        variance = float(np.var(values))
        return max(0.0, min(1.0, 1.0 / (1.0 + variance * 85.0)))


class MouseInput:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.motion = MotionState(config)

    def update(self, dt: float) -> FieldState:
        mx, my = pygame.mouse.get_pos()
        pos = self._screen_to_world(mx, my)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            return self.motion.update(None, pos, dt)
        return self.motion.update(pos, None, dt)

    def _screen_to_world(self, mx: int, my: int) -> np.ndarray:
        z = -1.9
        depth = max(1.0, self.config.camera_z - z)
        world_scale = self.config.projection_scale / depth
        x = (mx - self.config.width * 0.5) / world_scale
        y = (self.config.height * 0.53 - my) / world_scale
        return np.array([x, y, z], dtype=np.float32)


class ScriptInput:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.motion = MotionState(config)
        self.time = 0.0

    def update(self, dt: float) -> FieldState:
        self.time += dt
        t = self.time
        if t < 1.0:
            u = t / 1.0
            x = -4.2 + 8.4 * u
            y = 0.7 + math.sin(u * math.pi * 3.0) * 0.35
        elif t < 4.6:
            u = (t - 1.0) / 3.6
            x = 3.2 - 2.4 * u
            y = 0.9 + math.sin(u * math.pi * 2.0) * 0.45
        elif t < 7.6:
            u = (t - 4.6) / 3.0
            angle = u * math.pi * 1.1
            x = 0.8 + math.cos(angle) * 0.28
            y = 0.55 + math.sin(angle) * 0.28
        else:
            x = 0.62
            y = 0.5
        right = np.array([x, y, -1.9], dtype=np.float32)
        left = np.array([x - 1.15, y - 0.12, -2.1], dtype=np.float32)
        return self.motion.update(right, left, dt)


class CameraInput:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.motion = MotionState(config)
        self.fallback = MouseInput(config)
        self.cap = None
        self.pose = None
        self.message = "camera active"
        try:
            import cv2
            import mediapipe as mp

            self.cv2 = cv2
            self.mp_pose = mp.solutions.pose
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.message = "camera could not be opened; using mouse"
                self.cap = None
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
            self.pose = self.mp_pose.Pose(
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.message = f"camera/mediapipe unavailable; using mouse ({exc})"
            self.cap = None
            self.pose = None

    def update(self, dt: float) -> FieldState:
        if self.cap is None or self.pose is None:
            return self.fallback.update(dt)

        ok, frame = self.cap.read()
        if not ok:
            self.message = "camera frame missing; using mouse"
            return self.fallback.update(dt)

        frame = self.cv2.flip(frame, 1)
        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return self.motion.update(None, None, dt)

        landmarks = result.pose_landmarks.landmark
        right = self._landmark_to_world(landmarks[16])
        left = self._landmark_to_world(landmarks[15])
        if landmarks[16].visibility < 0.35:
            right = None
        if landmarks[15].visibility < 0.35:
            left = None
        return self.motion.update(right, left, dt)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.pose is not None:
            self.pose.close()

    def _landmark_to_world(self, landmark) -> np.ndarray:
        x = (float(landmark.x) - 0.5) * self.config.world_x * 2.0
        y = (0.5 - float(landmark.y)) * self.config.world_y * 2.0
        z = -2.0 + float(landmark.z) * 1.15
        return np.array([x, y, z], dtype=np.float32)
