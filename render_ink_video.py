from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import pygame

from bagua.audio import AudioEngine
from bagua.config import Config
from bagua.input_pose import FieldState, MotionState
from bagua.move_announcer import MoveAnnouncer
from bagua.particles import ParticleSystem
from bagua.pose_matcher import DEFAULT_LANDMARK_IDS, PoseMatch, PoseMatcher, normalize_landmarks
from bagua.renderer import LEVEL_COLORS, Renderer


POSE_CONNECTIONS = tuple(mp.solutions.pose.POSE_CONNECTIONS)


class SilentAudio(AudioEngine):
    def __init__(self, config: Config) -> None:
        self.config = config
        self.enabled = False

    def trail(self, pitch_y: float, speed: float, ga: float) -> None:
        return None

    def bloom(self, pitch_y: float) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pose skeleton with ink particles over a video.")
    parser.add_argument("--input", required=True, help="Source video path.")
    parser.add_argument("--output", required=True, help="Output mp4 path.")
    parser.add_argument("--particles", type=int, default=3200)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Optional short preview limit.")
    parser.add_argument("--show-skeleton", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--announce-moves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--move-effects", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recognition-mode", choices=("all", "left-kick", "none"), default="left-kick")
    parser.add_argument("--move-confidence", type=float, default=0.62)
    parser.add_argument("--move-confirm-frames", type=int, default=4)
    parser.add_argument("--move-cooldown", type=float, default=1.6)
    parser.add_argument("--detect-start", type=float, default=0.0, help="Do not trigger move recognition before this source-video time.")
    parser.add_argument("--frame-step", type=int, default=1, help="Write one out of every N source frames.")
    parser.add_argument("--output-fps", type=float, default=0.0, help="Override output fps; defaults to source_fps/frame_step.")
    return parser.parse_args()


class LeftKickDetector:
    """Stateful detector for 转身左蹬腿/左蹬脚.

    It ignores video text entirely. The trigger is based on MediaPipe body
    geometry: left foot/knee height relative to the support leg, foot extension,
    and a target-template sanity score. Once fired, it stays locked until the
    kicking leg drops back down, so one held kick cannot trigger repeatedly.
    """

    def __init__(
        self,
        matcher: PoseMatcher,
        output_fps: float,
        confidence_threshold: float = 0.55,
        confirm_frames: int = 3,
    ) -> None:
        self.matcher = matcher
        self.confidence_threshold = confidence_threshold
        self.confirm_frames = max(1, confirm_frames)
        self.target_names = {"第十五式 转身左蹬脚", "转身左蹬脚", "转身左蹬腿"}
        self.active_frames = 0
        self.reset_frames = 0
        self.locked = False
        self.cooldown_frames = 0
        self.cooldown_after_reset = max(1, int(round(output_fps * 0.35)))
        self.last_score = 0.0

    def update(self, landmarks) -> PoseMatch | None:
        self.cooldown_frames = max(0, self.cooldown_frames - 1)
        geometry_score, template_score = self._score(landmarks)
        score = geometry_score * 0.72 + template_score * 0.28
        self.last_score = score

        if self.locked:
            if self._leg_has_reset(landmarks):
                self.reset_frames += 1
                if self.reset_frames >= 5:
                    self.locked = False
                    self.active_frames = 0
                    self.reset_frames = 0
                    self.cooldown_frames = self.cooldown_after_reset
            else:
                self.reset_frames = 0
            return None

        if self.cooldown_frames > 0:
            return None

        if score >= self.confidence_threshold and geometry_score >= 0.62 and template_score >= 0.38:
            self.active_frames += 1
            if self.active_frames >= self.confirm_frames:
                self.locked = True
                self.active_frames = 0
                self.reset_frames = 0
                return PoseMatch(
                    name="转身左蹬腿",
                    confidence=score,
                    frame_count=self.confirm_frames,
                    keypoint_scores={
                        "geometry": geometry_score,
                        "template": template_score,
                    },
                )
        else:
            self.active_frames = max(0, self.active_frames - 1)
        return None

    def _score(self, landmarks) -> tuple[float, float]:
        normalized = normalize_landmarks(landmarks, DEFAULT_LANDMARK_IDS, 0.25)
        left_foot = normalized.get("left_foot")
        right_foot = normalized.get("right_foot")
        left_knee = normalized.get("left_knee")
        right_knee = normalized.get("right_knee")
        left_ankle = normalized.get("left_ankle")
        right_ankle = normalized.get("right_ankle")
        left_hip = normalized.get("left_hip")
        if not all((left_foot, right_foot, left_knee, right_knee, left_ankle, right_ankle, left_hip)):
            return 0.0, 0.0

        foot_gap = left_foot[1] - right_foot[1]
        knee_gap = left_knee[1] - right_knee[1]
        ankle_gap = left_ankle[1] - right_ankle[1]
        foot_above_hip = left_foot[1] < left_hip[1] + 0.05
        foot_extended = abs(left_foot[0] - left_hip[0]) > 0.34
        support_grounded = right_foot[1] > 0.24 and right_ankle[1] > 0.20

        criteria = [
            score_below(foot_gap, -0.42, -0.72) * 2.4,
            score_below(knee_gap, -0.24, -0.46) * 1.8,
            score_below(ankle_gap, -0.34, -0.62) * 1.6,
            (1.0 if foot_above_hip else 0.0) * 1.3,
            (1.0 if foot_extended else 0.0) * 1.2,
            (1.0 if support_grounded else 0.0) * 1.0,
        ]
        geometry_score = float(sum(criteria) / 9.3)

        template_score = 0.0
        matches = self.matcher.match(landmarks)
        for match in matches:
            if match.name in self.target_names or "左蹬脚" in match.name:
                template_score = max(template_score, match.confidence)
        return geometry_score, template_score

    def _leg_has_reset(self, landmarks) -> bool:
        normalized = normalize_landmarks(landmarks, DEFAULT_LANDMARK_IDS, 0.25)
        left_foot = normalized.get("left_foot")
        right_foot = normalized.get("right_foot")
        left_knee = normalized.get("left_knee")
        right_knee = normalized.get("right_knee")
        if not all((left_foot, right_foot, left_knee, right_knee)):
            return False
        foot_gap = left_foot[1] - right_foot[1]
        knee_gap = left_knee[1] - right_knee[1]
        return foot_gap > -0.16 and knee_gap > -0.12


def score_below(value: float, start: float, full: float) -> float:
    if value >= start:
        return 0.0
    if value <= full:
        return 1.0
    return float((start - value) / max(start - full, 1e-6))


class KickImpactOverlay:
    """Screen-space ink impact for the recognized kick moment."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.events: list[KickImpactEvent] = []
        self.rng = np.random.default_rng(20260708)

    def trigger(self, origin: np.ndarray, direction: np.ndarray) -> None:
        sx, sy = project_world(origin, self.config)
        sign = 1.0 if float(direction[0]) >= 0 else -1.0
        screen_dir = np.array([sign, -0.18], dtype=np.float32)
        screen_dir /= np.linalg.norm(screen_dir) + 1e-5
        normal = np.array([-screen_dir[1], screen_dir[0]], dtype=np.float32)

        splats = []
        for _ in range(86):
            forward = float(self.rng.uniform(25.0, 430.0))
            sideways = float(self.rng.normal(0.0, 76.0))
            radius = float(self.rng.uniform(2.0, 11.0) ** 1.08)
            delay = float(self.rng.uniform(0.0, 0.22))
            alpha = int(self.rng.uniform(55, 170))
            splats.append((forward, sideways, radius, delay, alpha))

        self.events.append(KickImpactEvent(
            sx=float(sx),
            sy=float(sy),
            direction=screen_dir,
            normal=normal,
            life=1.05,
            max_life=1.05,
            splats=splats,
        ))

    def update(self, dt: float) -> None:
        survived = []
        for event in self.events:
            event.life -= dt
            if event.life > 0.0:
                survived.append(event)
        self.events = survived

    def draw(self, overlay: pygame.Surface) -> None:
        if not self.events:
            return
        for event in self.events:
            self._draw_event(overlay, event)

    def _draw_event(self, overlay: pygame.Surface, event: "KickImpactEvent") -> None:
        w, h = overlay.get_size()
        t = 1.0 - event.life / event.max_life
        fade = float(np.clip(1.0 - t, 0.0, 1.0))
        burst = pygame.Surface((w, h), pygame.SRCALPHA)
        origin = np.array([event.sx, event.sy], dtype=np.float32)

        # Impact flash and expanding ring.
        ring_radius = int(34 + t * 245)
        ring_alpha = int(120 * fade)
        if ring_alpha > 0:
            ring_rect = (
                int(event.sx - ring_radius * 1.22),
                int(event.sy - ring_radius * 0.62),
                int(ring_radius * 2.44),
                int(ring_radius * 1.24),
            )
            pygame.draw.ellipse(burst, (12, 11, 9, ring_alpha), ring_rect, max(2, int(8 * fade)))
            pygame.draw.circle(burst, (142, 104, 42, int(46 * fade)), (int(event.sx), int(event.sy)), int(48 + t * 34), 3)

        # Heavy brush slash in the kick direction.
        slash_len = 225 + t * 245
        for i in range(22):
            u = i / 21.0
            center = origin + event.direction * (slash_len * u) + event.normal * np.sin(u * np.pi * 2.2) * (18.0 * (1.0 - u))
            radius = int((34.0 * (1.0 - u) + 6.0) * (0.72 + fade * 0.55))
            alpha = int((178 * fade) * (1.0 - u * 0.62))
            if alpha <= 0 or radius <= 0:
                continue
            pygame.draw.circle(burst, (8, 9, 9, alpha), (int(center[0]), int(center[1])), radius)
            if i % 3 == 0:
                pygame.draw.circle(burst, (126, 91, 38, int(alpha * 0.22)), (int(center[0]), int(center[1])), max(3, radius // 2))

        # Tapered ink hairlines.
        for i in range(7):
            offset = (i - 3) * 18.0
            start = origin + event.normal * offset
            end = start + event.direction * (300 + 36 * i) + event.normal * np.sin(t * 4.0 + i) * 24.0
            alpha = int(90 * fade * (1.0 - abs(i - 3) / 5.0))
            pygame.draw.line(burst, (10, 10, 9, alpha), (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), max(1, int(5 - abs(i - 3) * 0.8)))

        # Flying splatters with staggered reveal.
        for forward, sideways, radius, delay, base_alpha in event.splats:
            local_t = np.clip((t - delay) / 0.78, 0.0, 1.0)
            if local_t <= 0.0:
                continue
            pos = origin + event.direction * (forward * (0.55 + local_t * 0.70)) + event.normal * sideways
            alpha = int(base_alpha * fade * (0.35 + local_t * 0.65))
            if alpha <= 0:
                continue
            r = max(1, int(radius * (0.72 + local_t * 0.95)))
            pygame.draw.circle(burst, (9, 10, 10, alpha), (int(pos[0]), int(pos[1])), r)

        overlay.blit(burst, (0, 0))


class KickImpactEvent:
    __slots__ = ("sx", "sy", "direction", "normal", "life", "max_life", "splats")

    def __init__(
        self,
        sx: float,
        sy: float,
        direction: np.ndarray,
        normal: np.ndarray,
        life: float,
        max_life: float,
        splats: list[tuple[float, float, float, float, int]],
    ) -> None:
        self.sx = sx
        self.sy = sy
        self.direction = direction
        self.normal = normal
        self.life = life
        self.max_life = max_life
        self.splats = splats


def landmark_to_world(landmark, config: Config) -> np.ndarray | None:
    if float(landmark.visibility) < 0.28:
        return None
    z = -2.0 + float(np.clip(landmark.z, -0.75, 0.75)) * 0.85
    depth = max(1.0, config.camera_z - z)
    world_scale = config.projection_scale / depth
    x = (float(landmark.x) * config.width - config.width * 0.5) / world_scale
    y = (config.height * 0.53 - float(landmark.y) * config.height) / world_scale
    return np.array([x, y, z], dtype=np.float32)


def project_world(pos: np.ndarray, config: Config) -> tuple[int, int]:
    dist = max(1.0, config.camera_z - float(pos[2]))
    world_scale = config.projection_scale / dist
    sx = config.width * 0.5 + float(pos[0]) * world_scale
    sy = config.height * 0.53 - float(pos[1]) * world_scale
    return int(round(sx)), int(round(sy))


def surface_to_rgba(surface: pygame.Surface) -> Image.Image:
    raw = pygame.image.tostring(surface, "RGBA")
    return Image.frombytes("RGBA", surface.get_size(), raw)


def draw_clouds(renderer: Renderer, particles: ParticleSystem, field: FieldState) -> pygame.Surface:
    c = renderer.config
    renderer.layer.fill((0, 0, 0, 0))
    renderer.glow.fill((0, 0, 0, 0))
    renderer.ring.fill((0, 0, 0, 0))

    dark = tighten_dark_cloud(*particles.dark_draw_data(field.ga), field=field)
    trail = heavy_ink_cloud(*particles.trail_draw_data(), alpha_gain=2.15, size_gain=0.78)
    gold = heavy_ink_cloud(*particles.gold_draw_data(), alpha_gain=1.55, size_gain=0.70)
    bloom = heavy_ink_cloud(*particles.bloom_draw_data(), alpha_gain=1.45, size_gain=0.68)

    renderer._draw_cloud(*dark, glow_gain=0.08 + field.ga * 0.10)
    renderer._draw_cloud(*trail, glow_gain=0.42)
    renderer._draw_cloud(*gold, glow_gain=0.48)
    renderer._draw_cloud(*bloom, glow_gain=0.55)
    renderer._draw_cloud(*particles.fx_draw_data(), glow_gain=2.60)

    overlay = pygame.Surface((c.width, c.height), pygame.SRCALPHA)
    if c.bloom_enabled:
        small = pygame.transform.smoothscale(renderer.glow, (max(1, c.width // 3), max(1, c.height // 3)))
        blurred = pygame.transform.smoothscale(small, (c.width, c.height))
        overlay.blit(blurred, (0, 0))
    overlay.blit(renderer.layer, (0, 0))
    renderer._draw_field_rings(overlay, field)
    return overlay


def tighten_dark_cloud(
    pos: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
    alpha: np.ndarray,
    field: FieldState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(pos) == 0:
        return pos, colors, sizes, alpha

    mask = alpha > 70.0
    for hand, active in ((field.right_pos, field.right_active), (field.left_pos, field.left_active)):
        if not active:
            continue
        dist = np.linalg.norm(pos - hand[None, :], axis=1)
        mask |= dist < 1.35

    ink = np.array([14.0, 15.0, 16.0], dtype=np.float32)
    colors = colors[mask] * 0.24 + ink[None, :] * 0.76
    return pos[mask], colors, sizes[mask] * 0.58, np.clip(alpha[mask] * 1.75, 0.0, 245.0)


def heavy_ink_cloud(
    pos: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
    alpha: np.ndarray,
    alpha_gain: float,
    size_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(pos) == 0:
        return pos, colors, sizes, alpha
    ink = np.array([12.0, 13.0, 14.0], dtype=np.float32)
    muted_warm = np.array([70.0, 48.0, 28.0], dtype=np.float32)
    warmth = np.clip(colors[:, 0:1] / 180.0, 0.0, 1.0)
    target = ink[None, :] * (1.0 - warmth * 0.22) + muted_warm[None, :] * (warmth * 0.22)
    colors = colors * 0.18 + target * 0.82
    return pos, colors, sizes * size_gain, np.clip(alpha * alpha_gain, 0.0, 255.0)


def draw_skeleton(
    overlay: pygame.Surface,
    landmarks,
    config: Config,
    right_world: np.ndarray | None,
    left_world: np.ndarray | None,
) -> None:
    pts: list[tuple[int, int, float]] = []
    for lm in landmarks:
        x = int(np.clip(float(lm.x) * config.width, -2000, config.width + 2000))
        y = int(np.clip(float(lm.y) * config.height, -2000, config.height + 2000))
        pts.append((x, y, float(lm.visibility)))

    ink = (12, 13, 14, 178)
    wash = (10, 10, 9, 62)
    warm = (78, 48, 24, 150)
    for a, b in POSE_CONNECTIONS:
        if pts[a][2] < 0.35 or pts[b][2] < 0.35:
            continue
        p0 = (pts[a][0], pts[a][1])
        p1 = (pts[b][0], pts[b][1])
        pygame.draw.line(overlay, wash, p0, p1, 7)
        pygame.draw.line(overlay, ink, p0, p1, 3)

    for idx in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        if pts[idx][2] < 0.35:
            continue
        pygame.draw.circle(overlay, (10, 10, 9, 165), (pts[idx][0], pts[idx][1]), 5)
        pygame.draw.circle(overlay, (112, 82, 45, 130), (pts[idx][0], pts[idx][1]), 2)

    for world, color in ((right_world, LEVEL_COLORS[3]), (left_world, warm[:3])):
        if world is None:
            continue
        x, y = project_world(world, config)
        pygame.draw.circle(overlay, (*color, 34), (x, y), 26, 2)
        pygame.draw.circle(overlay, (12, 12, 10, 190), (x, y), 6)


def composite_frame(frame_bgr: np.ndarray, overlay: pygame.Surface) -> np.ndarray:
    base = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    base.alpha_composite(surface_to_rgba(overlay))
    return cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)


def has_audio(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return bool(result.stdout.strip())
    except OSError:
        return False


def mux_audio(source: Path, silent_video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(silent_video),
    ]
    if not has_audio(source):
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", str(output)]
        subprocess.run(cmd, check=True)
        return
    cmd += [
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    subprocess.run(cmd, check=True)


def select_move_match(
    matches: list[PoseMatch],
    landmarks,
    matcher: PoseMatcher,
) -> PoseMatch | None:
    """Choose a move match, giving high-kick leg geometry priority."""
    if not matches:
        return None

    best = matches[0]
    normalized = normalize_landmarks(landmarks, DEFAULT_LANDMARK_IDS, 0.25)
    left_foot = normalized.get("left_foot")
    right_foot = normalized.get("right_foot")
    left_knee = normalized.get("left_knee")
    right_knee = normalized.get("right_knee")
    if left_foot and right_foot and left_knee and right_knee:
        foot_gap = left_foot[1] - right_foot[1]
        knee_gap = left_knee[1] - right_knee[1]
        strong_left_raise = foot_gap < -0.38 or knee_gap < -0.30
        strong_right_raise = foot_gap > 0.38 or knee_gap > 0.30
        if strong_left_raise:
            kick = next((m for m in matches if "左蹬脚" in m.name), None)
            if kick is not None and kick.confidence >= best.confidence - 0.18:
                return kick
        if strong_right_raise:
            kick = next((m for m in matches if "右蹬脚" in m.name), None)
            if kick is not None and kick.confidence >= best.confidence - 0.18:
                return kick
    return best


def trigger_move_effect(
    particles: ParticleSystem,
    match: PoseMatch,
    landmarks,
    config: Config,
) -> tuple[np.ndarray, np.ndarray] | None:
    template = particles.config  # keep type checkers calm about Config use below
    del template

    left_foot = landmark_to_world(landmarks[31], config)
    right_foot = landmark_to_world(landmarks[32], config)
    left_wrist = landmark_to_world(landmarks[15], config)
    right_wrist = landmark_to_world(landmarks[16], config)

    origin = left_wrist if left_wrist is not None else right_wrist
    direction = np.array([0.0, -0.20, -0.85], dtype=np.float32)

    if ("左蹬脚" in match.name or "左蹬腿" in match.name) and left_foot is not None:
        origin = left_foot
        direction = np.array([0.55, -0.35, -0.55], dtype=np.float32)
    elif ("右蹬脚" in match.name or "右蹬腿" in match.name) and right_foot is not None:
        origin = right_foot
        direction = np.array([-0.55, -0.35, -0.55], dtype=np.float32)
    elif left_foot is not None and right_foot is not None:
        origin = left_foot if left_foot[1] > right_foot[1] else right_foot
        side = 1.0 if origin is left_foot else -1.0
        direction = np.array([0.35 * side, -0.20, -0.45], dtype=np.float32)

    if origin is None:
        return None

    particles.trigger_move_explosion(
        origin=origin,
        direction=direction,
        radius=3.35,
        max_particles=760,
        splatter_fraction=0.80,
        force=24.0,
        converge_delay=0.07,
    )
    screen_pos = project_world(origin, config)
    particles.add_fx_burst(
        origin=origin,
        direction=direction,
        count=260,
        force=24.0,
        lifetime=1.25,
        color_rgb=(92.0, 70.0, 38.0),
        screen_pos=screen_pos,
    )
    return origin, direction


def render_video(args: argparse.Namespace) -> None:
    source = Path(args.input)
    output = Path(args.output)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_step = max(1, int(args.frame_step))
    output_fps = float(args.output_fps) if args.output_fps > 0 else fps / frame_step
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.max_seconds > 0:
        total = min(total, int(round(args.max_seconds * fps))) if total else int(round(args.max_seconds * fps))
    output_total = int(np.ceil(total / frame_step)) if total else 0

    config = Config(
        width=width,
        height=height,
        fps=max(1, int(round(output_fps))),
        dark_particle_count=args.particles,
        max_trail_particles=1800,
        trail_particles_per_trigger=8,
        trail_particle_lifetime=2.15,
        speed_normalizer=17.0,
        influence_radius_base=0.92,
        influence_radius_escape=1.22,
        influence_radius_attract=2.05,
        influence_radius_vortex=2.70,
        influence_radius_max=3.35,
        drift_amplitude=0.045,
        audio_enabled=False,
        projection_scale=max(420.0, min(width, height) * 0.80),
        world_y=6.5 if height > width else 3.0,  # taller world for high kicks
        world_x=5.5,
    )

    pygame.init()
    renderer = Renderer(config)
    particles = ParticleSystem(config)
    motion = MotionState(config)
    audio = SilentAudio(config)
    kick_impact = KickImpactOverlay(config)

    temp_dir = Path(tempfile.mkdtemp(prefix="bagua_ink_"))
    silent_video = temp_dir / "silent.mp4"
    writer = cv2.VideoWriter(
        str(silent_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create output video writer.")

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    )

    # -- Pose matcher + move announcer --------------------------------
    matcher = PoseMatcher(reference_dir="reference_poses")
    announcer = MoveAnnouncer()
    left_kick_detector = LeftKickDetector(
        matcher=matcher,
        output_fps=output_fps,
        confidence_threshold=args.move_confidence,
        confirm_frames=args.move_confirm_frames,
    )
    move_active_frames = {name: 0 for name in matcher.known_moves}
    move_cooldowns = {name: 0 for name in matcher.known_moves}
    global_move_cooldown = 0
    move_cooldown_frames = max(1, int(round(args.move_cooldown * output_fps)))
    global_move_cooldown_frames = max(1, int(round(0.55 * output_fps)))

    dt = frame_step / fps
    frame_idx = 0
    output_idx = 0
    detected = 0
    move_triggered = 0
    try:
        while True:
            if args.max_seconds > 0 and frame_idx >= total:
                break
            ok, frame = cap.read()
            if not ok:
                break
            source_frame_idx = frame_idx
            frame_idx += 1
            if source_frame_idx % frame_step != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)

            right = None
            left = None
            landmarks = None
            if result.pose_landmarks:
                detected += 1
                landmarks = result.pose_landmarks.landmark
                right = landmark_to_world(landmarks[16], config)
                left = landmark_to_world(landmarks[15], config)

            # -- Generic 24-form detection ----------------------------
            for move_name in list(move_cooldowns):
                move_cooldowns[move_name] = max(0, move_cooldowns[move_name] - 1)
            global_move_cooldown = max(0, global_move_cooldown - 1)

            if (
                args.announce_moves
                and result.pose_landmarks
                and args.recognition_mode == "left-kick"
                and source_frame_idx / fps >= args.detect_start
            ):
                best = left_kick_detector.update(landmarks)
                if best is not None:
                    announcer.announce(
                        move_name=best.name,
                        screen_width=width,
                        screen_height=height,
                    )
                    if args.move_effects:
                        impact = trigger_move_effect(particles, best, landmarks, config)
                        if impact is not None:
                            kick_impact.trigger(*impact)
                    move_triggered += 1
                    print(
                        f"move trigger {move_triggered}: {best.name} "
                        f"at {source_frame_idx / fps:.2f}s "
                        f"score={best.confidence:.3f}"
                    )
            elif (
                args.announce_moves
                and result.pose_landmarks
                and args.recognition_mode == "all"
                and matcher.known_moves
                and global_move_cooldown == 0
                and source_frame_idx / fps >= args.detect_start
            ):
                matches = matcher.match(landmarks)
                best = select_move_match(matches, landmarks, matcher)
                if (
                    best is not None
                    and best.confidence >= args.move_confidence
                    and move_cooldowns.get(best.name, 0) == 0
                ):
                    move_active_frames[best.name] = move_active_frames.get(best.name, 0) + 1
                    for move_name in list(move_active_frames):
                        if move_name != best.name:
                            move_active_frames[move_name] = max(0, move_active_frames[move_name] - 1)
                    if move_active_frames[best.name] >= args.move_confirm_frames:
                        announcer.announce(
                            move_name=best.name,
                            screen_width=width,
                            screen_height=height,
                        )
                        if args.move_effects:
                            impact = trigger_move_effect(particles, best, landmarks, config)
                            if impact is not None:
                                kick_impact.trigger(*impact)
                        move_triggered += 1
                        move_cooldowns[best.name] = move_cooldown_frames
                        global_move_cooldown = global_move_cooldown_frames
                        move_active_frames[best.name] = 0
                else:
                    for move_name in list(move_active_frames):
                        move_active_frames[move_name] = max(0, move_active_frames[move_name] - 1)

            field = motion.update(right, left, dt)
            if right is None:
                field.right_active = False
            if left is None:
                field.left_active = False

            particles.update(dt, field, audio)
            announcer.update(dt)
            kick_impact.update(dt)
            overlay = draw_clouds(renderer, particles, field)
            # Draw shockwave rings (ink-colored now) on top
            renderer.draw_shockwaves(overlay, particles.shockwaves, field.ga)
            kick_impact.draw(overlay)
            if args.show_skeleton and landmarks is not None:
                draw_skeleton(overlay, landmarks, config, right, left)
            # ── Move announcement UI (drawn LAST, on top of everything) ──
            announcer.draw(overlay)

            writer.write(composite_frame(frame, overlay))
            output_idx += 1
            if output_idx % max(1, int(output_fps * 2)) == 0:
                pct = (source_frame_idx / total * 100.0) if total else 0.0
                print(
                    f"processed {output_idx}/{output_total or '?'} output frames "
                    f"({pct:.1f}%), pose hits: {detected}, moves: {move_triggered}"
                )
    finally:
        pose.close()
        cap.release()
        writer.release()
        pygame.quit()

    if args.audio:
        mux_audio(source, silent_video, output)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        silent_video.replace(output)
    print(f"saved: {output}")


def main() -> None:
    render_video(parse_args())


if __name__ == "__main__":
    main()
