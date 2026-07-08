from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    width: int = 1280
    height: int = 720
    fps: int = 60

    # Motion mapping. This normalizer makes hand/mouse movement feel natural
    # while keeping the design-doc thresholds meaningful.
    speed_normalizer: float = 6.2
    ema_alpha: float = 0.58
    stability_window: int = 60
    entry_threshold: float = 1.0
    attract_threshold: float = 0.18
    vortex_threshold: float = 0.045
    converge_threshold: float = 0.016

    repulsion_strength: float = 25.0
    weak_repulsion_strength: float = 4.8
    attraction_strength: float = 14.0
    vortex_strength: float = 5.8
    converge_strength: float = 13.5
    secondary_hand_strength: float = 0.62

    influence_radius_base: float = 1.25
    influence_radius_escape: float = 1.8
    influence_radius_attract: float = 3.0
    influence_radius_vortex: float = 4.1
    influence_radius_max: float = 7.0

    world_x: float = 5.0
    world_y: float = 3.0
    world_z_near: float = 0.0
    world_z_far: float = -8.0

    dark_particle_count: int = 3600
    max_trail_particles: int = 1100
    max_golden_particles: int = 260
    max_bloom_particles: int = 100
    trail_particle_lifetime: float = 2.8
    golden_particle_lifetime: float = 4.6

    drag_coefficient: float = 0.76
    drift_amplitude: float = 0.13
    activation_rate: float = 0.72
    deactivation_rate: float = 0.24
    trail_trigger_dist: float = 0.08
    trail_particles_per_trigger: int = 5

    stillness_frames: int = 58
    bloom_cooldown_s: float = 2.8
    bloom_explosion_force: float = 11.5
    bloom_particle_count: int = 92

    ga_fill_rate: float = 0.075
    ga_drain_rate: float = 0.10
    ga_stability_threshold: float = 0.70
    breath_interval: float = 3.4
    breath_brightness_delta: float = 0.06

    projection_scale: float = 620.0
    camera_z: float = 9.0
    space_color: tuple[int, int, int] = (250, 250, 247)

    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    audio_enabled: bool = True
    bloom_enabled: bool = True

    # Performance degradation thresholds
    perf_fps_warn: int = 35       # below this: reduce drift + vortex
    perf_fps_critical: int = 25   # below this: basic forces only
    perf_fps_ema_alpha: float = 0.12  # smoothing for FPS tracking
