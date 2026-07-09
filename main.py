#!/usr/bin/env python3
"""
演卦 · 星尘太极 (Python 版)
=============================
太极手势 → 星尘粒子实时交互系统

技术栈: MediaPipe + OpenCV + Pygame + moderngl (OpenGL)

三大运动 → 物理映射:
  手速 → 粘性 (慢=粘稠拖尾 / 快=干脆迸裂)
  曲直 → 旋涡 (直=两侧排开 / 弧=涡旋牵引)
  纵深 → 呼吸 (前推=膨胀扩散 / 后拉=坍缩吸纳)

运行:
  pip install -r requirements.txt
  python main.py
"""

import sys
import time
import numpy as np

import cv2
import pygame
import moderngl
import mediapipe as mp

# ============================================================
# 常量
# ============================================================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
PARTICLE_COUNT = 3000
INFLUENCE_RADIUS = 3.5
MAX_SPEED = 2.5
BOUNDS_X = 7.5
BOUNDS_Y = 5.5
BOUNDS_Z = 4.5
SMOOTH_ALPHA = 0.35
HISTORY_SIZE = 45
PRESENCE_HYSTERESIS = 8
BASE_DAMPING = 0.985

# ============================================================
# 运动特征分析器
# ============================================================
class MotionAnalyzer:
    """从手部关键点序列提取: speed / curvature / z-velocity"""

    def __init__(self):
        self.history = []          # [(pos, timestamp, has_hand), ...]
        self.speed = 0.0
        self.curvature = 0.0
        self.z_velocity = 0.0
        self.hand_velocity = np.zeros(3, dtype=np.float32)
        self.hand_world_pos = None
        self.last_direction = np.zeros(3, dtype=np.float32)
        self.hand_detected = False
        self.presence_counter = 0

    # ---- 对外接口 ----
    def process_landmarks(self, hands_data, timestamp):
        """输入 MediaPipe 手部数据，更新内部特征"""
        has_hand = hands_data is not None and len(hands_data) > 0

        if has_hand:
            palm = hands_data[0]['palm_center']
            pos = np.array([palm['x'], palm['y'], palm['z']], dtype=np.float32)
            self.presence_counter = min(self.presence_counter + 1, PRESENCE_HYSTERESIS)
        else:
            pos = np.zeros(3, dtype=np.float32)
            self.presence_counter = max(self.presence_counter - 1, 0)

        self.hand_detected = self.presence_counter >= PRESENCE_HYSTERESIS // 2
        self.history.append((pos, timestamp, has_hand))

        # 限制历史长度
        if len(self.history) > HISTORY_SIZE:
            self.history = self.history[-HISTORY_SIZE:]

        if self.hand_detected and has_hand:
            self._compute_features(pos)
        else:
            # 无手时特征衰减
            self.speed *= 0.9
            self.curvature *= 0.85
            self.z_velocity *= 0.85
            self.hand_velocity *= 0.9

    def get_features(self):
        return {
            'speed': self.speed,
            'curvature': self.curvature,
            'z_velocity': self.z_velocity,
            'hand_velocity': self.hand_velocity.copy(),
            'hand_detected': self.hand_detected,
        }

    def get_hand_world_position(self):
        return self.hand_world_pos

    # ---- 内部计算 ----
    def _compute_features(self, current_pos):
        recent = [(p, t) for p, t, h in reversed(self.history) if h]
        if len(recent) < 2:
            return

        # --- 速度 ---
        total_v = np.zeros(3, dtype=np.float32)
        total_w = 0.0
        for i in range(len(recent) - 1):
            cp, ct = recent[i]
            pp, pt = recent[i + 1]
            dt = ct - pt
            if dt > 0.001:
                w = 1.0 / (1.0 + i * 0.3)
                total_v += ((cp - pp) / dt) * w
                total_w += w

        if total_w > 0:
            raw_v = total_v / total_w
            raw_speed = float(np.linalg.norm(raw_v))
            self.speed += (raw_speed - self.speed) * SMOOTH_ALPHA
            self.hand_velocity += (raw_v - self.hand_velocity) * SMOOTH_ALPHA

        # --- 曲率 ---
        if self.speed > 0.003:
            v_norm = self.hand_velocity / (self.speed + 0.0001)
            dot = float(np.clip(np.dot(v_norm, self.last_direction), -1.0, 1.0))
            angle = np.arccos(dot)
            raw_curv = angle * min(self.speed * 25.0, 1.0)
            self.curvature += (raw_curv - self.curvature) * SMOOTH_ALPHA
            self.last_direction = v_norm
        else:
            self.curvature *= 0.85

        # --- 纵深速度 ---
        if len(recent) >= 3:
            np_ct = recent[0]
            op_ot = recent[min(len(recent) - 1, 5)]
            z_dt = np_ct[1] - op_ot[1]
            if z_dt > 0.01:
                raw_zv = (np_ct[0][2] - op_ot[0][2]) / z_dt
                self.z_velocity += (raw_zv - self.z_velocity) * SMOOTH_ALPHA

        # --- 世界坐标映射 ---
        self.hand_world_pos = np.array([
            (current_pos[0] - 0.5) * (BOUNDS_X * 2),
            (0.5 - current_pos[1]) * (BOUNDS_Y * 2),
            current_pos[2] * 6.0,
        ], dtype=np.float32)


# ============================================================
# 摄像头 + MediaPipe 手势追踪
# ============================================================
class HandTracker:
    """OpenCV 摄像头 + MediaPipe Hands 推理"""

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def get_frame_and_hands(self):
        """读取一帧 → 镜像 → MediaPipe → 返回 (帧, 手部数据)"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame = cv2.flip(frame, 1)  # 镜像
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        hands_data = []
        if results.multi_hand_landmarks:
            for i, hand_lm in enumerate(results.multi_hand_landmarks):
                wrist = hand_lm.landmark[0]
                middle_mcp = hand_lm.landmark[9]
                hands_data.append({
                    'handedness': (
                        results.multi_handedness[i].classification[0].label
                        if results.multi_handedness else 'Unknown'
                    ),
                    'palm_center': {
                        'x': (wrist.x + middle_mcp.x) / 2,
                        'y': (wrist.y + middle_mcp.y) / 2,
                        'z': (wrist.z + middle_mcp.z) / 2,
                    },
                })

        return frame, (hands_data if hands_data else None)

    def release(self):
        self.cap.release()
        self.hands.close()


# ============================================================
# 粒子系统 (moderngl / OpenGL 3.3)
# ============================================================
class ParticleSystem:
    """GPU 粒子渲染 —— 3000 星尘粒子 + 手势力场物理"""

    def __init__(self, ctx, count=PARTICLE_COUNT):
        self.ctx = ctx
        self.count = count
        self.rng = np.random.default_rng()

        # CPU 端数组
        self.positions = np.zeros((count, 3), dtype=np.float32)
        self.velocities = np.zeros((count, 3), dtype=np.float32)
        self.colors = np.zeros((count, 3), dtype=np.float32)
        self.sizes = np.zeros(count, dtype=np.float32)
        self.base_colors = np.zeros((count, 3), dtype=np.float32)

        self._init_particles()
        self._init_shaders()
        self._init_buffers()
        self._init_texture()

    # ================================================================
    # 初始化
    # ================================================================

    def _init_particles(self):
        c = self.count
        rng = self.rng

        # 球形随机分布
        theta = rng.uniform(0, 2 * np.pi, c)
        phi = np.arccos(rng.uniform(-1, 1, c))
        radius = rng.uniform(0, 1, c) * BOUNDS_X * 0.7 + rng.uniform(0, 1, c) * BOUNDS_X * 0.3

        self.positions[:, 0] = np.cos(theta) * np.sin(phi) * radius
        self.positions[:, 1] = np.cos(phi) * radius * 0.7
        self.positions[:, 2] = np.sin(theta) * np.sin(phi) * radius * 0.6

        self.velocities = (rng.uniform(-1, 1, (c, 3)) * 0.05).astype(np.float32)

        # 基色: 蓝紫系
        hue = rng.uniform(0.58, 0.75, c)
        sat = rng.uniform(0.5, 0.8, c)
        lum = rng.uniform(0.15, 0.45, c)
        for i in range(c):
            r, g, b = self._hsl2rgb(hue[i], sat[i], lum[i])
            self.base_colors[i] = [r, g, b]
            self.colors[i] = [r, g, b]

        self.sizes = rng.uniform(0.03, 0.12, c).astype(np.float32)

    def _init_shaders(self):
        """编译 GLSL 3.30 着色器"""
        vs = """
            #version 330
            in vec3 in_position;
            in vec3 in_color;
            in float in_size;
            out vec3 v_color;
            out float v_size;
            uniform mat4 projection;

            void main() {
                gl_Position = projection * vec4(in_position, 1.0);
                gl_PointSize = in_size * 120.0;
                gl_PointSize = clamp(gl_PointSize, 0.5, 50.0);
                v_color = in_color;
                v_size = in_size;
            }
        """

        fs = """
            #version 330
            in vec3 v_color;
            in float v_size;
            out vec4 fragColor;
            uniform sampler2D glow_tex;

            void main() {
                float alpha = texture(glow_tex, gl_PointCoord).r;
                float brightness = 1.0 + v_size * 4.0;
                fragColor = vec4(v_color * brightness * alpha, alpha);
            }
        """

        self.program = self.ctx.program(vertex_shader=vs, fragment_shader=fs)

        # 正交投影矩阵: 世界 → 裁剪空间
        self.projection = np.array([
            [1 / BOUNDS_X, 0, 0, 0],
            [0, 1 / BOUNDS_Y, 0, 0],
            [0, 0, 1 / BOUNDS_Z, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.program['projection'].write(self.projection.tobytes())

    def _init_buffers(self):
        """创建 VBO + VAO (交错布局: xyz rgb size = 7 floats/粒子)"""
        dtype = np.dtype([('pos', 'f4', (3,)), ('col', 'f4', (3,)), ('siz', 'f4')])
        data = np.empty(self.count, dtype=dtype)
        data['pos'] = self.positions
        data['col'] = self.colors
        data['siz'] = self.sizes

        self.vbo = self.ctx.buffer(data.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 1f', 'in_position', 'in_color', 'in_size')],
        )

    def _init_texture(self):
        """生成径向渐变发光纹理 (128×128)"""
        size = 128
        ys, xs = np.mgrid[0:size, 0:size]
        cx = cy = size / 2
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / (size / 2)

        arr = np.where(dist < 0.02, 1.0,
              np.where(dist < 0.10, 0.95 * (1.0 - (dist - 0.02) / 0.08),
              np.where(dist < 0.30, 0.30 * (1.0 - (dist - 0.10) / 0.20),
              np.where(dist < 0.65, 0.06 * (1.0 - (dist - 0.30) / 0.35),
              0.0))))
        arr = np.flipud(arr.astype(np.float32))

        self.glow_tex = self.ctx.texture((size, size), 1, data=arr.tobytes(), dtype='f4')
        self.glow_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.glow_tex.use(0)

    # ================================================================
    # 物理更新
    # ================================================================

    def update(self, dt, hand_pos, features, hand_detected):
        dt = min(dt, 0.1)
        c = self.count

        # 跟踪哪些粒子被手影响
        influenced = np.zeros(c, dtype=bool)

        # ---- 环境力 (全部粒子, 向量化) ----
        self.velocities += (self.rng.uniform(-0.5, 0.5, (c, 3)) * 0.03).astype(np.float32)
        self.velocities -= self.positions * 0.00015  # 弱中心引力

        # ---- 手势影响力 ----
        if hand_detected and hand_pos is not None:
            diff = self.positions - hand_pos
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            mask = dist < INFLUENCE_RADIUS

            if np.any(mask):
                influenced = mask.copy()
                m = np.sum(mask)

                md = diff[mask]                              # (m, 3)
                mdist = dist[mask]                           # (m,)
                mv = self.velocities[mask]                   # (m, 3)

                # 衰减曲线 (smoothstep)
                t = 1.0 - mdist / INFLUENCE_RADIUS
                falloff_1d = t ** 2 * (3.0 - 2.0 * t)
                falloff = falloff_1d[:, np.newaxis]          # (m, 1)

                # 径向单位向量
                rn = md / (mdist[:, np.newaxis] + 0.0001)

                # ---- 力 1: 粘性 (速度耦合) ----
                norm_speed = min(features['speed'] / MAX_SPEED, 1.0)
                viscosity = 1.0 - norm_speed
                mv += features['hand_velocity'] * (falloff * viscosity * 0.12)

                # 快移飞溅
                mv += rn * (falloff * norm_speed * 0.18)

                # ---- 力 2: 旋涡 (径向 × 速度方向) ----
                hv = features['hand_velocity']
                hv_len = float(np.linalg.norm(hv)) + 0.0001
                hv_norm = hv / hv_len
                cross = np.cross(rn, hv_norm)
                mv += cross * (falloff * features['curvature'] * 0.1)

                # ---- 力 3: 呼吸 (径向膨胀/收缩) ----
                mv += rn * (falloff * features['z_velocity'] * 0.08)

                self.velocities[mask] = mv

                # ---- 颜色 ----
                activity = np.clip(
                    norm_speed * 0.5 +
                    features['curvature'] * 2.5 +
                    abs(features['z_velocity']) * 8.0,
                    0.0, 1.0
                )
                blend = falloff_1d * 0.8
                warm = np.array([
                    0.6 + activity * 0.4,
                    0.5 + activity * 0.4,
                    0.8 - activity * 0.5,
                ], dtype=np.float32)

                self.colors[mask, 0] = self.base_colors[mask, 0] * (1 - blend) + warm[0] * blend
                self.colors[mask, 1] = self.base_colors[mask, 1] * (1 - blend) + warm[1] * blend
                self.colors[mask, 2] = self.base_colors[mask, 2] * (1 - blend) + warm[2] * blend

                # ---- 大小 ----
                target = 0.04 + falloff_1d * viscosity * 0.35 + activity * 0.06
                self.sizes[mask] += (target - self.sizes[mask]) * 0.15

        # ---- 不受影响的粒子回归基态 ----
        not_inf = ~influenced
        if np.any(not_inf):
            self.colors[not_inf] += (self.base_colors[not_inf] - self.colors[not_inf]) * 0.02
            self.sizes[not_inf] += (0.07 - self.sizes[not_inf]) * 0.015

        # ---- 位置积分 ----
        self.positions += self.velocities * dt

        # ---- 速度阻尼 ----
        self.velocities *= BASE_DAMPING

        # ---- 软边界 ----
        for dim, bound in [(0, BOUNDS_X), (1, BOUNDS_Y), (2, BOUNDS_Z)]:
            high = self.positions[:, dim] > bound
            low = self.positions[:, dim] < -bound
            self.velocities[high, dim] -= (self.positions[high, dim] - bound) * 0.08
            self.velocities[low, dim] -= (self.positions[low, dim] + bound) * 0.08

        # ---- 上传 GPU ----
        self._upload()

    def _upload(self):
        dtype = np.dtype([('pos', 'f4', (3,)), ('col', 'f4', (3,)), ('siz', 'f4')])
        data = np.empty(self.count, dtype=dtype)
        data['pos'] = self.positions
        data['col'] = self.colors
        data['siz'] = self.sizes
        self.vbo.write(data.tobytes())

    # ================================================================
    # 渲染
    # ================================================================

    def render(self):
        self.ctx.clear(0.02, 0.02, 0.06, 1.0)  # 深空底色
        self.vao.render(moderngl.POINTS)

    # ================================================================
    # 工具
    # ================================================================

    @staticmethod
    def _hsl2rgb(h, s, l):
        """HSL → RGB, 所有值 0..1"""
        a = s * min(l, 1 - l)
        def f(n):
            k = (n + h * 12) % 12
            return l - a * max(-1.0, min(k - 3.0, min(9.0 - k, 1.0)))
        return [f(0), f(8), f(4)]


# ============================================================
# 主程序
# ============================================================
def main():
    # --- 初始化 Pygame + moderngl ---
    pygame.init()
    pygame.display.set_caption("演卦 · 星尘太极")
    screen = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
    )

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)       # 叠加混合 → 光尘累积
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    # --- 子系统 ---
    print("Camera...", end=" ", flush=True)
    tracker = HandTracker()
    print("OK")

    print("Particles...", end=" ", flush=True)
    analyzer = MotionAnalyzer()
    particles = ParticleSystem(ctx, PARTICLE_COUNT)
    print("OK")

    print("=" * 50)
    print("  YanGua / Stardust Taichi - READY")
    print("  [ESC] quit  [F] fullscreen")
    print("  Raise your hand, let the stardust move with you")
    print("=" * 50)

    # --- 主循环 ---
    clock = pygame.time.Clock()
    running = True
    last_time = time.perf_counter()
    frame_count = 0
    fps_time = time.perf_counter()
    fps_frames = 0

    while running:
        # 事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
            elif event.type == pygame.VIDEORESIZE:
                # moderngl 自动处理视口
                pass

        # 时间
        current_time = time.perf_counter()
        dt = current_time - last_time
        last_time = current_time

        # 摄像头 → 手势
        frame, hands_data = tracker.get_frame_and_hands()

        # 手势 → 运动特征
        analyzer.process_landmarks(hands_data, current_time)
        features = analyzer.get_features()
        hand_pos = analyzer.get_hand_world_position()

        # 运动特征 → 粒子物理
        particles.update(dt, hand_pos, features, features['hand_detected'])

        # 渲染
        particles.render()
        pygame.display.flip()

        # FPS 统计
        frame_count += 1
        fps_frames += 1
        if current_time - fps_time >= 3.0:
            fps = fps_frames / (current_time - fps_time)
            status = "TRACKING" if features['hand_detected'] else "waiting..."
            pygame.display.set_caption(
                f"YanGua | {status} | FPS: {fps:.0f}"
            )
            fps_time = current_time
            fps_frames = 0

    # --- 清理 ---
    tracker.release()
    pygame.quit()
    print("\nYanGua - exited")


if __name__ == '__main__':
    main()
