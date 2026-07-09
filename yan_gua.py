#!/usr/bin/env python3
"""演卦 · 星尘太极 (py5 + Taichi GPU)

太极手势驱动的星尘粒子实时交互系统。不识别具体招式，读取手势"运动气质"
（手速/曲直/纵深）并翻译为粒子物理响应。

美学方向：暗宇宙星尘场 + 琥珀搅动 + 旋转星云 + 劲断意不断。

快捷键: [ESC] 退出  [F] 全屏  [D] 调试
"""

# ---- JDK 路径 (py5 依赖) ----
import os
os.environ['JAVA_HOME'] = 'C:/Program Files/Microsoft/jdk-17.0.19.10-hotspot'

# ---- 标准库 ----
import time
from collections import deque

# ---- 第三方库 ----
import cv2
import mediapipe as mp
import numpy as np
import py5
import taichi as ti

# ---- Taichi GPU 初始化 (纯计算, 不开启 GUI 窗口) ----
ti.init(arch=ti.cuda, random_seed=42)

# ============================================================
# 色彩常量 — 中式云气调色板
# ============================================================

# 背景: 深空暗墨
BG_R, BG_G, BG_B = 16, 12, 8

# 拖尾透明度 (值越小拖尾越长, 混沌感越强)
TRAIL_ALPHA = 7

# 粒子视觉范围
PARTICLE_ALPHA_MIN = 2
PARTICLE_ALPHA_MAX = 22
PARTICLE_SIZE_MIN = 3
PARTICLE_SIZE_MAX = 32

# 墨韵色阶: 深空星尘 — 克制琥珀调
INK_COLORS = [
    (180, 160, 130),   # 遥远星芒
    (140, 120, 95),    # 淡星尘
    (105, 85, 65),     # 中层云气
    (75, 58, 42),      # 暗流
    (48, 35, 24),      # 深空暗质
    (28, 18, 12),      # 虚空 (接近背景)
]
NUM_INK_LEVELS = len(INK_COLORS)

# 金色气韵 (手部活跃时混入)
WARM_ACCENT = (200, 155, 90)
WARM_LIGHT = (240, 210, 150)

# ---- 摄像头小窗 (右下角) ----
CAM_W = 280
CAM_H = 210
CAM_MARGIN = 20

# 水墨滤镜参数 (轻量版 — 人保持清晰可见)
CAM_BILATERAL_D = 5
CAM_BILATERAL_SIGMA = 40
CAM_WARM_BLEND = 0.35        # 暖色调混合比例 (越低越保留原色)
CAM_EDGE_STRENGTH = 0.2      # 墨线强度
CAM_VIGNETTE = 0.12          # 暗角强度

# 手部笔触参数
CAM_BRUSH_BLUR = 3           # 笔触扩散模糊
CAM_BRUSH_OPACITY = 0.35     # 笔触透明度
CAM_FINGERTIP_R = 4          # 指尖光点半径
CAM_JOINT_R = 2              # 关节光点半径

# ============================================================
# 系统常量
# ============================================================

# 窗口尺寸 — setup() 中被屏幕自适应覆盖
WINDOW_W, WINDOW_H = 1280, 720

# 粒子系统
PARTICLE_COUNT = 6000
INFLUENCE_RADIUS = 240        # 手掌影响范围 (1.5x 扩大)
MAX_SPEED = 800               # 归一化参考速度
CURVATURE_REF = 400           # 曲率参考速度 (更灵敏)
SMOOTH_ALPHA = 0.35           # EMA 平滑系数 (混沌流动感)
HISTORY_SIZE = 45             # 运动历史长度
BASE_DAMPING = 0.985          # 速度阻尼 (惯性尾迹)

# ============================================================
# Taichi GPU 粒子物理 Kernel
# ============================================================

@ti.kernel
def _particle_physics_kernel(
    # 粒子状态 (原地修改)
    px: ti.types.ndarray(dtype=ti.f32, ndim=1),
    py: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vx: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ink_level: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # 基准状态 (只读)
    base_alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_ink: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # 手部数据 (最多 2 只手, 未使用的填 0)
    hx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hspd_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hcurv_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hzvel_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hactive_arr: ti.types.ndarray(dtype=ti.i32, ndim=1),
    # 全局参数
    dt: ti.f32,
    win_w: ti.f32,
    win_h: ti.f32,
    infl_r: ti.f32,
    max_spd: ti.f32,
    curv_ref: ti.f32,
    base_damp: ti.f32,
):
    """GPU 并行粒子物理: 环境力 + 双手手势力 + 视觉响应 + 位置积分。

    每个 GPU 线程处理一个粒子, 6000 粒子完全并行。
    """
    for i in range(px.shape[0]):
        # ---- 环境: 有机漂移 ----
        vx[i] += (ti.random(dtype=ti.f32) * 1.2 - 0.6) * dt
        vy[i] += (ti.random(dtype=ti.f32) * 1.2 - 0.6) * dt

        # 弱中心引力
        cx_m = win_w * 0.5
        cy_m = win_h * 0.5
        vx[i] -= (px[i] - cx_m) * 0.00003
        vy[i] -= (py[i] - cy_m) * 0.00003

        influenced = ti.i32(0)

        # ---- 双手手势影响 (叠加) ----
        for h in ti.static(range(2)):
            if hactive_arr[h] == 1:
                _apply_hand_force(
                    i, h,
                    px, py, vx, vy, alpha, radius, ink_level,
                    base_alpha, base_radius,
                    hx_arr, hy_arr, hvx_arr, hvy_arr,
                    hspd_arr, hcurv_arr, hzvel_arr,
                    infl_r, max_spd,
                )
                # 检查该粒子是否在手的影响范围内
                dx = px[i] - hx_arr[h]
                dy = py[i] - hy_arr[h]
                dist = ti.sqrt(dx * dx + dy * dy)
                if dist < infl_r:
                    influenced = 1

        # ---- 不受影响 → 回归基态 ----
        if influenced == 0:
            alpha[i] += (base_alpha[i] - alpha[i]) * 0.03
            radius[i] += (base_radius[i] - radius[i]) * 0.02
            ink_level[i] = base_ink[i]

        # ---- 位置积分 ----
        px[i] += vx[i] * dt * 80.0
        py[i] += vy[i] * dt * 80.0

        # ---- 阻尼 ----
        vx[i] *= base_damp
        vy[i] *= base_damp

        # ---- 屏幕环绕 ----
        if px[i] < -50.0:
            px[i] = win_w + 50.0
        if px[i] > win_w + 50.0:
            px[i] = -50.0
        if py[i] < -50.0:
            py[i] = win_h + 50.0
        if py[i] > win_h + 50.0:
            py[i] = -50.0


@ti.func
def _apply_hand_force(
    i: ti.i32,
    h: ti.i32,
    px: ti.types.ndarray(dtype=ti.f32, ndim=1),
    py: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vx: ti.types.ndarray(dtype=ti.f32, ndim=1),
    vy: ti.types.ndarray(dtype=ti.f32, ndim=1),
    alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ink_level: ti.types.ndarray(dtype=ti.i32, ndim=1),
    base_alpha: ti.types.ndarray(dtype=ti.f32, ndim=1),
    base_radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvx_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hvy_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hspd_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hcurv_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    hzvel_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),
    infl_r: ti.f32,
    max_spd: ti.f32,
):
    """对单个粒子施加一只手的所有力: 中空推力/环壁拉力/粘性/飞溅/漩涡/呼吸 + 视觉响应。"""
    hx = hx_arr[h]
    hy = hy_arr[h]
    dx = px[i] - hx
    dy = py[i] - hy
    dist = ti.sqrt(dx * dx + dy * dy)

    if dist < infl_r:
        # ---- 距离衰减 (smoothstep) ----
        t_val = 1.0 - dist / infl_r
        falloff = t_val * t_val * (3.0 - 2.0 * t_val)

        # 径向单位向量
        rnx = dx / (dist + 0.001)
        rny = dy / (dist + 0.001)

        # 运动特征
        nspd = ti.math.clamp(hspd_arr[h] / max_spd, 0.0, 1.0)
        visc = 1.0 - nspd                     # 慢 = 粘稠, 快 = 稀薄
        hvx = hvx_arr[h]
        hvy = hvy_arr[h]

        # ---- 六大物理力 ----

        # 1. 中空推力 (0-25%): 粒子向外猛推 → 掌心虚空
        if dist < infl_r * 0.25:
            push = falloff * (8.0 + visc * 6.0)
            vx[i] += rnx * push
            vy[i] += rny * push

        # 2. 环壁拉力 (25-40%): 轻吸维持边界 → 漩涡环
        if dist >= infl_r * 0.25 and dist < infl_r * 0.4:
            attract = falloff * (0.8 + visc * 1.5)
            vx[i] -= rnx * attract
            vy[i] -= rny * attract

        # 3. 粘性拖拽: 手慢 → 粒子跟随手流动
        vx[i] += hvx * falloff * visc * 0.08
        vy[i] += hvy * falloff * visc * 0.08

        # 4. 飞溅: 手快 → 粒子向外迸裂
        vx[i] += rnx * falloff * nspd * 2.5
        vy[i] += rny * falloff * nspd * 2.5

        # 5. 漩涡: 基线旋转 + 曲率叠加 → 画弧时更猛
        tx = -rny
        ty = rnx
        vortex_f = visc * 3.0 + hcurv_arr[h] * (3.0 + visc * 10.0)
        vx[i] += tx * falloff * vortex_f
        vy[i] += ty * falloff * vortex_f

        # 6. 呼吸: 纵深位移 → 膨胀/收缩
        breath = hzvel_arr[h] * 0.6
        vx[i] += rnx * falloff * breath
        vy[i] += rny * falloff * breath

        # ---- 视觉响应 ----
        activity = ti.math.clamp(
            nspd * 0.4 + hcurv_arr[h] * 2.0 + ti.abs(hzvel_arr[h]) * 0.003,
            0.0, 1.0,
        )

        # 透明度
        if activity > 0.3:
            alpha[i] += (22.0 * activity * falloff - alpha[i]) * 0.2
        else:
            alpha[i] += (base_alpha[i] - alpha[i]) * 0.2

        # 半径
        if activity > 0.1:
            radius[i] += (base_radius[i] + falloff * 30.0 * activity - radius[i]) * 0.2
        else:
            radius[i] += (base_radius[i] - radius[i]) * 0.2

        # 活跃粒子偏暖色调
        if activity > 0.4:
            ink_level[i] = ti.min(ink_level[i] + 1, 5)

        # 环带粒子额外增亮 → 漩涡环可见
        in_ring = dist >= infl_r * 0.25 and dist < infl_r * 0.4
        if in_ring and activity > 0.15:
            alpha[i] = ti.min(alpha[i] + 3.0, 27.0)
            radius[i] = ti.min(radius[i] + 2.0, 37.0)


# ============================================================
# 运动特征分析器 (双手独立追踪)
# ============================================================

class MotionAnalyzer:
    """从手部关键点序列提取 speed / curvature / z-velocity。

    支持双手独立追踪, 使用 EMA 平滑和 weighted 速度估计。
    """

    MAX_HANDS = 2

    def __init__(self):
        self.states = [self._new_state() for _ in range(self.MAX_HANDS)]

    # ---- 公开接口 ----

    def process(self, hands_data, timestamp):
        """处理一帧的双手数据。

        Args:
            hands_data: MediaPipe 手部数据列表, 或 None。
            timestamp: 当前时间戳 (秒)。

        Returns:
            list[dict]: 每只手的状态字典。
        """
        results = []
        for i in range(self.MAX_HANDS):
            st = self.states[i]
            has = hands_data is not None and i < len(hands_data)

            if has:
                palm = hands_data[i]['palm_center']
                pos = np.array(
                    [palm['x'] * WINDOW_W, palm['y'] * WINDOW_H],
                    dtype=np.float32,
                )
                st['presence_counter'] = min(st['presence_counter'] + 1, 8)
            else:
                pos = np.zeros(2, dtype=np.float32)
                st['presence_counter'] = max(st['presence_counter'] - 1, 0)

            st['hand_detected'] = st['presence_counter'] >= 1
            st['history'].append((pos, timestamp, has))

            if st['hand_detected'] and has:
                self._compute_features(pos, st)
                st['hand_world_pos'] = pos.copy()
            else:
                # 无手时特征衰减
                st['speed'] *= 0.9
                st['curvature'] *= 0.85
                st['z_velocity'] *= 0.85
                st['hand_velocity'] *= 0.9

            results.append(st)
        return results

    # ---- 内部方法 ----

    @staticmethod
    def _new_state():
        """创建一只手的状态字典。"""
        return {
            'history': deque(maxlen=HISTORY_SIZE),
            'speed': 0.0,
            'curvature': 0.0,
            'z_velocity': 0.0,
            'hand_velocity': np.zeros(2, dtype=np.float32),
            'hand_world_pos': None,
            'last_direction': np.zeros(2, dtype=np.float32),
            'hand_detected': False,
            'presence_counter': 0,
        }

    def _compute_features(self, cur_pos, st):
        """从运动历史计算 speed / curvature / z-velocity。"""
        recent = [(p, t) for p, t, h in reversed(st['history']) if h]
        if len(recent) < 2:
            return

        # 加权速度估计 (越近的帧权重越高)
        tv = np.zeros(2, dtype=np.float32)
        tw = 0.0
        for j in range(len(recent) - 1):
            cp, ct = recent[j]
            pp, pt = recent[j + 1]
            dt_val = ct - pt
            if dt_val > 0.001:
                w = 1.0 / (1.0 + j * 0.3)
                tv += ((cp - pp) / dt_val) * w
                tw += w

        if tw > 0:
            rv = tv / tw
            st['speed'] += (float(np.linalg.norm(rv)) - st['speed']) * SMOOTH_ALPHA
            st['hand_velocity'] += (rv - st['hand_velocity']) * SMOOTH_ALPHA

        # 曲率 (方向变化率)
        if st['speed'] > 1.5:
            vn = st['hand_velocity'] / (st['speed'] + 0.0001)
            dot = float(np.clip(np.dot(vn, st['last_direction']), -1, 1))
            angle = np.arccos(dot)
            curv_raw = angle * min(st['speed'] / CURVATURE_REF, 1)
            st['curvature'] += (curv_raw - st['curvature']) * SMOOTH_ALPHA
            st['last_direction'] = vn
        else:
            st['curvature'] *= 0.85

        # 纵深速度 (z 轴位移 / 时间)
        if len(recent) >= 3:
            np_ct = recent[0]
            op_ot = recent[min(len(recent) - 1, 5)]
            zdt = np_ct[1] - op_ot[1]
            if zdt > 0.01:
                raw_zv = (np_ct[0][1] - op_ot[0][1]) / zdt
                st['z_velocity'] += (raw_zv - st['z_velocity']) * SMOOTH_ALPHA


# ============================================================
# 摄像头 + MediaPipe 手部/骨架检测
# ============================================================

class HandTracker:
    """1280×720 摄像头采集 + CLAHE 增强 + MediaPipe Hands/Pose 推理。

    检测策略 (双轨):
    1. MediaPipe Hands (21 点手指) — 优先, 手够大时使用
    2. MediaPipe Pose (33 点全身骨架) — 降级, 手太小/远时用腕关节补位
    """

    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # CLAHE 增强 — 提升远距离手部检出率 (~+20-27%)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # MediaPipe 模型
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )

    def read(self):
        """读取一帧, 返回 (BGR帧, 手部数据, Pose关键点)。

        Returns:
            tuple: (frame, hands_list, pose_landmarks)
                   hands_list 为 None 或手部字典列表。
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None

        frame = cv2.flip(frame, 1)

        # CLAHE 增强 — LAB 色彩空间 L 通道均衡化
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self.clahe.apply(l_ch)
        enhanced = cv2.cvtColor(
            cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR,
        )
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        # 手部检测
        hand_results = self.hands.process(rgb)
        # 全身骨架检测
        pose_results = self.pose.process(rgb)
        pose_lms = pose_results.pose_landmarks

        # 解析手部数据
        hands = []
        if hand_results.multi_hand_landmarks:
            for lm in hand_results.multi_hand_landmarks:
                wrist = lm.landmark[0]
                mid_mcp = lm.landmark[9]
                all_lms = [{'x': p.x, 'y': p.y, 'z': p.z} for p in lm.landmark]
                hands.append({
                    'palm_center': {
                        'x': (wrist.x + mid_mcp.x) / 2,
                        'y': (wrist.y + mid_mcp.y) / 2,
                        'z': (wrist.z + mid_mcp.z) / 2,
                    },
                    'landmarks': all_lms,
                })

        # Pose 腕关节降级 — Hands 检测不到时使用
        if not hands and pose_lms:
            for wrist_id in (15, 16):  # left_wrist, right_wrist
                lm = pose_lms.landmark[wrist_id]
                if lm.visibility > 0.4:
                    hands.append({
                        'palm_center': {'x': lm.x, 'y': lm.y, 'z': lm.z},
                        'landmarks': [],
                    })

        return frame, (hands if hands else None), pose_lms

    def release(self):
        """释放摄像头和 MediaPipe 资源。"""
        self.cap.release()
        self.hands.close()
        self.pose.close()


# ============================================================
# 摄像头水墨渲染器 (右下角小窗)
# ============================================================

class CameraRenderer:
    """水墨滤镜 + 毛笔笔触手部/Pose骨架可视化 → 右下角小窗。"""

    # 手部骨架连接 (MediaPipe Hands 21 点拓扑)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),           # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),      # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),    # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20),    # 小指
        (5, 9), (9, 13), (13, 17),                 # 掌纹横线
    ]
    TIP_IDS = {4, 8, 12, 16, 20}

    # Pose 骨架连接
    POSE_CONNECTIONS = [
        (11, 12), (11, 23), (12, 24), (23, 24),                # 躯干
        (11, 13), (13, 15), (12, 14), (14, 16),                # 手臂
        (15, 17), (17, 19), (19, 21), (15, 21),                # 左手
        (16, 18), (18, 20), (20, 22), (16, 22),                # 右手
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),     # 左腿
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),     # 右腿
    ]
    KEY_JOINTS = {11, 12, 15, 16, 23, 24, 25, 26, 27, 28}

    # BGR 暖金色
    LINE_COLOR = (90, 155, 200)     # WARM_ACCENT
    GLOW_COLOR = (150, 210, 240)    # WARM_LIGHT
    JOINT_COLOR = (80, 140, 190)

    def __init__(self):
        self.w = CAM_W
        self.h = CAM_H
        self._py5_img = None

    def create_py5_image(self):
        """预分配 Py5Image (在 setup 中调用一次)。"""
        arr = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        arr[:, :] = (BG_R, BG_G, BG_B)
        self._py5_img = py5.create_image_from_numpy(arr, 'RGB')

    def process(self, bgr_frame, hands_data, pose_landmarks=None):
        """处理一帧: 滤镜 → 骨架笔触 → 手部笔触 → 暗角 → Py5Image。

        Returns:
            tuple: (py5_img, x, y) 或 (None, 0, 0)
        """
        if bgr_frame is None or self._py5_img is None:
            return None, 0, 0

        # 1. 缩放
        small = cv2.resize(bgr_frame, (self.w, self.h))

        # 2. 水墨滤镜
        filtered = self._ink_wash(small)

        # 3. Pose 骨架笔触
        if pose_landmarks:
            filtered = self._draw_pose_skeleton(filtered, pose_landmarks)

        # 4. 手部毛笔笔触
        if hands_data:
            filtered = self._draw_hand_brush(filtered, hands_data)

        # 5. 暗角
        filtered = self._vignette(filtered)

        # 6. BGR→RGB → Py5Image
        rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
        self._py5_img = py5.create_image_from_numpy(rgb, 'RGB')

        x = py5.width - self.w - CAM_MARGIN
        y = py5.height - self.h - CAM_MARGIN
        return self._py5_img, x, y

    # ---- 内部渲染方法 ----

    def _ink_wash(self, img):
        """轻量水墨滤镜: 双边磨皮 → 暖色调映射 → 墨线叠加。

        人保持清晰可辨, 不完全风格化。
        """
        # 双边滤波 (磨皮保边)
        smooth = cv2.bilateralFilter(
            img, CAM_BILATERAL_D, CAM_BILATERAL_SIGMA, CAM_BILATERAL_SIGMA,
        )

        # 灰度 → 暖色映射 (暗→深褐, 亮→暖米)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        b = np.clip(16 + gray * 120, 0, 255).astype(np.uint8)
        g = np.clip(22 + gray * 155, 0, 255).astype(np.uint8)
        r = np.clip(28 + gray * 180, 0, 255).astype(np.uint8)
        warm = cv2.merge([b, g, r])

        # 混合原色 (保留真实色彩)
        result = cv2.addWeighted(smooth, 1.0 - CAM_WARM_BLEND,
                                 warm, CAM_WARM_BLEND, 0)

        # Canny 边缘 → 墨线叠加
        edges = cv2.Canny(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), 40, 120)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges_blur = cv2.GaussianBlur(
            edges.astype(np.float32), (3, 3), 0,
        ) / 255.0

        ink = np.array([20, 28, 38], dtype=np.float32)  # 深褐墨色 BGR
        ef = edges_blur * CAM_EDGE_STRENGTH
        result = (
            result.astype(np.float32) * (1.0 - ef[:, :, np.newaxis])
            + ink.reshape(1, 1, 3) * ef[:, :, np.newaxis]
        ).astype(np.uint8)

        return result

    def _draw_hand_brush(self, img, hands_data):
        """毛笔笔触风格手部关键点 — 暖金流动曲线 + 指尖光点。"""
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for hand in hands_data:
            lms = hand.get('landmarks', [])
            if len(lms) < 21:
                continue

            pts = self._landmarks_to_pixels(lms, w, h)

            # 连线 — 掌部略粗, 指部略细
            for a, b in self.HAND_CONNECTIONS:
                is_palm = (
                    (a == 0 and b in (5, 9, 13, 17))
                    or (a in (5, 9, 13) and b in (9, 13, 17))
                )
                thick = 3 if is_palm else 2
                cv2.line(overlay, pts[a], pts[b], self.LINE_COLOR,
                         thick, cv2.LINE_AA)

            # 光点 — 指尖大/亮, 关节小
            for j, pt in enumerate(pts):
                if j in self.TIP_IDS:
                    cv2.circle(overlay, pt, CAM_FINGERTIP_R + 2,
                               self.GLOW_COLOR, 1, cv2.LINE_AA)
                    cv2.circle(overlay, pt, CAM_FINGERTIP_R,
                               self.LINE_COLOR, -1, cv2.LINE_AA)
                else:
                    cv2.circle(overlay, pt, CAM_JOINT_R,
                               self.LINE_COLOR, -1, cv2.LINE_AA)

        # 高斯扩散 → 毛笔晕染 + 锐利底层叠加
        overlay_blur = cv2.GaussianBlur(
            overlay, (CAM_BRUSH_BLUR, CAM_BRUSH_BLUR), 0,
        )
        result = cv2.addWeighted(img, 1.0, overlay_blur, CAM_BRUSH_OPACITY, 0)
        result = cv2.addWeighted(result, 1.0, overlay, 0.12, 0)
        return result

    def _draw_pose_skeleton(self, img, pose_lms):
        """全身 Pose 骨架笔触 — 暖金流动线条 + 关节光点 + 腕部加强。"""
        h, w = img.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # 有效关键点
        pts = {}
        for j in range(33):
            lm = pose_lms.landmark[j]
            if lm.visibility > 0.4:
                pts[j] = (max(0, min(w - 1, int(lm.x * w))),
                          max(0, min(h - 1, int(lm.y * h))))

        # 骨架连线
        for a, b in self.POSE_CONNECTIONS:
            if a in pts and b in pts:
                thick = 2 if a in self.KEY_JOINTS and b in self.KEY_JOINTS else 1
                cv2.line(overlay, pts[a], pts[b], self.LINE_COLOR,
                         thick, cv2.LINE_AA)

        # 关节光点
        for j, pt in pts.items():
            if j in self.KEY_JOINTS:
                cv2.circle(overlay, pt, 4, self.GLOW_COLOR, 1, cv2.LINE_AA)
                cv2.circle(overlay, pt, 3, self.JOINT_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(overlay, pt, 2, self.LINE_COLOR, -1, cv2.LINE_AA)

        # 腕关节特别加强 (粒子效果中心)
        for wid in (15, 16):
            if wid in pts:
                pt = pts[wid]
                cv2.circle(overlay, pt, 7, self.GLOW_COLOR, 2, cv2.LINE_AA)
                cv2.circle(overlay, pt, 4, (110, 185, 230), -1, cv2.LINE_AA)

        # 晕染 + 叠加
        overlay_blur = cv2.GaussianBlur(overlay, (3, 3), 0)
        result = cv2.addWeighted(img, 1.0, overlay_blur, 0.35, 0)
        result = cv2.addWeighted(result, 1.0, overlay, 0.15, 0)
        return result

    @staticmethod
    def _landmarks_to_pixels(lms, w, h):
        """将归一化关键点坐标转换为像素坐标。"""
        return [
            (max(0, min(w - 1, int(lm['x'] * w))),
             max(0, min(h - 1, int(lm['y'] * h))))
            for lm in lms
        ]

    @staticmethod
    def _vignette(img):
        """柔和暗角效果。"""
        h, w = img.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        cx_m, cy_m = (w - 1) / 2, (h - 1) / 2
        dist = np.sqrt((xs - cx_m) ** 2 + (ys - cy_m) ** 2)
        max_d = np.sqrt(cx_m ** 2 + cy_m ** 2)
        v = 1.0 - (dist / max_d) ** 1.5 * CAM_VIGNETTE
        v = np.clip(v, 0.0, 1.0)
        return (img.astype(np.float32) * v[:, :, np.newaxis]).astype(np.uint8)


# ============================================================
# 粒子云气 (Taichi GPU 物理 + py5 渲染)
# ============================================================

class CloudParticles:
    """6000 粒子系统。Taichi CUDA kernel 处理物理, py5 OpenGL 渲染。"""

    def __init__(self, count=PARTICLE_COUNT):
        self.c = count
        rng = np.random.default_rng()

        # 位置 (屏幕空间均匀分布)
        self.px = rng.uniform(0, WINDOW_W, count).astype(np.float32)
        self.py = rng.uniform(0, WINDOW_H, count).astype(np.float32)

        # 速度
        self.vx = rng.uniform(-0.5, 0.5, count).astype(np.float32)
        self.vy = rng.uniform(-0.5, 0.5, count).astype(np.float32)

        # 视觉属性
        self.alpha = rng.uniform(2, 12, count).astype(np.float32)
        self.radius = rng.uniform(3, 15, count).astype(np.float32)
        self.ink_level = rng.integers(0, NUM_INK_LEVELS, count).astype(np.int32)

        # 基准状态 (不受手影响时的回归目标)
        self.base_alpha = self.alpha.copy()
        self.base_radius = self.radius.copy()
        self.base_ink = self.ink_level.copy()

        # 手部数据打包缓冲 (GPU 传输用, 2 手 × 7 特征)
        self._hx = np.zeros(2, dtype=np.float32)
        self._hy = np.zeros(2, dtype=np.float32)
        self._hvx = np.zeros(2, dtype=np.float32)
        self._hvy = np.zeros(2, dtype=np.float32)
        self._hspd = np.zeros(2, dtype=np.float32)
        self._hcurv = np.zeros(2, dtype=np.float32)
        self._hzvel = np.zeros(2, dtype=np.float32)
        self._hactive = np.zeros(2, dtype=np.int32)

    def update(self, dt, hands):
        """打包手部数据 → GPU kernel → 原地更新粒子状态。

        Args:
            dt: 帧间隔时间 (秒), 自动 clamp 到 0.1s。
            hands: [(pos_2d, features_dict), ...] — 双手独立施加影响力。
        """
        dt = min(dt, 0.1)
        self._pack_hand_data(hands)

        _particle_physics_kernel(
            self.px, self.py, self.vx, self.vy,
            self.alpha, self.radius, self.ink_level,
            self.base_alpha, self.base_radius, self.base_ink,
            self._hx, self._hy, self._hvx, self._hvy,
            self._hspd, self._hcurv, self._hzvel, self._hactive,
            dt,
            float(WINDOW_W), float(WINDOW_H),
            float(INFLUENCE_RADIUS), float(MAX_SPEED),
            float(CURVATURE_REF), float(BASE_DAMPING),
        )

    def draw(self, has_hand, hand_x, hand_y):
        """逐粒子渲染: 墨韵软圆 + 多层叠加。

        py5 GPU 渲染, 按 alpha 排序实现深度感 (暗粒子在下, 亮粒子在上)。
        """
        py5.no_stroke()
        py5.blend_mode(py5.BLEND)

        # 按透明度排序: 远/暗粒子先画, 近/亮粒子后画
        order = np.argsort(self.alpha)

        for j in range(self.c):
            idx = order[j]
            a = self.alpha[idx]
            if a < 1.0:
                continue

            r = self.radius[idx]
            ink = self.ink_level[idx]
            cr, cg, cb = INK_COLORS[min(ink, NUM_INK_LEVELS - 1)]

            # 手部附近混入暖金色
            if has_hand and hand_x is not None:
                cr, cg, cb = self._blend_warm(
                    cr, cg, cb, self.px[idx], self.py[idx], hand_x, hand_y,
                )

            py5.fill(cr, cg, cb, min(a, 255))
            py5.circle(self.px[idx], self.py[idx], r * 2)

    # ---- 内部方法 ----

    def _pack_hand_data(self, hands):
        """将活跃手部数据打包到固定大小数组 (最多 2 手)。"""
        for i in range(2):
            if i < len(hands) and hands[i][0] is not None:
                hp, feat = hands[i]
                self._hx[i] = hp[0]
                self._hy[i] = hp[1]
                self._hvx[i] = feat['hand_velocity'][0]
                self._hvy[i] = feat['hand_velocity'][1]
                self._hspd[i] = feat['speed']
                self._hcurv[i] = feat['curvature']
                self._hzvel[i] = feat['z_velocity']
                self._hactive[i] = 1
            else:
                self._hactive[i] = 0

    @staticmethod
    def _blend_warm(cr, cg, cb, px, py, hand_x, hand_y):
        """手部附近的粒子混入暖金色。"""
        dx = px - hand_x
        dy = py - hand_y
        d = np.sqrt(dx * dx + dy * dy)
        if d < INFLUENCE_RADIUS * 0.7:
            t = 1.0 - d / (INFLUENCE_RADIUS * 0.7)
            cr = int(cr + (WARM_LIGHT[0] - cr) * t * 0.5)
            cg = int(cg + (WARM_LIGHT[1] - cg) * t * 0.5)
            cb = int(cb + (WARM_LIGHT[2] - cb) * t * 0.5)
        return cr, cg, cb


# ============================================================
# py5 Sketch — 全局状态 & 生命周期
# ============================================================

# 全局单例
_tracker = None
_analyzer = None
_cloud = None
_cam_renderer = None
_last_time = 0.0
_show_debug = False
_fps_buffer = deque(maxlen=30)


def _draw_camera_border(x, y, w, h):
    """摄像头小窗装饰边框 — 水墨画装裱风格。"""
    # 外层阴影
    py5.no_fill()
    py5.stroke_weight(3)
    py5.stroke(*INK_COLORS[4], 35)
    py5.rect(x - 3, y - 3, w + 6, h + 6, 8)
    # 中层墨线
    py5.stroke_weight(2)
    py5.stroke(*INK_COLORS[2], 70)
    py5.rect(x - 1, y - 1, w + 2, h + 2, 6)
    # 内层暖金细线
    py5.stroke_weight(1)
    py5.stroke(*WARM_ACCENT, 45)
    py5.rect(x - 2, y - 2, w + 4, h + 4, 7)
    py5.no_stroke()


def _collect_active_hands(hand_states):
    """从 MotionAnalyzer 状态列表提取活跃手。

    Returns:
        list[tuple]: [(pos_2d, features_dict), ...]
    """
    active = []
    for st in hand_states:
        if st['hand_detected'] and st['hand_world_pos'] is not None:
            feat = {
                'speed': st['speed'],
                'curvature': st['curvature'],
                'z_velocity': st['z_velocity'],
                'hand_velocity': st['hand_velocity'].copy(),
                'hand_detected': True,
            }
            active.append((st['hand_world_pos'], feat))
    return active


def _draw_hand_glows(active_hands):
    """双手光晕 — 柔光 + 内核。"""
    for hand_pos, _ in active_hands:
        hx, hy = hand_pos[0], hand_pos[1]
        # 三层柔光晕
        for r, a in [(35, 6), (22, 12), (12, 22)]:
            py5.fill(*WARM_ACCENT, a)
            py5.circle(hx, hy, r * 2)
        # 内核亮点
        py5.fill(*WARM_LIGHT, 35)
        py5.circle(hx, hy, 10)


def _draw_debug_info(active_hands):
    """调试信息叠加 (D 键切换)。"""
    py5.fill(255, 255, 255, 200)
    py5.text_size(13)
    py5.text_align(py5.LEFT)

    any_hand = len(active_hands) > 0
    status = f"TRACKING x{len(active_hands)}" if any_hand else "waiting..."
    py5.text(f"status: {status}", 15, 25)

    if any_hand:
        primary = active_hands[0][1]
        v = 1.0 - primary['speed'] / MAX_SPEED
        py5.text(
            f"viscosity: {v:.2f}  curvature: {primary['curvature']:.3f}"
            f"  breath: {primary['z_velocity']:+.1f}",
            15, 45,
        )

    py5.text(
        f"particles: {PARTICLE_COUNT}  fps: {py5.frame_rate:.0f}",
        15, 65,
    )

    for i, (hand_pos, _) in enumerate(active_hands):
        py5.text(
            f"hand{i + 1}: ({hand_pos[0]:.0f}, {hand_pos[1]:.0f})",
            15, 85 + i * 20,
        )


def _draw_idle_prompt():
    """无手时的提示文字。"""
    py5.fill(200, 190, 170, 60)
    py5.text_align(py5.CENTER)
    py5.text_size(18)
    py5.text(
        "Raise your hands / 举起双手",
        WINDOW_W / 2, WINDOW_H - 60,
    )


# ---- py5 生命周期回调 ----

def settings():
    """py5 在 setup() 前调用 — 计算自适应窗口大小。"""
    global WINDOW_W, WINDOW_H
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        WINDOW_W = int(sw * 0.85)
        WINDOW_H = int(sh * 0.85)
    except Exception:
        pass  # 回退到默认 1280×720
    py5.size(WINDOW_W, WINDOW_H, py5.P2D)


def setup():
    """py5 初始化 — 创建所有子系统。"""
    global _tracker, _analyzer, _cloud, _cam_renderer, _last_time

    py5.frame_rate(60)
    py5.window_title("演卦 · 星尘太极")

    print("Camera...", end=" ", flush=True)
    _tracker = HandTracker()
    print("OK")

    print("Cloud...", end=" ", flush=True)
    _analyzer = MotionAnalyzer()
    _cloud = CloudParticles(PARTICLE_COUNT)
    _last_time = time.perf_counter()
    print("OK")

    print("Camera renderer...", end=" ", flush=True)
    _cam_renderer = CameraRenderer()
    _cam_renderer.create_py5_image()
    print("OK")

    print("=== 演卦 · 星尘太极 (py5 + Taichi GPU) ===")
    print("  [ESC] quit  [F] fullscreen  [D] debug")


def draw():
    """py5 主循环 — 每帧: 摄像头 → 手势 → 粒子物理(GPU) → 渲染。"""
    global _last_time, _show_debug, _cam_renderer

    # 帧时间
    now = time.perf_counter()
    dt = now - _last_time
    _last_time = now

    # ---- 摄像头 + 手势 + 骨架 ----
    frame, hands, pose_lms = _tracker.read()
    hand_states = _analyzer.process(hands, now)
    active_hands = _collect_active_hands(hand_states)
    any_hand = len(active_hands) > 0

    # ---- 摄像头小窗 (水墨滤镜) ----
    cam_img, cam_x, cam_y = _cam_renderer.process(frame, hands, pose_lms)

    # ---- 拖尾: 半透明覆盖 ----
    py5.no_stroke()
    py5.fill(BG_R, BG_G, BG_B, TRAIL_ALPHA)
    py5.rect(0, 0, WINDOW_W, WINDOW_H)

    # ---- 粒子物理 + 渲染 ----
    _cloud.update(dt, active_hands)
    main_hx = active_hands[0][0][0] if any_hand else None
    main_hy = active_hands[0][0][1] if any_hand else None
    _cloud.draw(any_hand, main_hx, main_hy)

    # ---- 双手光晕 ----
    _draw_hand_glows(active_hands)

    # ---- 摄像头小窗 (右下角) ----
    if cam_img is not None:
        py5.blend_mode(py5.BLEND)
        _draw_camera_border(cam_x, cam_y, CAM_W, CAM_H)
        py5.image(cam_img, cam_x, cam_y)

    # ---- 调试信息 / 提示 ----
    if _show_debug:
        _draw_debug_info(active_hands)

    if not any_hand:
        _draw_idle_prompt()


def key_pressed():
    """键盘事件处理。"""
    global _show_debug
    if py5.key == py5.ESC:
        py5.exit_sketch()
    elif py5.key in ('f', 'F'):
        py5.full_screen(not py5.is_full_screen)
    elif py5.key in ('d', 'D'):
        _show_debug = not _show_debug


def exiting():
    """py5 退出清理。"""
    global _tracker
    if _tracker:
        _tracker.release()
    print("\nYanGua - exited")


# ============================================================
# 入口 — GPU Kernel 预热后启动 py5
# ============================================================
if __name__ == '__main__':
    # Taichi CUDA kernel 必须在主线程编译, py5 渲染在另一线程
    # 此处预热: 用伪数据触发 kernel 编译, 之后 py5 线程可直接调用
    print("GPU kernel warmup...", end=" ", flush=True)
    _N_WARM = 100
    _warm_kwargs = dict(
        px=np.random.uniform(0, 1280, _N_WARM).astype(np.float32),
        py=np.random.uniform(0, 720, _N_WARM).astype(np.float32),
        vx=np.zeros(_N_WARM, dtype=np.float32),
        vy=np.zeros(_N_WARM, dtype=np.float32),
        alpha=np.full(_N_WARM, 5.0, dtype=np.float32),
        radius=np.full(_N_WARM, 10.0, dtype=np.float32),
        ink_level=np.zeros(_N_WARM, dtype=np.int32),
        base_alpha=np.full(_N_WARM, 5.0, dtype=np.float32),
        base_radius=np.full(_N_WARM, 10.0, dtype=np.float32),
        base_ink=np.zeros(_N_WARM, dtype=np.int32),
        hx_arr=np.array([640.0, 0.0], dtype=np.float32),
        hy_arr=np.array([360.0, 0.0], dtype=np.float32),
        hvx_arr=np.zeros(2, dtype=np.float32),
        hvy_arr=np.zeros(2, dtype=np.float32),
        hspd_arr=np.array([50.0, 0.0], dtype=np.float32),
        hcurv_arr=np.zeros(2, dtype=np.float32),
        hzvel_arr=np.zeros(2, dtype=np.float32),
        hactive_arr=np.array([1, 0], dtype=np.int32),
        dt=0.016, win_w=1280.0, win_h=720.0,
        infl_r=240.0, max_spd=800.0, curv_ref=400.0, base_damp=0.985,
    )
    _particle_physics_kernel(**_warm_kwargs)
    print("OK")

    py5.run_sketch()
