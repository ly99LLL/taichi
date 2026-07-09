"""演卦 · 星尘太极 — 全局常量配置.

所有可调参数集中管理, 方便实验和调优。
"""

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
    (180, 160, 130),  # 遥远星芒
    (140, 120, 95),  # 淡星尘
    (105, 85, 65),  # 中层云气
    (75, 58, 42),  # 暗流
    (48, 35, 24),  # 深空暗质
    (28, 18, 12),  # 虚空 (接近背景)
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
CAM_WARM_BLEND = 0.35  # 暖色调混合比例 (越低越保留原色)
CAM_EDGE_STRENGTH = 0.2  # 墨线强度
CAM_VIGNETTE = 0.12  # 暗角强度

# 手部笔触参数
CAM_BRUSH_BLUR = 3  # 笔触扩散模糊
CAM_BRUSH_OPACITY = 0.35  # 笔触透明度
CAM_FINGERTIP_R = 4  # 指尖光点半径
CAM_JOINT_R = 2  # 关节光点半径

# ============================================================
# 系统常量
# ============================================================

# 窗口尺寸 — 默认值, 运行时被屏幕自适应覆盖
WINDOW_W = 1280
WINDOW_H = 720

# 摄像头采集分辨率
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# 粒子系统
PARTICLE_COUNT = 6000
INFLUENCE_RADIUS = 240  # 手掌影响范围 (1.5x 扩大)
MAX_SPEED = 800  # 归一化参考速度
CURVATURE_REF = 400  # 曲率参考速度 (更灵敏)
SMOOTH_ALPHA = 0.35  # EMA 平滑系数 (混沌流动感)
HISTORY_SIZE = 45  # 运动历史长度
BASE_DAMPING = 0.985  # 速度阻尼 (惯性尾迹)

# MediaPipe 模型参数
HANDS_MODEL_COMPLEXITY = 1
HANDS_DETECTION_CONFIDENCE = 0.4
HANDS_TRACKING_CONFIDENCE = 0.4
POSE_MODEL_COMPLEXITY = 1
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.4

# CLAHE 增强参数
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# Pose 腕关节降级可见度阈值
POSE_WRIST_VISIBILITY = 0.4

# py5 帧率目标
TARGET_FRAME_RATE = 60
