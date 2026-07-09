"""演卦 · 星尘太极 — 全局常量配置.

所有可调参数集中管理, 方便实验和调优。
"""

# ============================================================
# 色彩常量 — 中式云气调色板
# ============================================================

# 背景: 深空暗墨
BG_R, BG_G, BG_B = 16, 12, 8

# 拖尾透明度 (适度提高，避免剧烈运动时残影糊成一片)
TRAIL_ALPHA = 20

# 粒子视觉范围
PARTICLE_ALPHA_MIN = 2
PARTICLE_ALPHA_MAX = 22
PARTICLE_SIZE_MIN = 3
PARTICLE_SIZE_MAX = 32

# 星尘色盘: 冷暖交错，保持水墨气质的同时增加颜色辨识度
INK_COLORS = [
    (225, 178, 105),  # 琥珀金
    (105, 180, 205),  # 月青
    (105, 190, 150),  # 青玉
    (185, 125, 205),  # 淡紫
    (210, 115, 125),  # 胭脂
    (125, 155, 220),  # 星蓝
    (195, 190, 145),  # 月白
    (105, 135, 120),  # 松烟绿
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
CENTER_GRAVITY = 0.00009  # 屏幕中心回聚力 (原值 0.00003)
HAND_FORCE_MULTIPLIER = 1.35  # 六种手掌作用力的统一增强倍率

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
