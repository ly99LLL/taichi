"""演卦 · 双生涡场的全局配置。

物理参数按职责分组。粒子始终存在于低亮度尘场中，手势只改变它们的
相干性与可见度，不创建或销毁粒子。
"""

# ============================================================
# 视觉系统 — 电影黑场 + 冷银尘埃 + 少量琥珀
# ============================================================

BG_R, BG_G, BG_B = 3, 3, 4
TRAIL_ALPHA = 24

PARTICLE_ALPHA_MIN = 2
PARTICLE_ALPHA_MAX = 12
PARTICLE_SIZE_MIN = 0.8
PARTICLE_SIZE_MAX = 2.8

INK_COLORS = [
    (74, 81, 94),  # 深空蓝灰
    (100, 109, 124),  # 冷石
    (125, 140, 153),  # 月银
    (91, 132, 151),  # 微弱青光
    (137, 122, 157),  # 暗紫
    (181, 153, 106),  # 旧金
    (199, 207, 211),  # 星白
    (101, 132, 119),  # 松烟绿
]
NUM_INK_LEVELS = len(INK_COLORS)

COHERENT_COLOR = (218, 195, 148)
SCATTER_COLOR = (111, 145, 174)
ECHO_COLOR = (93, 101, 115)
UI_PRIMARY = (214, 218, 224)
UI_MUTED = (118, 125, 136)
UI_BORDER = (39, 39, 42)

# 兼容外部调用；新代码使用语义更清楚的颜色名。
WARM_ACCENT = COHERENT_COLOR
WARM_LIGHT = (235, 220, 184)

# ---- 摄像头小窗：原彩影像 + 光学识别层 ----
CAM_W = 280
CAM_H = 210
CAM_MARGIN = 20
CAM_COLOR_CONTRAST = 1.025
CAM_COLOR_SATURATION = 1.02
CAM_VIGNETTE = 0.10
CAM_BRUSH_BLUR = 5
CAM_BRUSH_OPACITY = 0.42
CAM_FINGERTIP_R = 4
CAM_JOINT_R = 2

# ============================================================
# 运行环境
# ============================================================

WINDOW_W = 1280
WINDOW_H = 720
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
TARGET_FRAME_RATE = 60

# ============================================================
# 双生涡场
# ============================================================

PARTICLE_COUNT = 6000
VORTEX_INFLUENCE_RADIUS = 320.0
VORTEX_ORBIT_RADIUS = 92.0
VORTEX_ORBIT_SPEED = 185.0
VORTEX_SLOW_SPEED = 105.0
VORTEX_BREAK_SPEED = 520.0
VORTEX_FORM_SECONDS = 0.55
VORTEX_ECHO_SECONDS = 2.4
VORTEX_PAIR_DISTANCE = 520.0
VORTEX_MAX_DRIFT_SPEED = 180.0

# 兼容旧的导入名。它现在表示涡场外缘，不再表示“手部扰动力”半径。
INFLUENCE_RADIUS = VORTEX_INFLUENCE_RADIUS
MAX_SPEED = VORTEX_BREAK_SPEED

HISTORY_SIZE = 24
SMOOTH_ALPHA = 0.32
CURVATURE_REF = 400.0
REACQUIRE_GAP_SECONDS = 0.20
BASE_DAMPING = 0.973
AMBIENT_DRIFT = 7.0

# ============================================================
# MediaPipe / 图像增强
# ============================================================

HANDS_MODEL_COMPLEXITY = 1
HANDS_DETECTION_CONFIDENCE = 0.4
HANDS_TRACKING_CONFIDENCE = 0.4
POSE_MODEL_COMPLEXITY = 1
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.4
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
POSE_WRIST_VISIBILITY = 0.4
