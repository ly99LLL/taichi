"""运动特征分析器 — 从手部关键点序列提取 speed / curvature / z-velocity.

支持双手独立追踪, 使用 EMA 平滑和加权速度估计。
"""

from collections import deque

import numpy as np

from yan_gua.config import CURVATURE_REF, HISTORY_SIZE, SMOOTH_ALPHA


class MotionAnalyzer:
    """从手部关键点序列提取运动特征。

    每只手独立追踪: speed (手速), curvature (曲率/方向变化),
    z_velocity (纵深速度), hand_velocity (2D 速度向量)。
    """

    MAX_HANDS = 2

    def __init__(self, win_w, win_h):
        """初始化运动分析器。

        Args:
            win_w: 窗口宽度 (用于归一化坐标→像素转换)。
            win_h: 窗口高度。
        """
        self.win_w = win_w
        self.win_h = win_h
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
                    [palm['x'] * self.win_w, palm['y'] * self.win_h],
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
