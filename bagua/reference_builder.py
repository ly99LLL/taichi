"""Build MediaPipe reference templates from the 24-form Tai Chi infographic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from .pose_matcher import DEFAULT_LANDMARK_IDS, normalize_landmarks


REFERENCE_POSES = [
    ("01_qi_shi", "第一式 起势", (80, 80, 560, 760)),
    ("02_zuo_you_ye_ma_fen_zong", "第二式 左右野马分鬃", (760, 80, 1390, 760)),
    ("03_bai_he_liang_chi", "第三式 白鹤亮翅", (1450, 80, 1980, 760)),
    ("04_zuo_you_lou_xi_ao_bu", "第四式 左右搂膝拗步", (2040, 80, 2580, 760)),
    ("05_shou_hui_pi_pa", "第五式 手挥琵琶", (2780, 80, 3260, 760)),
    ("06_zuo_you_dao_juan_gong", "第六式 左右倒卷肱", (3320, 80, 3890, 760)),
    ("07_zuo_lan_que_wei", "第七式 左揽雀尾", (80, 860, 650, 1510)),
    ("08_you_lan_que_wei", "第八式 右揽雀尾", (800, 840, 1500, 1510)),
    ("09_dan_bian", "第九式 单鞭", (1580, 900, 2140, 1510)),
    ("10_yun_shou", "第十式 云手", (2300, 900, 2750, 1510)),
    ("11_dan_bian", "第十一式 单鞭", (2860, 900, 3350, 1510)),
    ("12_gao_tan_ma", "第十二式 高探马", (3400, 900, 3960, 1510)),
    ("13_you_deng_jiao", "第十三式 右蹬脚", (80, 1650, 650, 2350)),
    ("14_shuang_feng_guan_er", "第十四式 双峰贯耳", (800, 1650, 1450, 2350)),
    ("15_zhuan_shen_zuo_deng_jiao", "第十五式 转身左蹬脚", (1650, 1650, 2300, 2350)),
    ("16_zuo_xia_shi_du_li", "第十六式 左下势独立", (2600, 1650, 3270, 2350)),
    ("17_you_xia_shi_du_li", "第十七式 右下势独立", (3280, 1650, 3900, 2350)),
    ("18_zuo_you_chuan_suo", "第十八式 左右穿梭", (80, 2450, 700, 3260)),
    ("19_hai_di_zhen", "第十九式 海底针", (970, 2450, 1520, 3260)),
    ("20_shan_tong_bei", "第二十式 闪通臂", (1800, 2450, 2360, 3260)),
    ("21_zhuan_shen_ban_lan_chui", "第二十一式 转身搬拦捶", (2790, 2450, 3350, 3260)),
    ("22_ru_feng_si_bi", "第二十二式 如封似闭", (1650, 3290, 2220, 4040)),
    ("23_shi_zi_shou", "第二十三式 十字手", (2300, 3290, 2890, 4040)),
    ("24_shou_shi", "第二十四式 收势", (3000, 3290, 3800, 4040)),
]


MATCH_WEIGHTS = {
    "left_shoulder": 0.7,
    "right_shoulder": 0.7,
    "left_elbow": 1.15,
    "right_elbow": 1.15,
    "left_wrist": 1.55,
    "right_wrist": 1.55,
    "left_hip": 0.75,
    "right_hip": 0.75,
    "left_knee": 1.25,
    "right_knee": 1.25,
    "left_ankle": 1.65,
    "right_ankle": 1.65,
    "left_foot": 1.8,
    "right_foot": 1.8,
}


TOLERANCES = {
    "left_shoulder": 0.24,
    "right_shoulder": 0.24,
    "left_elbow": 0.31,
    "right_elbow": 0.31,
    "left_wrist": 0.38,
    "right_wrist": 0.38,
    "left_hip": 0.24,
    "right_hip": 0.24,
    "left_knee": 0.34,
    "right_knee": 0.34,
    "left_ankle": 0.40,
    "right_ankle": 0.40,
    "left_foot": 0.44,
    "right_foot": 0.44,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut the 24-form chart and build pose templates.")
    parser.add_argument("--image", default="", help="Infographic image path. Defaults to the first jimeng-*.png.")
    parser.add_argument("--output-dir", default="reference_poses")
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def find_default_image() -> Path:
    matches = sorted(Path(".").glob("jimeng-*.png"))
    if not matches:
        raise FileNotFoundError(
            "No jimeng-*.png infographic found. Pass --image with a local chart path; "
            "the source chart is intentionally excluded from the public repository."
        )
    return matches[0]


def make_template(name: str, slug: str, bbox: tuple[int, int, int, int], landmarks: list) -> dict:
    normalized = normalize_landmarks(landmarks, DEFAULT_LANDMARK_IDS, visibility_threshold=0.0)
    if not normalized:
        raise RuntimeError(f"Could not normalize landmarks for {name}")

    keypoints = {}
    for joint, point in normalized.items():
        keypoints[joint] = {
            "x": round(point[0], 6),
            "y": round(point[1], 6),
            "z": round(point[2], 6),
            "tolerance": TOLERANCES.get(joint, 0.34),
        }

    return {
        "name": name,
        "slug": slug,
        "description": "MediaPipe template cut from the 24-form Tai Chi infographic.",
        "source": "二十四式图解图",
        "normalization": "body_center_scale",
        "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]},
        "landmark_ids": DEFAULT_LANDMARK_IDS,
        "keypoints": keypoints,
        "match_weights": MATCH_WEIGHTS,
        "effect": {"type": "announce_only"},
    }


def build_templates(image_path: Path, output_dir: Path, save_crops: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.15,
    )
    try:
        built = 0
        for slug, name, bbox in REFERENCE_POSES:
            x1, y1, x2, y2 = bbox
            crop = np.ascontiguousarray(image_np[y1:y2, x1:x2])
            result = pose.process(crop)
            if not result.pose_landmarks:
                raise RuntimeError(f"MediaPipe did not detect a pose for {name}")

            if save_crops:
                Image.fromarray(crop).save(output_dir / f"reference_pose_{slug}.png")

            template = make_template(name, slug, bbox, result.pose_landmarks.landmark)
            (output_dir / f"{slug}.json").write_text(
                json.dumps(template, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            built += 1
            print(f"built {name}: {slug}.json")
    finally:
        pose.close()
    print(f"built {built} templates in {output_dir}")


def main() -> None:
    args = parse_args()
    image_path = Path(args.image) if args.image else find_default_image()
    build_templates(image_path, Path(args.output_dir), save_crops=args.save_crops)


if __name__ == "__main__":
    main()
