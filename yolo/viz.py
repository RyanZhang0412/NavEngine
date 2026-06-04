"""YOLO 分割结果可视化（互斥单类）。"""
from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np

from yolo.scene import ScenePrediction, select_scene_label

CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

CLASS_COLORS_BY_NAME = {
    "线段一端": (255, 120, 0),
    "正在巡线": (0, 255, 0),
    "线段末尾": (0, 120, 255),
}


def draw_text(img: np.ndarray, text: str, org: tuple[int, int], color=(0, 200, 255)) -> None:
    if not any(ord(c) > 127 for c in text):
        cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return
    if not CJK_FONT.is_file():
        cv.putText(img, text.encode("ascii", "replace").decode(), org,
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(str(CJK_FONT), 18)
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv.cvtColor(np.asarray(pil), cv.COLOR_RGB2BGR)


def overlay_scene(frame: np.ndarray, result, scene: ScenePrediction | None, alpha: float = 0.45) -> np.ndarray:
    out = frame.copy()
    if scene is None:
        draw_text(out, "无检测", (10, 30), (128, 128, 128))
        return out

    h, w = frame.shape[:2]
    color = np.array(CLASS_COLORS_BY_NAME.get(scene.class_name, (0, 255, 0)), dtype=np.uint8)

    if scene.mask_index is not None and result.masks is not None:
        mask_tensor = result.masks.data[scene.mask_index]
        mask = mask_tensor.cpu().numpy()
        if mask.shape != (h, w):
            mask = cv.resize(mask, (w, h), interpolation=cv.INTER_NEAREST)
        m = mask > 0.5
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)

    if scene.box_xyxy is not None:
        x1, y1, x2, y2 = map(int, scene.box_xyxy)
        cv.rectangle(out, (x1, y1), (x2, y2), tuple(int(c) for c in color), 2)

    draw_text(out, f"{scene.class_name} {scene.confidence:.2f}", (10, 30), tuple(int(c) for c in color))
    return out


def render_frame(frame: np.ndarray, result, min_conf: float = 0.0, alpha: float = 0.45) -> np.ndarray:
    scene = select_scene_label(result, min_conf=min_conf)
    return overlay_scene(frame, result, scene, alpha=alpha)
