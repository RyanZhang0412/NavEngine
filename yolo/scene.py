"""YOLO 场景级三分类：每帧只允许一个类别（互斥）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScenePrediction:
    """一帧图像的唯一场景标签。"""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float] | None
    mask_index: int | None  # result.masks 中的下标


def select_scene_label(result, min_conf: float = 0.0) -> ScenePrediction | None:
    """从 YOLO seg 结果中选出置信度最高的一条，作为整图唯一标签。"""
    if result.boxes is None or len(result.boxes) == 0:
        return None

    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    valid = confs >= min_conf
    if not np.any(valid):
        return None

    idx = int(np.argmax(confs[valid]))
    valid_indices = np.where(valid)[0]
    best_i = int(valid_indices[idx])

    box = result.boxes.xyxy[best_i].cpu().numpy().tolist()
    name = result.names.get(clss[best_i], str(clss[best_i]))
    return ScenePrediction(
        class_id=clss[best_i],
        class_name=name,
        confidence=float(confs[best_i]),
        box_xyxy=tuple(box),
        mask_index=best_i if result.masks is not None and len(result.masks) > best_i else None,
    )
