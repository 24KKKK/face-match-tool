# -*- coding: utf-8 -*-
"""相似度计算与百分比映射。

为什么要做映射：ArcFace 的余弦相似度真实分布很窄——
同一个人的不同照片大约在 0.3~0.8，不同人大约在 0.0~0.3。
若直接乘 100 当百分比，十个档位里会挤成一坨，且高分区永远为空。
因此默认用 calibrated 模式，把有效的判别区间线性拉伸到 0~100%。
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度。insightface 输出的 embedding 已做 L2 归一化。"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def to_percent(cos: float, cfg: Dict) -> float:
    """把余弦值映射成 0~100 的相似度百分比。"""
    sim_cfg = cfg.get("similarity", {})
    mode = sim_cfg.get("mapping", "calibrated")
    cos = max(-1.0, min(1.0, float(cos)))

    if mode == "raw":
        # 负值直接截断到 0，避免负数百分比出现在文件名里
        return round(max(0.0, cos) * 100.0, 2)

    low = float(sim_cfg.get("calib_low", 0.25))
    high = float(sim_cfg.get("calib_high", 0.75))
    if high <= low:
        raise ValueError("similarity.calib_high 必须大于 calib_low")
    scaled = (cos - low) / (high - low)
    return round(max(0.0, min(1.0, scaled)) * 100.0, 2)


def bin_label(score: float, cfg: Dict) -> str:
    """把相似度百分比归入档位目录名，如 87.3 -> "80-90"。

    100 分会被并入最高档（90-100），避免出现空的 "100-110"。
    """
    width = int(cfg.get("bins", {}).get("width", 10))
    score = max(0.0, min(100.0, float(score)))
    upper = min(100, int(score // width) * width + width)
    lower = upper - width
    if score >= 100.0:
        lower, upper = 100 - width, 100
    return f"{lower:02d}-{upper:02d}"
