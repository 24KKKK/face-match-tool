# -*- coding: utf-8 -*-
"""从参考样本构建目标孩子的"模板向量"。

为什么不是拿一张照片当参考：孩子在几岁内长相变化很大，
单张样本会让模板严重偏向某个特定角度、光线和年龄。
做法是多张样本求均值 + L2 归一化，并剔除明显偏离的脏样本。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .similarity import cosine


def _quality_ok(sample: Dict, cfg: Dict) -> Tuple[bool, str]:
    """质量门禁：把会带偏模板的低质量样本挡在外面。"""
    ref_cfg = cfg.get("reference", {})
    if sample.get("det_score", 0) < float(ref_cfg.get("min_det_score", 0.6)):
        return False, "det_score_low"
    if sample.get("face_px", 0) < float(ref_cfg.get("min_face_px", 80)):
        return False, "face_too_small"
    if abs(sample.get("yaw", 0)) > float(ref_cfg.get("max_abs_yaw", 35)):
        return False, "extreme_yaw"
    if abs(sample.get("pitch", 0)) > float(ref_cfg.get("max_abs_pitch", 35)):
        return False, "extreme_pitch"
    return True, "ok"


def build_template(samples: List[Dict], cfg: Dict) -> Tuple[np.ndarray, Dict, List[Dict]]:
    """输入若干带 embedding 的样本字典，输出 (模板向量, 统计信息, 逐样本明细)。

    明细用于生成 reference_samples.csv，方便你核对"哪些参考样本被丢弃了"。
    """
    if not samples:
        raise ValueError("参考目录中没有提取到任何有效人脸，请检查样本是否清晰正面")

    kept, rejected = [], []
    for item in samples:
        ok, reason = _quality_ok(item, cfg)
        (kept if ok else rejected).append((item, reason))

    if not kept:
        raise ValueError(
            f"参考样本全部被质量过滤掉（共 {len(samples)} 个），"
            f"请放宽 config.yaml 中 reference 的阈值"
        )

    matrix = np.stack([np.asarray(item["embedding"], dtype=np.float32) for item, _ in kept])
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)

    mean_vec = matrix.mean(axis=0)
    mean_vec /= (np.linalg.norm(mean_vec) + 1e-12)

    # 离群样本剔除：可能框到了别人，或是极端模糊帧
    dropped_outlier: List[Dict] = []
    if cfg.get("reference", {}).get("reject_outliers", True) and len(matrix) >= 3:
        sims = np.array([cosine(row, mean_vec) for row in matrix])
        threshold = max(0.30, float(sims.mean() - 1.5 * sims.std()))
        keep_mask = sims >= threshold
        if keep_mask.sum() >= 1 and keep_mask.sum() < len(matrix):
            good = matrix[keep_mask]
            mean_vec = good.mean(axis=0)
            mean_vec /= (np.linalg.norm(mean_vec) + 1e-12)
            for (item, _), survive in zip(kept, keep_mask):
                if not survive:
                    dropped_outlier.append(item)
            kept = [pair for pair, survive in zip(kept, keep_mask) if survive]

    detail: List[Dict] = []
    for item, reason in rejected:
        record = {k: v for k, v in item.items() if k != "embedding"}
        detail.append({**record, "accepted": False, "reject_reason": reason})
    for item in dropped_outlier:
        record = {k: v for k, v in item.items() if k != "embedding"}
        detail.append({**record, "accepted": False, "reject_reason": "outlier"})
    for item, _ in kept:
        record = {k: v for k, v in item.items() if k != "embedding"}
        detail.append({**record, "accepted": True, "reject_reason": ""})

    stats = {
        "total_faces": len(samples),
        "accepted_faces": len(kept),
        "rejected_faces": len(rejected),
        "outlier_dropped": len(dropped_outlier),
        "source_files": len({s.get("source", "") for s, _ in kept}),
    }
    return mean_vec.astype(np.float32), stats, detail
