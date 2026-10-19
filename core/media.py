# -*- coding: utf-8 -*-
"""视频抽帧。

两个实际问题的处理：
1. Windows 上 cv2.VideoCapture 遇到中文路径会打开失败 —— 回退到临时英文路径副本
2. 跳过中间帧用 grab() 而不是 read()，避免不必要的解码开销
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Dict, Iterator, Tuple

import cv2
import numpy as np

from .io_utils import ext_of


def _open_video(path: str):
    """返回 (VideoCapture 或 None, 临时目录或 None)。"""
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        return cap, None
    cap.release()

    # 回退：复制到临时 ASCII 路径再打开
    tmp_dir = tempfile.mkdtemp(prefix="fmt_video_")
    tmp_path = os.path.join(tmp_dir, "clip" + (ext_of(path) or ".mp4"))
    try:
        shutil.copyfile(path, tmp_path)
        cap = cv2.VideoCapture(tmp_path)
        if cap.isOpened():
            return cap, tmp_dir
    except Exception:
        pass
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
    return None, None


def iter_video_frames(path: str, cfg: Dict, override_ms: int = None) -> Iterator[Tuple[int, np.ndarray, float]]:
    """按配置的毫秒间隔产出 (序号, BGR 帧, 秒位置)。

    override_ms 用于参考样本阶段使用比检索阶段更密的抽帧率。
    """
    vcfg = cfg.get("scan", {}).get("video", {})
    interval_ms = int(override_ms if override_ms else vcfg.get("frame_interval_ms", 2000))
    max_frames = int(vcfg.get("max_frames_per_video", 3000))

    cap, tmp_dir = _open_video(path)
    if cap is None:
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0 or fps > 240:
            fps = 25.0
        step = max(1, int(round(fps * interval_ms / 1000.0)))

        produced = 0
        while produced < max_frames:
            if step > 1:
                skipped_ok = True
                for _ in range(step - 1):
                    if not cap.grab():
                        skipped_ok = False
                        break
                if not skipped_ok:
                    break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            sec = round(produced * step / fps, 2)
            yield produced, frame, sec
            produced += 1
    finally:
        cap.release()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def probe_video_info(path: str) -> Tuple[float, int]:
    """取 (fps, 总帧数)，失败返回 (0, 0)。用于预估耗时。"""
    cap, tmp_dir = _open_video(path)
    if cap is None:
        return 0.0, 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return (fps if fps and fps == fps else 0.0), count
    finally:
        cap.release()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
