# -*- coding: utf-8 -*-
"""进程池 worker。

Windows 默认用 spawn 启动子进程，因此这里的初始化函数和任务函数
都必须定义在模块顶层，且参数能被 pickle（config dict + numpy 数组即可）。
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .face import FaceEngine
from .media import iter_video_frames
from .io_utils import read_image, get_timestamp, _HAS_HEIF
from .similarity import cosine, to_percent

_ENGINE: FaceEngine = None
_CFG: Dict = None


def init_worker(cfg: Dict) -> None:
    """每个子进程各自持有一份模型实例，避免跨进程序列化模型。"""
    global _ENGINE, _CFG
    _CFG = cfg
    _ENGINE = FaceEngine(cfg)


def _blank_result(path: str, kind: str) -> Dict:
    return {
        "path": path, "kind": kind, "size": None, "mtime": None,
        "cos": None, "score": None, "faces": 0, "frames": 0,
        "best_sec": None, "ts": None, "ts_source": "unknown",
        "status": "ok", "message": "",
    }


def extract_reference(task: Dict) -> Dict:
    """参考样本阶段：抽出该文件中所有可用人脸。"""
    path, kind = task["path"], task["kind"]
    cfg = _CFG
    interval = int(cfg.get("reference", {}).get("video_frame_interval_ms", 1000))

    try:
        samples: List[Dict] = []
        if kind in ("image", "heic"):
            img, _ = read_image(path)
            if img is None:
                reason = "HEIC 缺少解码器" if (kind == "heic" and not _HAS_HEIF) else "无法解码"
                return {"path": path, "ok": False, "message": reason, "samples": []}
            for face in _ENGINE.analyze(img):
                face["source"] = path
                face["frame"] = 0
                samples.append(face)
        else:
            for idx, frame, sec in iter_video_frames(path, cfg, override_ms=interval):
                for face in _ENGINE.analyze(frame):
                    face["source"] = path
                    face["frame"] = idx
                    face["sec"] = sec
                    samples.append(face)
        return {"path": path, "ok": True, "message": "", "samples": samples}
    except Exception as exc:
        return {"path": path, "ok": False, "message": f"{type(exc).__name__}: {exc}", "samples": []}


def score_file(task: Dict) -> Dict:
    """检索阶段：计算单个文件与目标模板的最高相似度。"""
    path, kind = task["path"], task["kind"]
    template = task["template"]
    cfg = _CFG
    res = _blank_result(path, kind)

    try:
        stat = os.stat(path)
        res["size"], res["mtime"] = stat.st_size, stat.st_mtime

        if kind in ("image", "heic"):
            img, meta = read_image(path)
            if img is None:
                res["status"] = "skipped"
                res["message"] = ("HEIC 格式缺少解码器，请 pip install pillow-heif"
                                  if kind == "heic" and not _HAS_HEIF else "无法解码图片")
                return res

            moment, source = get_timestamp(path, "image", cfg, meta)
            res["ts"] = moment.isoformat(sep=" ") if moment else None
            res["ts_source"] = source

            faces = _ENGINE.analyze(img)
            res["frames"], res["faces"] = 1, len(faces)
            if not faces:
                res["status"] = "no_face"
                return res

            best = max(cosine(f["embedding"], template) for f in faces)
            res["cos"] = round(float(best), 4)
            res["score"] = to_percent(best, cfg)
            res["status"] = "scored"
            return res

        # 视频：按抽帧逐帧比对后聚合
        moment, source = get_timestamp(path, "video", cfg)
        res["ts"] = moment.isoformat(sep=" ") if moment else None
        res["ts_source"] = source

        vcfg = cfg.get("scan", {}).get("video", {})
        hits = []
        for idx, frame, sec in iter_video_frames(path, cfg):
            res["frames"] += 1
            faces = _ENGINE.analyze(frame)
            res["faces"] += len(faces)
            if faces:
                best = max(cosine(f["embedding"], template) for f in faces)
                hits.append((float(best), sec, idx))

        if not hits:
            res["status"] = "no_frame" if res["frames"] == 0 else "no_face"
            return res
        if len(hits) < int(vcfg.get("min_hit_frames", 1)):
            res["status"] = "no_face"
            return res

        if vcfg.get("aggregate", "max") == "top3_mean":
            top = sorted(hits, key=lambda item: -item[0])[:3]
            best_score = float(np.mean([item[0] for item in top]))
            best_idx = top[0][2]
        else:
            top = max(hits, key=lambda item: item[0])
            best_score, best_idx = top[0], top[2]

        seconds_elapsed = best_idx * max(1, int(vcfg.get("frame_interval_ms", 2000))) / 1000.0
        res["cos"] = round(best_score, 4)
        res["score"] = to_percent(best_score, cfg)
        res["best_sec"] = round(seconds_elapsed, 2)
        res["status"] = "scored"
        return res

    except Exception as exc:
        res["status"] = "error"
        res["message"] = f"{type(exc).__name__}: {exc}"
        return res
