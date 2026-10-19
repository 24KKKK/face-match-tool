# -*- coding: utf-8 -*-
"""insightface 引擎封装：一次前向同时得到检测框、姿态和特征向量。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


class FaceEngine:
    """薄封装，隔离 CPU / GPU 提供器差异，向外只暴露 analyze()。"""

    def _session_options(self, mcfg: Dict):
        """构造 onnxruntime 会话选项，限制单个进程内部的推理线程数。

        为什么必须限：本机 CPU 只有 12 个逻辑核，onnxruntime 默认会给每个进程
        开满核数的线程。开 6 个进程就是 72 条线程抢 12 个核，相互抢占之下
        实测并行效率掉到 12% 左右 —— 加进程不但没加速，反而比单进程慢。
        0 或负数表示不限（保持 onnxruntime 默认行为）。
        """
        intra = int(mcfg.get("intra_op_num_threads", 0) or 0)
        inter = int(mcfg.get("inter_op_num_threads", 0) or 0)
        if intra <= 0 and inter <= 0:
            return None
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra > 0:
            opts.intra_op_num_threads = intra
        if inter > 0:
            opts.inter_op_num_threads = inter
        return opts

    def __init__(self, cfg: Dict):
        mcfg = cfg.get("model", {})
        from insightface.app import FaceAnalysis

        name = mcfg.get("name", "buffalo_l")
        providers = list(mcfg.get("providers", ["CPUExecutionProvider"]))
        sess_options = self._session_options(mcfg)

        # 绿色版/打包版支持：若项目根目录下自带 models/<name>，优先使用它，
        # 这样发给别人的机器无需联网下载 buffalo_l。非打包环境找不到该目录时
        # root 保持 None，由 insightface 走默认的 ~/.insightface 路径（当前开发机即如此）。
        root = None
        try:
            _proj = Path(__file__).resolve().parent.parent
            if (_proj / "models" / name).is_dir():
                root = str(_proj)
        except Exception:
            root = None

        # 不同 insightface 版本构造签名不一致，做兼容回退
        try:
            kwargs = {"name": name, "providers": providers, "sess_options": sess_options}
            if root:
                kwargs["root"] = root
            self.app = FaceAnalysis(**kwargs)
        except TypeError:
            self.app = FaceAnalysis(name=name, providers=providers)
        except Exception:
            self.app = FaceAnalysis(name=name)

        try:
            self.app.prepare(ctx_id=int(mcfg.get("ctx_id", -1)),
                             det_size=tuple(mcfg.get("det_size", (640, 640))))
        except TypeError:
            self.app.prepare(ctx_id=int(mcfg.get("ctx_id", -1)))

    def analyze(self, bgr: np.ndarray) -> List[Dict]:
        """返回每张脸的结构化信息。多人合照会返回多个元素。"""
        faces = self.app.get(bgr)
        results: List[Dict] = []
        for face in faces:
            emb = getattr(face, "normed_embedding", None)
            if emb is None:
                raw = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(raw)
                emb = raw / (norm + 1e-12) if norm > 0 else raw
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            x1, y1, x2, y2 = [float(v) for v in face.bbox]
            pose = getattr(face, "pose", None)
            if pose is None or len(pose) < 3:
                yaw = pitch = 0.0
            else:
                yaw, pitch = float(pose[0]), float(pose[1])

            results.append({
                "embedding": emb,
                "det_score": float(face.det_score),
                "bbox": [x1, y1, x2, y2],
                "face_px": float(min(x2 - x1, y2 - y1)),
                "yaw": yaw,
                "pitch": pitch,
            })
        return results
