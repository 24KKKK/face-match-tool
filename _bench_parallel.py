# -*- coding: utf-8 -*-
"""并行扩展效率基准测试。

背景：单进程实测「一张图 177 ms / 一帧视频 157 ms」，
但满负荷跑批时换算出来的单位成本是 1.45 秒，差了约 9 倍。
怀疑是多进程与 onnxruntime 内部线程池互相抢占导致的过度订阅。

本脚本用同一批固定样本，逐个改变进程数，看实际吞吐与加速比，
顺便对比「限制每进程推理线程数」前后的差别。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import load_config
from core.io_utils import iter_media_files
from core import worker


def run_once(cfg, tasks, workers, template):
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=worker.init_worker,
                             initargs=(cfg,)) as pool:
        for res in pool.map(worker.score_file,
                            [{**t, "template": template} for t in tasks]):
            done += 1
    return time.perf_counter() - t0, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=40, help="样本数")
    ap.add_argument("--kind", default="image", choices=["image", "video"])
    ap.add_argument("--cap", type=int, default=10, help="视频每片最多抽帧数")
    ap.add_argument("--threads", type=int, default=0, help="每进程推理线程上限，0=不限制")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if not cfg["paths"]["source_dir"] and os.path.isfile("_webui_config.yaml"):
        cfg = load_config("_webui_config.yaml")
    if a.threads > 0:
        cfg["model"]["intra_op_num_threads"] = a.threads
        cfg["model"]["inter_op_num_threads"] = max(1, a.threads // 2)

    out = cfg["paths"]["output_dir"]
    template = np.load(os.path.join(out, "_template.npy"))
    src = cfg["paths"]["source_dir"]

    items = list(iter_media_files(src, want_image=True, want_video=True,
                                  exclude=["_thumb"]))
    if a.kind == "image":
        picked = [{"path": p, "kind": k} for p, k in items
                  if k in ("image", "heic")][:a.n]
        unit = "张图"
    else:
        if a.cap > 0:
            cfg["scan"]["video"]["max_frames_per_video"] = a.cap
        picked = [{"path": p, "kind": k} for p, k in items if k == "video"][:a.n]
        unit = "个视频"

    print(f"CPU 逻辑核数 : {os.cpu_count()}")
    print(f"样本         : {len(picked)} {unit}"
          f"{f'，每片限 {a.cap} 帧' if a.kind == 'video' and a.cap else ''}")
    print(f"每进程推理线程: {a.threads or '不限制'}")
    print()
    print(f"{'进程数':>6} {'耗时 s':>9} {'吞吐 样本/s':>12} {'加速比':>8} {'相对并行效率':>12}")

    cpu = os.cpu_count() or 8
    base = None
    results = []
    for w in (1, 2, 3, 4, 6, 8):
        if w > (cpu + 2):
            continue
        elapsed, done = run_once(cfg, picked, w, template)
        if base is None:
            base = elapsed
        thr = done / elapsed
        print(f"{w:>6} {elapsed:>9.2f} {thr:>12.2f} {base / elapsed:>8.2f}x"
              f"{base / elapsed / w * 100:>11.0f}%")
        results.append((w, elapsed))
    print()
    print("「相对并行效率」= 加速比 / 进程数。100% 表示完全线性，")
    print("低于 60% 说明进程之间在互相抢资源，继续加进程只会适得其反。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
