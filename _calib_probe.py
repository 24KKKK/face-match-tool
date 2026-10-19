# -*- coding: utf-8 -*-
"""用已保存的模板，反过来给「参考样本目录」打分。

参考样本全部是目标孩子本人，它们的余弦分布就是纯正样本分布，
是标定 similarity.calib_low / calib_high 最可靠的锚点：

  - 正样本的下沿（最低的那几个值）告诉我们「再低就不可能是本人了」
  - 正样本的主体区间告诉我们「百分之百是本人」应该落在哪
"""

from __future__ import annotations

import os
import statistics
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import load_config
from core.io_utils import iter_media_files
from core import worker

CFG_PATH = r"D:\图片分类\归档输出\_webui_config.yaml"


def main():
    cfg = load_config(CFG_PATH)
    out = cfg["paths"]["output_dir"]
    tpl = np.load(os.path.join(out, "_template.npy"))
    items = list(iter_media_files(cfg["paths"]["reference_dir"], True, True, []))
    print(f"参考样本文件 {len(items)}")

    res = []
    with ProcessPoolExecutor(max_workers=2, initializer=worker.init_worker,
                             initargs=(cfg,)) as pool:
        for r in pool.map(worker.score_file,
                          [{"path": p, "kind": k, "template": tpl} for p, k in items]):
            res.append(r)

    cos = sorted(r["cos"] for r in res if r.get("cos") is not None)
    print(f"状态分布 {dict(Counter(r['status'] for r in res))}")
    if not cos:
        print("没有可用样本")
        return 0

    n = len(cos)
    print(f"\n正样本 n={n}")
    print(f"  min {cos[0]:.4f}   p25 {cos[n // 4]:.4f}   中位 "
          f"{statistics.median(cos):.4f}   p75 {cos[3 * n // 4]:.4f}   max {cos[-1]:.4f}")
    print("  全部取值（升序）:")
    print("   ", " ".join(f"{c:.3f}" for c in cos))

    print("\n  低于各候选阈值的正样本个数（这些都是「确定是本人却被卡掉」的误伤）:")
    for th in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        cnt = sum(1 for c in cos if c < th)
        print(f"    余弦 < {th:.2f} : {cnt:>3} 个  ({cnt / n * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
