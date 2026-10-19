# -*- coding: utf-8 -*-
"""视频推理速度受控基准测试。

目的：回答「改完之后视频推理是不是变慢了」。

方法：在同一个进程、同一份模型、同一批源文件上，分别测量三类单位成本：
  A. 单张图片：读图 + 时间戳 + 人脸检测 + 特征提取
  B. 单帧视频：解码（grab/read）与推理分开计时
  C. 视频时间戳提取里的 ffprobe 子进程开销（这是视频独有的额外成本）

同一台机器同一时刻测得，三者可直接横向对比。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import load_config
from core.io_utils import read_image, get_timestamp, iter_media_files
from core.media import iter_video_frames
from core.face import FaceEngine


def fmt(sec: float) -> str:
    return f"{sec * 1000:.0f} ms"


def bench_images(engine, template, cfg, paths, extra=None):
    """测图片：整体一次过，再单独抠出推理部分。"""
    from core.similarity import cosine

    rows = []
    for path in paths:
        t_all = time.perf_counter()
        img, meta = read_image(path)
        if img is None:
            continue
        get_timestamp(path, "image", cfg, meta)
        t_inf = time.perf_counter()
        faces = engine.analyze(img)
        if faces:
            max(cosine(f["embedding"], template) for f in faces)
        t_end = time.perf_counter()
        rows.append((t_end - t_all, t_end - t_inf, len(faces)))
    if not rows:
        return None
    total = sum(r[0] for r in rows) / len(rows)
    infer = sum(r[1] for r in rows) / len(rows)
    return {
        "n": len(rows),
        "total_ms": total * 1000,
        "infer_ms": infer * 1000,
        "overhead_ms": (total - infer) * 1000,
    }


def bench_videos(engine, template, cfg, paths, max_frames=25):
    """测视频帧：把「抽解码」与「推理」分别计时。"""
    from core.similarity import cosine

    vcfg = dict(cfg.get("scan", {}).get("video", {}))
    vcfg["max_frames_per_video"] = max_frames
    cfg2 = {**cfg, "scan": {**cfg["scan"], "video": vcfg}}

    dec_total = 0.0
    inf_total = 0.0
    n_frames = 0
    n_videos = 0
    per_video = []

    for path in paths:
        frames = list(iter_video_frames(path, cfg2))
        if not frames:
            continue
        n_videos += 1
        per_video.append((Path(path).name, len(frames)))
        for _idx, frame, _sec in frames:
            n_frames += 1
            t0 = time.perf_counter()
            faces = engine.analyze(frame)
            t1 = time.perf_counter()
            if faces:
                max(cosine(f["embedding"], template) for f in faces)
            t2 = time.perf_counter()
            inf_total += (t1 - t0)
            dec_total += 0.0  # 解码已在生成阶段完成，单独在下面重测

    # 单独测纯解码成本：同一批视频再跑一次 grab/read 循环（不推理）
    dec_loop = 0.0
    dec_frames = 0
    for path in paths:
        t0 = time.perf_counter()
        for _ in iter_video_frames(path, cfg2):
            dec_frames += 1
        dec_loop += time.perf_counter() - t0

    if not n_frames:
        return None
    return {
        "n_videos": n_videos,
        "n_frames": n_frames,
        "infer_ms": inf_total / n_frames * 1000,
        "decode_ms": dec_loop / max(1, dec_frames) * 1000,
        "per_video": per_video,
    }


def bench_ffprobe(cfg, paths, n=12):
    """测视频时间戳提取耗时（内含 ffprobe 子进程）。"""
    from core.io_utils import find_ffprobe

    exe = find_ffprobe(cfg)
    sample = paths[:n]
    t0 = time.perf_counter()
    kinds = []
    for p in sample:
        _m, src = get_timestamp(p, "video", cfg)
        kinds.append(src)
    el = time.perf_counter() - t0
    return {"exe": exe, "n": len(sample), "ms_each": el / max(1, len(sample)) * 1000,
            "sources": kinds}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="配置文件路径；默认依次尝试 config.yaml 与 _webui_config.yaml")
    ap.add_argument("--src", default=None, help="待检索目录")
    ap.add_argument("--out", default=None, help="输出目录（内含 _template.npy）")
    ap.add_argument("--n-img", type=int, default=8)
    ap.add_argument("--n-vid", type=int, default=6)
    ap.add_argument("--cap", type=int, default=25, help="每个视频最多测多少帧")
    a = ap.parse_args()

    cfg = load_config(a.config)
    # 仓库里的 config.yaml 路径字段是空的，实际跑批用的是 webui 生成的配置，
    # 这里补一层兜底，保证脚本独立可用。
    if not cfg["paths"]["source_dir"] and os.path.isfile("_webui_config.yaml"):
        cfg = load_config("_webui_config.yaml")
    if a.src:
        cfg["paths"]["source_dir"] = a.src
    if a.out:
        cfg["paths"]["output_dir"] = a.out

    src = cfg["paths"]["source_dir"]
    out = Path(cfg["paths"]["output_dir"])
    if not src or not os.path.isdir(src):
        print(f"[错误] 待检索目录无效：{src}", file=sys.stderr)
        return 2
    try:
        tpl = np.load(out / "_template.npy")
    except OSError as exc:
        print(f"[错误] 读不到模板 {out / '_template.npy'}：{exc}", file=sys.stderr)
        print("       请先跑完一次 scan.py", file=sys.stderr)
        return 2

    n_img, n_vid, cap = a.n_img, a.n_vid, a.cap

    items = list(iter_media_files(src, want_image=True, want_video=True,
                                  exclude=["_thumb"]))
    imgs = [p for p, k in items if k in ("image", "heic")][:n_img]
    vids = [p for p, k in items if k == "video"][:n_vid]

    print("=" * 60)
    print("视频推理速度基准测试（单进程，无任何并行）")
    print("=" * 60)
    print(f"图片样本 {len(imgs)} 张，视频样本 {len(vids)} 个（每视频上限 {cap} 帧）")

    print("\n加载模型 ...", flush=True)
    engine = FaceEngine(cfg)
    # 预热：第一次推理含懒加载开销，必须排除在外
    warm = [p for p, k in items if k in ("image", "heic")][-1:]
    if warm:
        img, _m = read_image(warm[0])
        if img is not None:
            engine.analyze(img)
    print("模型就绪\n", flush=True)

    r_img = bench_images(engine, tpl, cfg, imgs)
    print("[A] 单张图片")
    if r_img:
        print(f"    样本数       : {r_img['n']}")
        print(f"    端到端平均   : {fmt(r_img['total_ms'] / 1000)}")
        print(f"    其中推理     : {fmt(r_img['infer_ms'] / 1000)}")
        print(f"    其余(读图+时间): {fmt(r_img['overhead_ms'] / 1000)}")
    else:
        print("    无可用样本")

    r_vid = bench_videos(engine, tpl, cfg, vids, cap)
    print("\n[B] 视频单帧")
    if r_vid:
        print(f"    视频数/帧数  : {r_vid['n_videos']} / {r_vid['n_frames']}")
        print(f"    单帧推理平均 : {fmt(r_vid['infer_ms'] / 1000)}")
        print(f"    单帧解码平均 : {fmt(r_vid['decode_ms'] / 1000)}")
        print(f"    单帧合计     : {fmt((r_vid['infer_ms'] + r_vid['decode_ms']) / 1000)}")
        print("    明细:")
        for name, nf in r_vid["per_video"]:
            print(f"      {nf:>4} 帧  {name}")
    else:
        print("    无可用帧")

    r_ff = bench_ffprobe(cfg, vids)
    print("\n[C] 视频时间戳提取（含 ffprobe 子进程）")
    print(f"    ffprobe 路径 : {r_ff['exe']}")
    print(f"    样本数       : {r_ff['n']}")
    print(f"    平均每个视频 : {fmt(r_ff['ms_each'] / 1000)}")
    print(f"    实际来源分布 : {r_ff['sources']}")

    print("\n" + "=" * 60)
    print("结论要用到的对比：")
    if r_img and r_vid:
        ratio = r_vid["infer_ms"] / r_img["infer_ms"]
        print(f"  单帧推理 / 单图推理 = {ratio:.3f}")
        print(f"  （两者跑的是同一个 detect+embed 流程，"
              f"{'基本一致说明视频侧没有额外损耗' if abs(ratio - 1) < 0.15 else '偏离较大，需要进一步查'}）")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())
