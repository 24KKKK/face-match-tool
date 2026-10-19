# -*- coding: utf-8 -*-
"""阶段一：扫描打分。

这一阶段对所有源文件只读，不产生任何复制动作，只输出：
  _state.db        SQLite 状态库，支持断点续跑
  scan_report.csv  全量评分报告（Excel 可直接打开）
  reference_samples.csv  参考样本采纳情况，用于核对模板质量
  _template.npy    目标人脸模板向量，便于后续复用与比对

用法示例：
  python scan.py --ref D:\\photos\\ref --src D:\\photos\\all --out D:\\photos\\_out
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

# 被图形界面调用时不能刷 tqdm 进度条（会淹没日志），改用简洁计数输出
_PLAIN_MODE = os.environ.get("FMT_PLAIN") == "1" or _tqdm is None


class _PlainProgress:
    """非交互环境下的轻量进度输出，每行形如 "参考样本  45% (10/22)"。

    只在百分比整数变化时打印一行，这样无论总量是 22 还是几万，
    日志都稳定在一百行以内，不会淹没真正有用的告警信息。
    """

    def __init__(self, iterable=None, desc="", total=0, name_of=None, **kwargs):
        self.iterable = iterable if iterable is not None else []
        self.desc = desc or "进度"
        self.total = int(total or 0)
        self.n = 0
        self._last_pct = -1
        self._name_of = name_of

    def __iter__(self):
        for item in self.iterable:
            self.n += 1
            if self.total:
                pct = min(100, int(self.n * 100 / self.total))
                # 最后一个必须打印，否则会停在 99%
                if pct != self._last_pct or self.n == self.total:
                    self._last_pct = pct
                    # 尾部补当前文件名，方便从日志判断卡在哪一个文件上
                    tail = ""
                    if self._name_of:
                        try:
                            name = self._name_of(item, self.n)
                        except Exception:
                            name = None
                        if name:
                            tail = f"  {name}"
                    print(f"{self.desc} {pct:>3}% ({self.n}/{self.total}){tail}",
                          flush=True)
            yield item

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def progress(iterable=None, **kwargs):
    if _PLAIN_MODE:
        return _PlainProgress(iterable, kwargs.get("desc"), kwargs.get("total"),
                              kwargs.get("name_of"))
    return _tqdm(iterable, **{k: v for k, v in kwargs.items() if k != "name_of"})


def fmt_duration(seconds: float) -> str:
    """把秒数格式化成「1 分 42 秒」这种人类可读形式。"""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} 秒"
    if seconds < 3600:
        return f"{seconds // 60} 分 {seconds % 60:02d} 秒"
    return f"{seconds // 3600} 时 {(seconds % 3600) // 60} 分 {seconds % 60:02d} 秒"

from core.config import load_config, get_output_paths, ALIVE_STATUSES
from core.io_utils import iter_media_files
from core.pairing import build_pair_index, make_reuse_result, unify_pair_scores
from core.store import Store
from core.similarity import bin_label
from core.template import build_template
from core import worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="扫描图片与视频，计算与目标孩子的相似度（只读源文件）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--ref", dest="reference_dir", help="参考样本目录（目标孩子的照片/视频）")
    parser.add_argument("--src", dest="source_dir", help="待检索目录")
    parser.add_argument("--out", dest="output_dir", help="输出根目录")
    parser.add_argument("--workers", type=int, help="并行进程数")
    parser.add_argument("--interval-ms", type=int, help="视频抽帧间隔（毫秒）")
    parser.add_argument("--mapping", choices=["calibrated", "raw"], help="相似度映射口径")
    parser.add_argument("--min-score", type=float, help="最低相似度阈值（仅影响报告标记）")
    parser.add_argument("--det-size", type=int, help="检测输入边长，合照小脸可调大到 768")
    parser.add_argument("--image-only", action="store_true", help="只处理图片")
    parser.add_argument("--video-only", action="store_true", help="只处理视频")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有进度，强制重扫")
    parser.add_argument("--limit", type=int, help="最多处理 N 个文件（小样本试跑用）")
    return parser


def build_overrides(args) -> dict:
    """把命令行参数折算成配置覆盖项。"""
    overrides: dict = {}
    path_override = {}
    if args.reference_dir:
        path_override["reference_dir"] = os.path.abspath(args.reference_dir)
    if args.source_dir:
        path_override["source_dir"] = os.path.abspath(args.source_dir)
    if args.output_dir:
        path_override["output_dir"] = os.path.abspath(args.output_dir)
    if path_override:
        overrides["paths"] = path_override
    if args.workers:
        overrides["runtime"] = {"workers": args.workers}
    if args.interval_ms:
        overrides["scan"] = {"video": {"frame_interval_ms": args.interval_ms}}
    if args.mapping:
        overrides["similarity"] = {"mapping": args.mapping}
    if args.min_score is not None:
        overrides["bins"] = {"min_score": args.min_score}
    if args.det_size:
        overrides["model"] = {"det_size": [args.det_size, args.det_size]}
    return overrides


def write_reference_csv(path: str, detail: list) -> None:
    columns = ["source", "frame", "accepted", "reject_reason", "det_score", "face_px", "yaw", "pitch"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in detail:
            writer.writerow(row)


def write_report_csv(path: str, rows: list, cfg: dict) -> None:
    columns = ["path", "kind", "cos", "score", "bin", "status", "faces", "frames",
               "best_sec", "ts", "ts_source", "size", "message"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            if record.get("score") is not None:
                record["bin"] = bin_label(record["score"], cfg)
            writer.writerow(record)


def print_summary(rows: list, cfg: dict, elapsed: dict | None = None) -> None:
    """打印统计摘要与分档直方图，便于判断阈值是否合适。

    elapsed 形如 {"image": 秒}, {"video": 秒}，用于在摘要里交代各阶段耗时。
    """
    elapsed = elapsed or {}
    # paired 是「沿用封面图分数」的视频，同样有分数值，必须和 scored 一起统计，
    # 否则摘要里的数字会和实际能归档的文件数对不上
    scored = [r for r in rows if r.get("status") in ALIVE_STATUSES
              and r.get("score") is not None]
    reused_rows = [r for r in scored if r.get("status") == "paired"]
    min_score = float(cfg.get("bins", {}).get("min_score", 30))

    status_counter = Counter(r.get("status") or "unknown" for r in rows)
    kind_counter = Counter(r.get("kind") for r in scored)

    print("\n" + "=" * 58)
    print("扫描结果摘要")
    print("=" * 58)
    print(f"  处理文件总数 : {len(rows)}")
    for status, count in status_counter.most_common():
        print(f"    {status:<14}: {count}")

    if elapsed.get("image") or elapsed.get("video"):
        print()
        if elapsed.get("image"):
            print(f"    图片阶段耗时 : {fmt_duration(elapsed['image'])}")
        if elapsed.get("video"):
            print(f"    视频阶段耗时 : {fmt_duration(elapsed['video'])}")

    if scored:
        hit_kind = Counter(r.get("kind") for r in scored if r.get("score") >= min_score)
        print(f"\n  成功评分     : {len(scored)}"
              f"（图片 {kind_counter.get('image', 0) + kind_counter.get('heic', 0)}"
              f" / 视频 {kind_counter.get('video', 0)}"
              f"，其中 {len(reused_rows)} 个视频沿用封面图分数）")
        print(f"  达到阈值     : {sum(1 for r in scored if r['score'] >= min_score)}"
              f"（>= {min_score}%）— 图片 {hit_kind.get('image', 0) + hit_kind.get('heic', 0)}"
              f" / 视频 {hit_kind.get('video', 0)}")

        print("\n  相似度分布:")
        width = int(cfg.get("bins", {}).get("width", 10))
        counts = Counter(bin_label(r["score"], cfg) for r in scored)
        peak = max(counts.values()) if counts else 1
        for low in range(100 - width, -1, -width):
            label = f"{low:02d}-{low + width:02d}"
            count = counts.get(label, 0)
            bar = "#" * int(round(30 * count / peak)) if count else ""
            mark = "  <- 命中区间" if (low + width) > min_score >= low else ""
            print(f"    {label} : {count:>5}  {bar}{mark}")

        # 大量堆在满分区间说明校准区间设置偏窄，需要提示用户调整
        saturated = sum(1 for r in scored if r["score"] >= 99.0)
        if scored and saturated / len(scored) > 0.5:
            print("\n  [调优提示] 超过半数文件的相似度已饱和到 100%，")
            print("              说明 similarity.calib_high 设得偏低。")
            print("              建议把 calib_high 提高到接近当前最高余弦值，")
            print("              或直接查看报告里的 cos 列来了解真实分布。")
    else:
        print("\n  没有任何文件命中，建议检查参考样本是否清晰正面")
    print("=" * 58)


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config, build_overrides(args))
    paths = get_output_paths(cfg)

    ref_dir = cfg["paths"]["reference_dir"]
    src_dir = cfg["paths"]["source_dir"]
    out_root = cfg["paths"]["output_dir"]

    if not ref_dir or not os.path.isdir(ref_dir):
        print(f"[错误] 参考目录不存在：{ref_dir}", file=sys.stderr)
        return 2
    if not src_dir or not os.path.isdir(src_dir):
        print(f"[错误] 待检索目录不存在：{src_dir}", file=sys.stderr)
        return 2
    if not out_root:
        print("[错误] 必须指定输出目录 --out", file=sys.stderr)
        return 2

    os.makedirs(out_root, exist_ok=True)
    workers = max(1, int(cfg.get("runtime", {}).get("workers", 6)))
    want_image = not args.video_only
    want_video = not args.image_only

    print(f"参考目录 : {ref_dir}")
    print(f"检索目录 : {src_dir}")
    print(f"输出目录 : {out_root}")
    print(f"并行进程 : {workers}    映射口径: {cfg['similarity']['mapping']}")

    # 进程数与每进程线程数不匹配时会严重过度订阅，这里给个显式提醒，
    # 免得以后在别的机器上又把进程数往上调而不知道为什么更慢
    os_cpu = os.cpu_count() or 8
    threads = int(cfg.get("model", {}).get("intra_op_num_threads", 0) or os_cpu)
    if workers * threads > os_cpu * 1.5:
        print(f"[提醒] {workers} 进程 × {threads} 线程 = {workers * threads} 条线程，"
              f"远超本机 {os_cpu} 个逻辑核，")
        print(f"       过度订阅会把吞吐拖下来。建议：workers={max(1, os_cpu // threads)} "
              f"或调小 model.intra_op_num_threads")

    # 排除规则：默认跳过 xxx_thumb.jpg 这类缩略图
    exclude = list(cfg.get("scan", {}).get("exclude_patterns", ["_thumb"]))
    if exclude:
        print(f"已跳过   : 文件名含 {' / '.join(exclude)} 的文件")

    # ---------- 第一步：构建目标孩子模板 ----------
    ref_items = list(iter_media_files(ref_dir, want_image=True, want_video=True,
                                      exclude=exclude))
    if not ref_items:
        print("[错误] 参考目录中没有找到支持的图片或视频", file=sys.stderr)
        return 2
    print(f"\n[1/2] 解析参考样本 {len(ref_items)} 个文件...")

    all_samples = []
    failed_refs = []
    ref_names = [Path(p).name for p, _k in ref_items]
    t_ref = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=worker.init_worker,
                             initargs=(cfg,)) as pool:
        tasks = [{"path": p, "kind": k} for p, k in ref_items]
        try:
            for result in progress(pool.map(worker.extract_reference, tasks),
                                   total=len(tasks), desc="参考样本",
                                   name_of=lambda _r, n:
                                   ref_names[min(n, len(ref_names)) - 1]):
                if result["ok"]:
                    all_samples.extend(result["samples"])
                else:
                    failed_refs.append((result["path"], result["message"]))
        except Exception as exc:
            print(f"[错误] 模型加载或推理失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            print("       请确认 insightface 模型已正确下载（~/.insightface/models/buffalo_l）",
                  file=sys.stderr)
            return 3

    if not all_samples:
        print("[错误] 参考样本中没有检出任何人脸，请换几张清晰正面的照片", file=sys.stderr)
        return 3

    try:
        template, stats, detail = build_template(all_samples, cfg)
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 3

    np.save(os.path.join(out_root, "_template.npy"), template)
    write_reference_csv(paths["ref_report"], detail)

    print(f"      检出人脸 {stats['total_faces']} 个，采纳 {stats['accepted_faces']} 个，"
          f"剔除离群 {stats['outlier_dropped']} 个")
    print(f"      参考文件 {stats['source_files']} 个有效；"
          f"解析失败 {len(failed_refs)} 个")
    print(f"      耗时 {fmt_duration(time.perf_counter() - t_ref)}；"
          f"模板明细已写入 {paths['ref_report']}")

    # ---------- 第二步：全量扫描打分 ----------
    src_items = list(iter_media_files(src_dir, want_image, want_video, exclude=exclude))
    if not src_items:
        print("[错误] 待检索目录中没有找到支持的图片或视频，"
              "若文件名普遍含 exclude_patterns 中的关键词，请调整配置项", file=sys.stderr)
        return 2

    store = Store(paths["db"])
    if not args.no_resume:
        done = store.done_paths()
        todo = [(p, k) for p, k in src_items if p not in done]
        skipped = len(src_items) - len(todo)
        if skipped:
            print(f"      断点续跑：跳过已完成 {skipped} 个")
    else:
        todo = src_items
    if args.limit:
        todo = todo[:args.limit]

    print(f"\n[2/2] 扫描 {len(todo)} 个文件（共发现 {len(src_items)} 个）...")

    def run_batch(tasks: List[tuple], desc: str) -> Tuple[List[dict], float]:
        """提交一批任务到进程池，边完成边分批落库。返回 (结果列表, 耗时秒)。"""
        results: List[dict] = []
        started = time.perf_counter()
        if not tasks:
            return results, 0.0
        pending: List[dict] = []
        with ProcessPoolExecutor(max_workers=workers, initializer=worker.init_worker,
                                 initargs=(cfg,)) as pool:
            futures = {
                pool.submit(worker.score_file,
                            {"path": p, "kind": k, "template": template}): p
                for p, k in tasks
            }
            name_of = ((lambda fut, _n: Path(futures.get(fut, "") or "").name)
                       if futures else None)
            try:
                for future in progress(as_completed(futures), total=len(futures),
                                       desc=desc, name_of=name_of):
                    result = future.result()
                    results.append(result)
                    pending.append(result)
                    # 分批落库，保证中途退出也不丢太多进度
                    if len(pending) >= 50:
                        store.upsert_many(pending)
                        pending.clear()
            except KeyboardInterrupt:
                print("\n[中断] 正在保存已完成部分...")
                if pending:
                    store.upsert_many(pending)
            except Exception as exc:
                print(f"\n[错误] 扫描异常：{type(exc).__name__}: {exc}", file=sys.stderr)
        if pending:
            store.upsert_many(pending)
        return results, time.perf_counter() - started

    # ---------- 图片与视频配对：先扫图片，达标则视频免抽帧 ----------
    pair_cfg = cfg.get("scan", {}).get("pairing") or {}
    pairing_on = bool(pair_cfg.get("enabled", True))
    reuse_min = float(pair_cfg.get("reuse_min_score", 70))

    collected: List[dict] = []
    reused = 0
    t_image = 0.0
    t_video = 0.0
    if pairing_on:
        pair_index = build_pair_index(todo)
        image_tasks = [(p, k) for p, k in todo if k in ("image", "heic")]
        video_tasks = [(p, k) for p, k in todo if k == "video"]

        image_results, t_image = run_batch(image_tasks, "检索图片")
        collected += image_results

        # 图片分数达标 → 视频直接沿用，省掉整段抽帧
        by_path = {r["path"]: r for r in collected}
        need_scan: List[tuple] = []
        for vpath, _kind in video_tasks:
            mate = pair_index.get(Path(vpath).stem, {}).get("image")
            mate_row = by_path.get(mate) if mate else None
            score = mate_row.get("score") if mate_row else None
            if mate_row and mate_row.get("status") == "scored" and score is not None \
                    and score >= reuse_min:
                collected.append(make_reuse_result(vpath, mate_row, cfg))
                reused += 1
            else:
                need_scan.append((vpath, "video"))

        # 先报图片阶段耗时，紧接着才是配对复用的结论，读起来才是完整的一段账单
        print(f"      图片阶段耗时 {fmt_duration(t_image)}"
              f"（{len(image_tasks)} 个文件）")
        if reused:
            print(f"      配对复用：{reused} 个视频沿用封面图分数，已跳过抽帧")
        video_results, t_video = run_batch(need_scan, "检索视频")
        collected += video_results
    else:
        collected, t_image = run_batch(todo, "检索评分")

    if reused:
        store.upsert_many([r for r in collected if r.get("status") == "paired"])

    store.set_meta("last_scan", dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    store.set_meta("mapping", cfg["similarity"]["mapping"])

    # 报告需包含历史结果，不能只写本次增量
    rows = store.fetch_all()

    # 配对组统一分数，保证同一条内容的图片与视频归档到同一分档目录
    unified = 0
    if pairing_on and pair_cfg.get("unify_score", True):
        unified = unify_pair_scores(rows, cfg)
        if unified:
            store.upsert_many([r for r in rows if "配对组" in (r.get("message") or "")])
            print(f"      配对归档：{unified} 个文件按组内最高分归入同一档")

    rows.sort(key=lambda r: (r.get("score") is None, -(r.get("score") or 0)))
    write_report_csv(paths["report"], rows, cfg)
    print_summary(rows, cfg, {"image": t_image, "video": t_video})
    print(f"\n完整报告: {paths['report']}")
    print(f"下一步: python archive.py --out \"{out_root}\"")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
