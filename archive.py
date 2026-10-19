# -*- coding: utf-8 -*-
"""阶段二：归档（真正落地文件的阶段）。

读取 scan.py 产生的状态库，把达到阈值的文件按相似度分档搬运到输出目录，
并按"拍摄时间"重命名。

用法：
  python archive.py --out D:\\photos\\_out
  python archive.py --out D:\\photos\\_out --dry-run   # 只预览，不落盘
  python archive.py --out D:\\photos\\_out --mode copy # 强制真实复制
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from core.config import load_config, get_output_paths, ALIVE_STATUSES
from core.io_utils import format_stamp
from core.similarity import bin_label
from core.store import Store

# Windows 文件名非法字符
_ILLEGAL = re.compile(r'[\\/:*?"<>|\r\n\t]')



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按相似度分档搬运并重命名文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument(
        "--out", dest="output_dir", default=None,
        help="输出根目录；省略时自动取配置文件里的 paths.output_dir",
    )
    parser.add_argument("--min-score", type=float, help="最低相似度阈值，低于此值不搬运")
    parser.add_argument("--mode", choices=["link", "copy"], help="硬链接优先 / 真实复制")
    parser.add_argument("--max-score", type=float, default=100.0, help="相似度上限过滤")
    parser.add_argument("--dry-run", action="store_true", help="只预览不落盘")
    parser.add_argument("--limit", type=int, help="最多搬运 N 个")
    return parser


def build_name(row: dict, cfg: dict) -> str:
    """按模板生成新文件名，并把 Windows 非法字符替换为下划线。"""
    template = cfg.get("archive", {}).get("naming", "{date}_{time}__s{score}__{stem}{ext}")
    fallback = cfg.get("timestamp", {}).get("fallback", "unknown")

    moment = None
    if row.get("ts"):
        try:
            moment = dt.datetime.fromisoformat(row["ts"])
        except ValueError:
            moment = None
    date_part, time_part = format_stamp(moment, fallback)

    source = Path(row["path"])
    score = int(round(float(row["score"])))
    name = template.format(
        date=date_part, time=time_part, score=score,
        stem=source.stem, ext=source.suffix,
    )
    name = _ILLEGAL.sub("_", name).strip().rstrip(".")
    return name or f"{date_part}_{time_part}{source.suffix}"


def place_file(source: str, target: str, mode: str, fallback_copy: bool) -> str:
    """搬运单个文件，返回实际使用的方式：link / copy / failed。"""
    try:
        if mode == "link":
            try:
                os.link(source, target)
                return "link"
            except OSError:
                # 跨分区或文件系统不支持硬链接时回退
                if not fallback_copy:
                    return "failed"
        shutil.copy2(source, target)
        return "copy"
    except Exception:
        return "failed"


def main() -> int:
    args = build_parser().parse_args()
    # 命令行没给 --out 时回落到配置文件里的 paths.output_dir，
    # 这样网页端只传 --config 也能正常工作
    override = {"paths": {}}
    if args.output_dir:
        override["paths"]["output_dir"] = os.path.abspath(args.output_dir)
    if args.min_score is not None:
        override["bins"] = {"min_score": args.min_score}
    if args.mode:
        override["archive"] = {"mode": args.mode}
    cfg = load_config(args.config, override)
    paths = get_output_paths(cfg)

    min_score = float(cfg.get("bins", {}).get("min_score", 30))
    max_score = float(args.max_score)
    mode = cfg.get("archive", {}).get("mode", "link")
    fallback_copy = bool(cfg.get("archive", {}).get("link_fallback_copy", True))
    dataset_root = paths["dataset"]

    db_path = paths["db"]
    if not os.path.isfile(db_path):
        print(f"[错误] 找不到状态库 {db_path}，请先运行 scan.py", file=sys.stderr)
        return 2

    store = Store(db_path)
    rows = store.fetch_all()
    store.close()

    # 候选必须是「有分数」的行，而不仅仅是 status == "scored"。
    # status == "paired" 表示这个视频沿用了同名封面图的分数，它同样有分数值、
    # 同样应该被归档；早期版本只认 scored，导致配对的视频被整体漏掉，
    # 出现「图片进了归档目录、同名视频却没有」的现象。
    candidates = [r for r in rows
                  if r.get("status") in ALIVE_STATUSES
                  and r.get("score") is not None
                  and min_score <= float(r["score"]) <= max_score]
    candidates.sort(key=lambda r: -float(r["score"]))
    if args.limit:
        candidates = candidates[:args.limit]

    if not candidates:
        print(f"[提示] 没有相似度落在 [{min_score}, {max_score}] 区间的文件，无需归档")
        return 0

    print(f"待归档文件 : {len(candidates)}")
    print(f"搬运方式   : {'硬链接（跨盘自动回退复制）' if mode == 'link' else '真实复制'}"
          f"{'  [预览模式]' if args.dry_run else ''}")

    counter = Counter()
    report_rows = []
    for row in candidates:
        label = bin_label(row["score"], cfg)
        target_dir = os.path.join(dataset_root, label)
        new_name = build_name(row, cfg)
        target = os.path.join(target_dir, new_name)

        # 幂等：已存在且大小一致说明上次搬过了
        if os.path.exists(target):
            try:
                if os.path.getsize(target) == (row.get("size") or -1):
                    counter["skipped"] += 1
                    report_rows.append({**row, "bin": label, "new_name": new_name,
                                        "target": target, "mode": "exists"})
                    continue
            except OSError:
                pass
            stem, ext = os.path.splitext(target)
            idx = 1
            while os.path.exists(f"{stem}-{idx}{ext}"):
                idx += 1
            target = f"{stem}-{idx}{ext}"

        used = "preview"
        if not args.dry_run:
            os.makedirs(target_dir, exist_ok=True)
            used = place_file(row["path"], target, mode, fallback_copy)
            if used == "failed":
                counter["failed"] += 1
                report_rows.append({**row, "bin": label, "new_name": new_name,
                                    "target": target, "mode": "failed"})
                continue
        counter[used] += 1
        report_rows.append({**row, "bin": label, "new_name": new_name,
                            "target": target, "mode": used})

    report_path = os.path.join(paths["root"], "archive_report.csv")
    if not args.dry_run:
        columns = ["path", "bin", "new_name", "mode", "score", "cos", "kind",
                   "ts", "ts_source", "target"]
        os.makedirs(paths["root"], exist_ok=True)
        with open(report_path, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report_rows)

    print("\n" + "=" * 58)
    print("归档结果")
    print("=" * 58)
    for key, count in counter.most_common():
        print(f"  {key:<10}: {count}")
    if not args.dry_run:
        print(f"\n输出目录   : {dataset_root}")
        print(f"归档明细   : {report_path}")
        if mode == "link":
            print("注意       : 硬链接与源文件共享数据，删除输出不会影响源文件；")
            print("             但若要拷贝到其他设备，请改用 --mode copy 重新归档")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
