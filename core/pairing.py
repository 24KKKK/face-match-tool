"""图片与视频的配对处理。

微信、QQ 等应用在保存视频时，会同时生成一张同名（仅扩展名不同）的 jpg 作为封面，
内容通常就是视频的第一帧。对这类成对文件：

  1. 图片是视频第一帧，视频按 max 聚合后的分数必然不低于图片分数。
     所以图片分数达标时，视频无需再抽帧——直接沿用即可，结论是保守的（只会低估不会漏）。
  2. 两者属于同一条内容，理应归档到同一个分档目录，
     因此组内分数统一取较高者。

实测收益：目标相册中 172 个视频有 154 个存在同名配图（90%），
图片达标后跳过视频抽帧，可省掉绝大部分视频处理时间。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io_utils import get_timestamp


def build_pair_index(items: List[Tuple[str, str]]) -> Dict[str, Dict[str, Optional[str]]]:
    """把 [(路径, 类型)] 按文件名主干归组。

    返回 {主干: {"image": 图片路径或 None, "video": 视频路径或 None}}。
    只有同时存在图片和视频的组才有配对意义，其余组在后续判断中会被忽略。
    """
    index: Dict[str, Dict[str, Optional[str]]] = {}
    for path, kind in items:
        stem = Path(path).stem
        slot = index.setdefault(stem, {"image": None, "video": None})
        if kind in ("image", "heic"):
            slot["image"] = path
        elif kind == "video":
            slot["video"] = path
    return index


def make_reuse_result(video_path: str, mate: Dict, cfg: Dict) -> Dict:
    """为「沿用配对图片分数」的视频构造结果记录。

    不做抽帧，只补上归档必需的时间戳与体积信息，成本可忽略。
    """
    try:
        stat = os.stat(video_path)
        size, mtime = stat.st_size, stat.st_mtime
    except OSError:
        size, mtime = None, None

    moment, source = get_timestamp(video_path, "video", cfg)
    return {
        "path": video_path,
        "kind": "video",
        "size": size,
        "mtime": mtime,
        "cos": mate.get("cos"),
        "score": mate.get("score"),
        "faces": mate.get("faces", 0),
        "frames": 0,          # 0 表示没有真正抽帧，便于事后区分
        "best_sec": 0.0,
        "ts": moment.isoformat(sep=" ") if moment else None,
        "ts_source": source,
        "status": "paired",
        "message": f"沿用配对封面图分数 {mate.get('score')}%",
    }


def unify_pair_scores(rows: List[Dict], cfg: Dict) -> int:
    """把配对组内分数统一为组内最高分，保证图片与视频落进同一分档目录。

    只处理「同时含图片和视频」的组，避免误伤同名但不同类型扩展名的巧合。
    直接原地修改 rows，返回被调整的行数。
    """
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(Path(row.get("path", "")).stem, []).append(row)

    changed = 0
    for members in groups.values():
        kinds = {m.get("kind") for m in members}
        if not ("image" in kinds or "heic" in kinds) or "video" not in kinds:
            continue

        scored = [m for m in members if m.get("score") is not None]
        if not scored:
            continue

        best_row = max(scored, key=lambda m: m["score"])
        best_score = best_row["score"]
        # 余弦取组内分数最高者对应的值，保证 score 与 cos 始终自洽
        best_cos = best_row.get("cos")

        for member in members:
            if member is best_row:
                continue
            if member.get("score") == best_score:
                continue
            if member.get("score") is None:
                member["message"] = "配对组内无独立评分，沿用组内最高分"
            else:
                member["message"] = (
                    f"配对组统一分数（原 {member.get('score')}% → {best_score}%）"
                )
            member["score"] = best_score
            member["cos"] = best_cos
            changed += 1
    return changed
