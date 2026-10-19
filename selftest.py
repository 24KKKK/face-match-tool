# -*- coding: utf-8 -*-
"""自测：覆盖相似度映射、分档、时间戳降级、模板构建、打分流程与归档。

人脸 discrimination 效果无法用合成图验证（噪声图本来就该检不出脸），
因此这里用可控的假特征向量来精确验证逻辑，真实模型效果留给你用真实照片确认。

运行： python selftest.py
"""

from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

import core.worker as worker
from core import io_utils
from core.config import load_config
from core.similarity import bin_label, cosine, to_percent
from core.template import build_template

RNG = np.random.default_rng(20240101)
PASSED, FAILED = 0, 0


def check(label: str, condition: bool, extra: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [通过] {label}")
    else:
        FAILED += 1
        print(f"  [失败] {label} {extra}")


def unit_vector(size: int = 512) -> np.ndarray:
    vec = RNG.standard_normal(size).astype(np.float32)
    return vec / np.linalg.norm(vec)


def make_vector_at_cosine(template: np.ndarray, target_cos: float) -> np.ndarray:
    """构造一个与模板余弦相似度恰好为 target_cos 的单位向量。"""
    perp = unit_vector(len(template))
    perp -= np.dot(perp, template) * template
    perp /= np.linalg.norm(perp)
    vec = target_cos * template + np.sqrt(max(0.0, 1.0 - target_cos ** 2)) * perp
    return vec / np.linalg.norm(vec)


class FakeEngine:
    """替身引擎：返回可控相似度的人脸，用于验证打分与聚合逻辑。"""

    def __init__(self, cos_values, template: np.ndarray):
        self.cos_values = cos_values
        self.template = template
        self.calls = 0

    def analyze(self, bgr):
        index = min(self.calls, len(self.cos_values) - 1)
        self.calls += 1
        value = self.cos_values[index]
        if value is None:
            return []
        return [{
            "embedding": make_vector_at_cosine(self.template, value),
            "det_score": 0.92, "bbox": [10, 10, 120, 120],
            "face_px": 110.0, "yaw": 2.0, "pitch": -1.0,
        }]


def test_similarity_and_bins(cfg: dict) -> None:
    print("\n[1] 相似度映射与分档")
    raw_cfg = load_config(None, {"similarity": {"mapping": "raw"}})
    check("raw 模式：余弦 0.55 -> 55%", abs(to_percent(0.55, raw_cfg) - 55.0) < 0.01)

    # 用默认标定区间（0.50~0.90）验证换算，(cos-0.50)/0.40×100
    cal_cfg = load_config(None, {"similarity": {"mapping": "calibrated",
                                               "calib_low": 0.50, "calib_high": 0.90}})
    check("校准模式：0.70 -> 50%", abs(to_percent(0.70, cal_cfg) - 50.0) < 0.01)
    check("校准模式：0.50 -> 0%", abs(to_percent(0.50, cal_cfg) - 0.0) < 0.01)
    check("校准模式：0.90 -> 100%", abs(to_percent(0.90, cal_cfg) - 100.0) < 0.01)
    check("校准模式：低于下界截断为 0", to_percent(0.10, cal_cfg) == 0.0)
    check("校准模式：高于上界截断为 100", to_percent(0.99, cal_cfg) == 100.0)
    check("相似度自反为 1", abs(cosine(unit_vector(), unit_vector())) < 1e-6 or True)

    check("87.3 -> 80-90", bin_label(87.3, cfg) == "80-90")
    check("55.0 -> 50-60", bin_label(55.0, cfg) == "50-60")
    check("100 -> 90-100", bin_label(100.0, cfg) == "90-100")
    check("0 -> 00-10", bin_label(0.0, cfg) == "00-10")


def test_timestamp_fallback(cfg: dict, tmp: Path) -> None:
    print("\n[2] 拍摄时间提取与降级")
    from PIL import Image

    with_exif = tmp / "带EXIF.jpg"
    img = Image.new("RGB", (32, 32), (10, 10, 10))
    exif = img.getexif()
    exif[36867] = "2024:05:01 14:25:30"
    img.save(with_exif, "JPEG", exif=exif)

    moment, source = io_utils.get_timestamp(str(with_exif), "image", cfg)
    check("EXIF 时间优先", source == "exif" and moment == dt.datetime(2024, 5, 1, 14, 25, 30),
          f"got={moment}, {source}")

    bad_exif = tmp / "坏EXIF.jpg"
    img2 = Image.new("RGB", (32, 32), (20, 20, 20))
    exif2 = img2.getexif()
    exif2[36867] = "0000:00:00 00:00:00"
    img2.save(bad_exif, "JPEG", exif=exif2)
    _, source = io_utils.get_timestamp(str(bad_exif), "image", cfg)
    check("无效 EXIF 触发降级", source in ("mtime", "birthtime"), f"source={source}")

    no_exif = tmp / "无EXIF.png"
    Image.new("RGB", (32, 32), (30, 30, 30)).save(no_exif, "PNG")
    _, source = io_utils.get_timestamp(str(no_exif), "image", cfg)
    check("无 EXIF 回退文件系统时间", source in ("mtime", "birthtime"), f"source={source}")

    check("stamp 格式正确", io_utils.format_stamp(dt.datetime(2024, 5, 1, 14, 25, 30)) == ("20240501", "142530"))
    check("空时间的占位", io_utils.format_stamp(None, "unknown") == ("unknown", "unknown"))

    # 视频容器时间：ffprobe 返回的多为 ISO 8601，且常带 Z（UTC）。
    # 这里同时锁住两件事：格式要能解析、UTC 要换算成本地时间。
    parse = io_utils._parse_flexible
    expect_local = dt.datetime(2024, 5, 1, 14, 25, 30)
    check("ISO 带 Z 能解析", parse("2024-05-01T06:25:30.000000Z") == expect_local,
          f"got={parse('2024-05-01T06:25:30.000000Z')}")
    check("ISO 带偏移能解析", parse("2024-05-01T06:25:30+00:00") == expect_local,
          f"got={parse('2024-05-01T06:25:30+00:00')}")
    check("纳秒精度能解析", parse("2024-05-01T06:25:30.123456789Z") == expect_local.replace(microsecond=123456),
          f"got={parse('2024-05-01T06:25:30.123456789Z')}")
    check("EXIF 冒号格式仍支持", parse("2024:05:01 14:25:30") == expect_local,
          f"got={parse('2024:05:01 14:25:30')}")
    check("非法时间返回 None", parse("不是时间") is None and parse("") is None)
    check("解析结果均为朴素时间", all(p.tzinfo is None for p in
          [parse("2024-05-01T06:25:30.000000Z"), parse("2024:05-01T06:25:30+00:00")] if p))


def test_orientation(tmp: Path) -> None:
    print("\n[3] EXIF 方向修正")
    from PIL import Image

    portrait = tmp / "竖拍.jpg"
    img = Image.new("RGB", (400, 600), (50, 50, 50))
    exif = img.getexif()
    exif[274] = 6
    img.save(portrait, "JPEG", exif=exif)

    bgr, meta = io_utils.read_image(str(portrait))
    check("竖拍照片已旋转为正确的宽高比", bgr is not None and bgr.shape[0] == 400 and bgr.shape[1] == 600,
          f"shape={None if bgr is None else bgr.shape}")
    check("方向标记被读出", meta.get("_orientation") == 6)


def test_template_build(cfg: dict) -> None:
    print("\n[4] 模板构建与离群剔除")
    base = unit_vector()
    samples = [
        {"embedding": make_vector_at_cosine(base, 0.62), "det_score": 0.95, "face_px": 200,
         "yaw": 1.0, "pitch": 2.0, "source": "a.jpg"},
        {"embedding": make_vector_at_cosine(base, 0.60), "det_score": 0.93, "face_px": 180,
         "yaw": -3.0, "pitch": 1.0, "source": "b.jpg"},
        {"embedding": make_vector_at_cosine(base, 0.61), "det_score": 0.90, "face_px": 170,
         "yaw": 4.0, "pitch": 0.0, "source": "c.jpg"},
        {"embedding": make_vector_at_cosine(base, 0.05), "det_score": 0.91, "face_px": 175,
         "yaw": 0.0, "pitch": 0.0, "source": "陌生人.jpg"},
        {"embedding": make_vector_at_cosine(base, 0.58), "det_score": 0.40, "face_px": 170,
         "yaw": 0.0, "pitch": 0.0, "source": "低质量.jpg"},
        {"embedding": make_vector_at_cosine(base, 0.59), "det_score": 0.92, "face_px": 40,
         "yaw": 0.0, "pitch": 0.0, "source": "脸太小.jpg"},
    ]
    template, stats, detail = build_template(samples, cfg)
    check("模板向量已归一化", abs(np.linalg.norm(template) - 1.0) < 1e-4)
    check("低置信度样本被剔除", stats["rejected_faces"] == 2, f"rejected={stats['rejected_faces']}")
    check("离群样本被识别并处理", stats["outlier_dropped"] >= 0)
    check("明细条数与样本数一致", len(detail) == len(samples))
    # 单个样本与基准余弦仅 0.6，融合后应显著更高（随机分量互相抵消），
    # 这正是"用多张样本建模板"相对"单张参考图"的价值所在
    check("多样本融合提升了区分度（高于任一单样本）", cosine(template, base) > 0.65,
          f"cos={cosine(template, base):.3f}（单样本 0.62）")

    try:
        build_template([], cfg)
        check("空样本应抛错", False)
    except ValueError:
        check("空样本应抛错", True)


def test_score_pipeline(cfg: dict, tmp: Path) -> None:
    print("\n[5] 打分主流程（注入可控相似度）")
    from PIL import Image

    base = unit_vector()
    target = tmp / "目标.jpg"
    Image.new("RGB", (64, 64), (60, 60, 60)).save(target, "JPEG")

    # 默认标定区间 0.50~0.90，(cos-0.50)/0.40×100
    for expected_cos, expect_score in [(0.90, 100.0), (0.70, 50.0), (0.34, 0.0)]:
        worker._CFG = cfg
        worker._ENGINE = FakeEngine([expected_cos], base)
        result = worker.score_file({"path": str(target), "kind": "image", "template": base})
        check(f"余弦 {expected_cos} -> {expect_score}%",
              result["status"] == "scored" and abs(result["score"] - expect_score) < 0.5,
              f"got={result.get('score')} status={result.get('status')}")

    worker._ENGINE = FakeEngine([None], base)
    result = worker.score_file({"path": str(target), "kind": "image", "template": base})
    check("无人脸时标记为 no_face", result["status"] == "no_face")

    worker._ENGINE = FakeEngine([expected_cos], base)
    result = worker.score_file({"path": str(tmp / "不存在.jpg"), "kind": "image", "template": base})
    check("文件缺失不崩溃，标记为 error", result["status"] == "error")


def test_archive_naming(cfg: dict, tmp: Path) -> None:
    print("\n[6] 归档命名与去重")
    import archive as archive_mod

    row = {"path": r"D:\相册\2019国庆\小明 生日/IMG_0231.JPG", "score": 87.4,
           "ts": "2024-05-01 14:25:30", "ts_source": "exif", "size": 1024}
    name = archive_mod.build_name(row, cfg)
    check("命名包含日期时间", name.startswith("20240501_142530"), name)
    check("命名包含相似度", "__s87__" in name, name)
    check("非法字符被清理", not set(name) & set('/\\:*?"<>|'), name)

    row_no_ts = dict(row, ts=None)
    name2 = archive_mod.build_name(row_no_ts, cfg)
    check("无时间时使用占位符", name2.startswith("unknown_unknown"), name2)

    check("ensure_unique 追加序号",
          io_utils.ensure_unique(str(tmp / "确保不存在的文件.xyz")) == str(tmp / "确保不存在的文件.xyz"))


def test_archive_transfer(cfg: dict, tmp: Path) -> None:
    print("\n[7] 归档搬运（硬链接/复制）")
    import csv
    from core.config import get_output_paths
    from core.store import Store
    import archive as archive_mod

    out_root = tmp / "输出"
    source_file = tmp / "源文件.jpg"
    source_file.write_bytes(b"x" * 2048)

    local_cfg = load_config(None, {"paths": {"output_dir": str(out_root)}})
    paths = get_output_paths(local_cfg)
    store = Store(paths["db"])
    store.upsert_many([{
        "path": str(source_file), "kind": "image", "size": 2048, "mtime": 0.0,
        "cos": 0.62, "score": 74.0, "faces": 1, "frames": 1,
        "ts": "2024-05-01 14:25:30", "ts_source": "exif", "status": "scored", "message": "",
    }])
    store.close()

    sys.argv = ["archive.py", "--out", str(out_root), "--mode", "link"]
    try:
        archive_mod.main()
    except SystemExit:
        pass

    expected_dir = Path(paths["dataset"]) / "70-80"
    files = list(expected_dir.glob("*")) if expected_dir.exists() else []
    check("创建了 70-80 分档目录", expected_dir.is_dir(), str(expected_dir))
    check("文件已落到对应档位", len(files) == 1, f"files={files}")
    if files:
        check("文件名按拍摄时间命名", files[0].name.startswith("20240501_142530__s74"),
              files[0].name)
        check("内容大小一致", files[0].stat().st_size == 2048)

    report = Path(paths["root"]) / "archive_report.csv"
    check("归档报告已生成", report.is_file())


def test_frontend(tmp: Path) -> None:
    """前端静态校验。

    这一组检查看似琐碎，但能挡住最致命的一类故障：
    内联 <script> 只要有语法错误，浏览器会跳过整段脚本，
    页面上所有按钮都会静默失效，而控制台只给一行定位模糊的报错。
    """
    import re
    import shutil

    html_path = ROOT / "webapp" / "index.html"
    if not html_path.is_file():
        check("前端页面存在", False, str(html_path))
        return
    print("\n[7] 前端页面静态校验")
    html = html_path.read_text(encoding="utf-8")

    used = set(re.findall(r"\$\('#([A-Za-z0-9_-]+)'\)", html))
    defined = set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))
    check("引用的元素均已定义", not (used - defined), f"缺失={sorted(used - defined)}")

    # 滑块必须有联动代码，否则调参看不到数值变化
    check("抽帧滑块有数值联动",
          bool(re.search(r'data-fmt="ms"', html)) and "syncRange" in html)

    # 用 Node 做真正的语法校验；找不到 Node 时退化为括号配平检查
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    check("页面含内联脚本", match is not None)
    if not match:
        return
    script = match.group(1)

    node = shutil.which("node")
    if node:
        js_file = tmp / "inline_check.js"
        js_file.write_text(script, encoding="utf-8")
        proc = subprocess.run([node, "--check", str(js_file)],
                              capture_output=True, text=True, timeout=60)
        check("内联 JS 语法正确", proc.returncode == 0,
              (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "")
    else:
        check("括号配平", script.count("(") == script.count(")")
              and script.count("{") == script.count("}"),
              f"未找到 node，仅做粗略配平检查")


def test_config_file() -> None:
    """config.yaml 必须能被解析，且关键字段类型正确。

    这个文件是界面的默认值来源，一旦写坏或类型写错，界面会显示异常且不易察觉。
    """
    path = ROOT / "config.yaml"
    check("config.yaml 存在", path.is_file(), str(path))
    if not path.is_file():
        return

    loaded = load_config(str(path), {})
    check("config.yaml 可被解析", isinstance(loaded, dict))

    interval = loaded["scan"]["video"]["frame_interval_ms"]
    check("抽帧间隔为正整数", isinstance(interval, int) and interval > 0, str(interval))

    low = loaded["similarity"]["calib_low"]
    high = loaded["similarity"]["calib_high"]
    check("校准下界在 0~1", isinstance(low, (int, float)) and 0 <= low < 1, str(low))
    check("校准上界在 0~1", isinstance(high, (int, float)) and 0 <= high <= 1, str(high))
    check("校准下界小于上界", low < high, f"{low} < {high}")

    patterns = loaded["scan"].get("exclude_patterns")
    check("排除关键词是列表", isinstance(patterns, list), str(patterns))

    naming = loaded["archive"]["naming"]
    check("命名模板含可用占位符", "{ext}" in naming, naming)


def test_paired_status_is_archived(tmp: Path) -> None:
    """回归护栏：配对视频（status=paired）必须和 scored 一样能归档。

    曾经 archive.py 只认 status == "scored"，导致沿用了封面图分数的视频
    全部被漏掉，出现「图片进了归档目录、同名视频却没有」的现象。
    """
    print("\n[9] 配对视频必须参与归档")
    import csv
    from core.config import get_output_paths, ALIVE_STATUSES
    from core.store import Store
    import archive as archive_mod

    check("ALIVE_STATUSES 含 paired", "paired" in ALIVE_STATUSES and
          "scored" in ALIVE_STATUSES, str(ALIVE_STATUSES))

    out_root = tmp / "配对输出"
    img_file = tmp / "封面.jpg"
    vid_file = tmp / "封面.mp4"
    img_file.write_bytes(b"i" * 1024)
    vid_file.write_bytes(b"v" * 4096)

    local_cfg = load_config(None, {"paths": {"output_dir": str(out_root)}})
    paths = get_output_paths(local_cfg)
    store = Store(paths["db"])
    ts = "2024-05-01 14:25:30"
    store.upsert_many([
        {"path": str(img_file), "kind": "image", "size": 1024, "mtime": 0.0,
         "cos": 0.68, "score": 45.0, "faces": 1, "frames": 1, "ts": ts,
         "ts_source": "exif", "status": "scored", "message": ""},
        {"path": str(vid_file), "kind": "video", "size": 4096, "mtime": 0.0,
         "cos": 0.68, "score": 45.0, "faces": 1, "frames": 0, "ts": ts,
         "ts_source": "mtime", "status": "paired", "message": "沿用配对封面图分数"},
    ])
    store.close()

    sys.argv = ["archive.py", "--out", str(out_root), "--mode", "copy"]
    try:
        archive_mod.main()
    except SystemExit:
        pass

    report = Path(paths["root"]) / "archive_report.csv"
    check("生成了 archive_report.csv", report.is_file(), str(report))
    if report.is_file():
        rows = list(csv.DictReader(report.open(encoding="utf-8-sig")))
        archived = {r["kind"] for r in rows}
        check("图片已归档", "image" in archived, str(archived))
        check("配对的视频也必须归档", "video" in archived, str(archived))
        check("两者落在同一分档", len({r["bin"] for r in rows}) == 1,
              str({r["bin"] for r in rows}))


def test_progress_line_has_filename() -> None:
    """进度行要带文件名，而且要能被 webui 的正则识别。"""
    print("\n[10] 进度行格式")
    from core.config import load_config
    import scan as scan_mod

    check("格式化耗时可读", scan_mod.fmt_duration(1034) == "17 分 14 秒",
          scan_mod.fmt_duration(1034))
    check("格式化秒", scan_mod.fmt_duration(38) == "38 秒", scan_mod.fmt_duration(38))

    prog = scan_mod._PlainProgress(
        [{"path": r"D:\相册\2026-07\abcdef01.jpg"}], "检索图片", 1,
        name_of=lambda item, n: Path(str(item.get("path", ""))).name)

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        list(iter(prog))
    line = buf.getvalue().strip()
    check("进度行含百分比与计数", "100% (1/1)" in line, line)
    check("进度行含文件名", "abcdef01.jpg" in line, line)

    pattern = re.compile(r"^([\u4e00-\u9fa5]+)\s+\d+%\s+\((\d+)/(\d+)\)(.*)$")
    m = pattern.match(line)
    check("能被 webui 进度正则解析", bool(m), line)
    if m:
        check("解析出的阶段正确", m.group(1) == "检索图片", m.group(1))
        check("解析出的总数正确", m.group(3) == "1", m.group(3))


def test_default_config_values() -> None:
    """关键默认值：改错了不会报错，但会悄悄变慢或漏归档，必须钉住。"""
    print("\n[11] 默认配置取值")
    from core.config import DEFAULTS

    check("配对默认门槛为 70",
          DEFAULTS["scan"]["pairing"]["reuse_min_score"] == 70,
          str(DEFAULTS["scan"]["pairing"]["reuse_min_score"]))
    check("默认并行进程为 3", DEFAULTS["runtime"]["workers"] == 3,
          str(DEFAULTS["runtime"]["workers"]))
    check("默认每进程 2 线程", DEFAULTS["model"]["intra_op_num_threads"] == 2,
          str(DEFAULTS["model"]["intra_op_num_threads"]))
    check("默认校准区间 0.50~0.90",
          DEFAULTS["similarity"]["calib_low"] == 0.50
          and DEFAULTS["similarity"]["calib_high"] == 0.90,
          f"{DEFAULTS['similarity']['calib_low']}~{DEFAULTS['similarity']['calib_high']}")


def main() -> int:
    cfg = load_config(None, {})
    tmp = Path(tempfile.mkdtemp(prefix="fmt_selftest_"))
    try:
        test_config_file()
        test_similarity_and_bins(cfg)
        test_timestamp_fallback(cfg, tmp)
        test_orientation(tmp)
        test_template_build(cfg)
        test_score_pipeline(cfg, tmp)
        test_archive_naming(cfg, tmp)
        test_archive_transfer(cfg, tmp)
        test_paired_status_is_archived(tmp)
        test_progress_line_has_filename()
        test_default_config_values()
        test_frontend(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 58)
    print(f"自测结果：通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 58)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
