# -*- coding: utf-8 -*-
"""文件枚举、安全读写与拍摄时间提取。

重点处理三个 Windows 上的真实坑：
1. cv2.imread 不支持中文路径 —— 改用 np.fromfile + cv2.imdecode
2. OpenCV 不自动应用 EXIF 方向 —— 竖拍照片会以躺姿进入检测器，导致漏检
3. HEIC/HEIF 格式 Pillow 默认不支持 —— 显式识别并跳过，而不是静默失败
"""

from __future__ import annotations

import os
import datetime as dt
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import cv2

try:
    from PIL import Image, ExifTags
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

# HEIC 支持是可选的：装上 pillow_heif 就能处理 iPhone 默认格式
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    _HAS_HEIF = True
except Exception:
    _HAS_HEIF = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg", ".ts", ".webm"}
HEIC_EXTS = {".heic", ".heif"}

# 工具自带的 ffprobe 位置。把 ffprobe.exe 放到 <项目根>/_vendor/ 下即可免配置生效，
# 好处是不污染系统 PATH，也不会被别的软件版本覆盖。
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_vendor")
_VENDOR_FFPROBE = os.path.join(_VENDOR_DIR, "ffprobe.exe")

# 重要：OpenCV 5.x 的 imdecode 默认会自行套用 EXIF 方向，4.x 则不会。
# 这里显式关掉它的自动处理，改由本模块统一控制，
# 否则会出现"双重旋转"——竖拍照片被转 180 度，人脸必然漏检。
_READ_FLAG = cv2.IMREAD_COLOR
if getattr(cv2, "IMREAD_IGNORE_ORIENTATION", None):
    _READ_FLAG = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION

# EXIF 中的取景贾方向标记 -> (cv2 旋转码, 是否水平镜像)
_ORIENTATION_MAP = {
    2: (None, True),
    3: (cv2.ROTATE_180, False),
    4: (None, False),                    # 垂直镜像，罕见，忽略
    5: (cv2.ROTATE_90_COUNTERCLOCKWISE, True),
    6: (cv2.ROTATE_90_CLOCKWISE, False),
    7: (cv2.ROTATE_90_CLOCKWISE, True),
    8: (cv2.ROTATE_90_COUNTERCLOCKWISE, False),
}

_EXIF_DATETIME_KEYS = ("DateTimeOriginal", "DateTimeDigitized", "DateTime")


def ext_of(path: str) -> str:
    return os.path.splitext(path)[1].lower()


def classify(path: str) -> Optional[str]:
    """返回 image / video / heic / None（None 表示不支持的类型）。"""
    ext = ext_of(path)
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in HEIC_EXTS:
        return "heic"
    return None


def iter_media_files(root: str, want_image: bool = True, want_video: bool = True,
                     exclude: Sequence[str] | None = None) -> Iterator[Tuple[str, str]]:
    """递归枚举目录下的媒体文件，返回 (绝对路径, 类型)。

    HEIC 会以 "heic" 类型产出，由调用方决定是否跳过。

    exclude 是一组「文件名子串」，命中即跳过。这是为缩略图准备的：
    不少相册会把 xxx_thumb.jpg 和原图 xxx.jpg 放在一起，缩略图只有 200~400px，
    人脸检出率极低却占用绝大部分扫描预算。
    """
    patterns = [p.lower() for p in (exclude or []) if p]
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            if patterns:
                lowered = name.lower()
                if any(p in lowered for p in patterns):
                    continue
            path = os.path.join(dirpath, name)
            kind = classify(path)
            if kind is None:
                continue
            if kind in ("image", "heic") and not want_image:
                continue
            if kind == "video" and not want_video:
                continue
            yield path, kind


def read_exif_meta(path: str) -> Dict:
    """一次性读出 EXIF 的方向与拍摄时间，避免重复打开文件。"""
    if not _HAS_PIL:
        return {}
    try:
        with Image.open(path) as im:
            top = dict(im.getexif())
            exif_sub = dict(im.getexif().get_ifd(ExifTags.IFD.Exif))
    except Exception:
        return {}

    merged: Dict = {}
    for src in (top, exif_sub):
        for tag_id, value in src.items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            merged[name] = value
    merged["_orientation"] = int(top.get(274, 1) or 1)
    return merged


def _apply_orientation(bgr: np.ndarray, orientation: int) -> np.ndarray:
    rotate_code, need_flip = _ORIENTATION_MAP.get(orientation, (None, False))
    if rotate_code is not None:
        bgr = cv2.rotate(bgr, rotate_code)
    if need_flip:
        bgr = cv2.flip(bgr, 1)
    return bgr


def read_image(path: str, apply_orientation: bool = True) -> Tuple[Optional[np.ndarray], Dict]:
    """读取图片为 BGR ndarray，失败返回 (None, meta)。中文路径安全。"""
    meta = read_exif_meta(path)
    ext = ext_of(path)

    # HEIC 需要 pillow_heif 才能解码，且必须走 Pillow 通道而不是 cv2
    if ext in HEIC_EXTS:
        if not _HAS_HEIF:
            return None, meta
        try:
            with Image.open(path) as im:
                rgb = np.array(im.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            return None, meta
        if apply_orientation:
            try:
                bgr = _apply_orientation(bgr, meta.get("_orientation", 1))
            except Exception:
                pass
        return bgr, meta

    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None, meta
        # 用 _READ_FLAG 读取原始朝向，旋转交给 _apply_orientation 统一处理
        bgr = cv2.imdecode(buf, _READ_FLAG)
    except Exception:
        return None, meta
    if bgr is None:
        return None, meta
    if apply_orientation:
        try:
            bgr = _apply_orientation(bgr, meta.get("_orientation", 1))
        except Exception:
            pass
    return bgr, meta


def _to_local_naive(moment: dt.datetime) -> dt.datetime:
    """把带时区的 datetime 转成本地时区的朴素时间。

    必须统一为朴素本地时间：mtime/birthtime 本身就是本地朴素时间，
    若这里返回带时区的值，后续的大小比较会直接抛 TypeError。
    """
    if moment.tzinfo is None:
        return moment
    return moment.astimezone().replace(tzinfo=None)


def _parse_flexible(value) -> Optional[dt.datetime]:
    """兼容多种日期格式：EXIF 冒号格式、ISO 8601、带 Z 或时区偏移的视频标签。

    注意解析顺序很重要：
      1. 先按原样尝试 ISO 系列（视频的 creation_time 多为 "2024-05-01T06:25:30.000000Z"）
      2. 失败后按 EXIF 的冒号分隔（"2024:05:01 14:25:30"）重排再试
    反过来会踩坑——EXIF 的替换会误伤 ISO 时间部分的冒号。
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "ignore")
        except Exception:
            return None
    if not isinstance(value, str):
        return None

    text = value.strip().strip("\x00")
    if not text:
        return None

    # ① 原生 ISO：Python 3.11+ 的 fromisoformat 能直接吃掉 Z 与 ±HH:MM 偏移
    iso_candidate = text
    if len(iso_candidate) > 26 and "." in iso_candidate[:26]:  # 超过微秒精度会解析失败
        head, _, tail = iso_candidate.partition(".")
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            else:
                break
        iso_candidate = f"{head}.{digits[:6]}{tail[len(digits):]}"
    for candidate in (iso_candidate, iso_candidate.replace("Z", "+00:00")):
        try:
            return _to_local_naive(dt.datetime.fromisoformat(candidate))
        except ValueError:
            pass

    # ② 降级：手动套 strptime，覆盖 EXIF 冒号格式与不带偏移的 ISO
    for candidate in (text, text.replace(":", "-", 2)):
        core = candidate.rstrip("Z")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(core[:26], fmt)
            except ValueError:
                continue
    return None


def _is_plausible(moment: dt.datetime, min_year: int) -> bool:
    if moment is None:
        return False
    now = dt.datetime.now()
    return moment.year >= min_year and moment <= now + dt.timedelta(days=1)


def get_timestamp(path: str, kind: str, cfg: Dict, exif_meta: Optional[Dict] = None) -> Tuple[Optional[dt.datetime], str]:
    """按配置的优先级链提取"拍摄时间"。

    返回一个 (datetime, 来源标记)，全部失败时为 (None, fallback)。
    来源标记会写进 CSV，方便你事后判断命名依据是否可靠。
    """
    ts_cfg = cfg.get("timestamp", {})
    min_year = int(ts_cfg.get("min_valid_year", 1975))
    sources = ts_cfg.get("prefer_sources", {}).get(kind, ["exif", "mtime", "birthtime"])

    exif_meta = exif_meta or {}
    fetched: Dict[str, Optional[dt.datetime]] = {}

    # 调用方没有预先读取元信息时（例如归档重试、外部调用），这里要自己补读一次，
    # 否则 EXIF 拍摄时间会永远取不到
    if "exif" in sources and not exif_meta:
        exif_meta = read_exif_meta(path)

    if "exif" in sources:
        for key in _EXIF_DATETIME_KEYS:
            got = _parse_flexible(exif_meta.get(key))
            if _is_plausible(got, min_year):
                fetched["exif"] = got
                break
    if "meta" in sources:
        fetched["meta"] = _probe_video_creation_time(path, min_year, cfg)

    try:
        st = os.stat(path)
        # Windows 上 st_birthtime 才是真正的"创建时间"
        birth = getattr(st, "st_birthtime", None)
        fetched["birthtime"] = dt.datetime.fromtimestamp(birth) if birth else None
        fetched["mtime"] = dt.datetime.fromtimestamp(st.st_mtime)
    except OSError:
        fetched["birthtime"] = None
        fetched["mtime"] = None

    for source in sources:
        moment = fetched.get(source)
        if _is_plausible(moment, min_year):
            return moment, source
    return None, ts_cfg.get("fallback", "unknown")


def find_ffprobe(cfg: Dict | None = None) -> str | None:
    """定位 ffprobe 可执行文件，按优先级：

      1. 工具自带的 _vendor 目录（推荐，免配置、不依赖系统环境）
         自动向下找两级，兼容官方压缩包解压后的目录结构：
            _vendor/ffprobe.exe
            _vendor/ffmpeg-*/bin/ffprobe.exe
            _vendor/*/ffprobe.exe
      2. 配置文件里 paths.ffprobe_path 指定的路径
      3. 系统 PATH 中的 ffprobe

    三者都没有时返回 None，视频拍摄时间会自动降级到其他来源。
    """
    import glob

    patterns = [
        os.path.join(_VENDOR_DIR, "ffprobe.exe"),
        os.path.join(_VENDOR_DIR, "*", "bin", "ffprobe.exe"),
        os.path.join(_VENDOR_DIR, "*", "ffprobe.exe"),
    ]
    for pattern in patterns:
        hits = glob.glob(pattern)
        if hits:
            return hits[0]

    configured = ((cfg or {}).get("paths") or {}).get("ffprobe_path", "")
    if configured and os.path.isfile(configured):
        return configured

    import shutil
    return shutil.which("ffprobe")


_VIDEO_TIME_TAGS = (
    "creation_time",
    "com.apple.quicktime.creationdate",  # iPhone/MOV 常见
    "date",
)


def _probe_video_creation_time(path: str, min_year: int, cfg: Dict | None = None) -> Optional[dt.datetime]:
    """用 ffprobe 读取视频容器里的拍摄时间。

    同时查 format 级与 stream 级标签：MP4 通常写在 format 上，
    部分 MOV / 转码产物会把 creation_time 挂在视频流上。
    找不到 ffprobe 或没有有效标签时静默返回 None，让上层降级到其他时间来源。
    """
    import json as _json
    import subprocess

    exe = find_ffprobe(cfg)
    if not exe:
        return None

    entries = "format_tags=" + ":format_tags=".join(_VIDEO_TIME_TAGS)
    entries += ":stream_tags=" + ":stream_tags=".join(_VIDEO_TIME_TAGS)
    try:
        proc = subprocess.run(
            [exe, "-v", "quiet", "-print_format", "json", "-show_entries", entries, path],
            capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            return None
        data = _json.loads(proc.stdout.decode("utf-8", "ignore"))
    except Exception:
        return None

    # 先查容器级，再查流级
    candidates: list = [data.get("format", {}).get("tags", {})]
    candidates += [s.get("tags", {}) for s in data.get("streams", [])]

    for tags in candidates:
        for key in _VIDEO_TIME_TAGS:
            got = _parse_flexible(tags.get(key))
            if _is_plausible(got, min_year):
                return got
    return None


def format_stamp(moment: Optional[dt.datetime], fallback: str = "unknown") -> Tuple[str, str]:
    """输出 (20240501, 142530) 形式的两段字符串，供文件名模板使用。"""
    if moment is None:
        return fallback, fallback
    return moment.strftime("%Y%m%d"), moment.strftime("%H%M%S")


def ensure_unique(target: str) -> str:
    """目标已存在时追加 -1、-2 后缀，避免同名覆盖。"""
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(target)
    idx = 1
    while True:
        candidate = f"{stem}-{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1
