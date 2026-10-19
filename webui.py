# -*- coding: utf-8 -*-
"""face-match-tool 图形化界面（后端）。

设计要点：这里不重复实现识别逻辑，而是以子进程方式调用已验证的 scan.py / archive.py。
这样做有三个实际好处：
  1. 任务独立于 Web 服务，关闭浏览器甚至重开都不会中断正在跑的任务
  2. 不会出现"命令行一个行为、界面另一个行为"的两套实现
  3. 识别进程崩溃时 Web 服务依然存活，可以直接看日志重试

启动方式：双击 启动工具.bat，或在命令行执行 python webui.py
"""

from __future__ import annotations

import io
import os
import queue
import re
import string
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "webapp"
# 项目自带的配置文件。界面启动时以它为基础，用户在 config.yaml 里改过的参数
# 会直接反映到界面上，不必逐个在页面里重设。
CONFIG_PATH = ROOT / "config.yaml"
sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402

PORT = int(os.environ.get("FMT_PORT", "8777"))
HOST = os.environ.get("FMT_HOST", "127.0.0.1")

app = FastAPI(title="face-match-tool")

# 进度行形如 "参考样本  45% (10/22)"、"检索图片  30% (47/157)"
# 阶段名可能有多个（参考样本 / 检索图片 / 检索视频 / 检索评分），故用中文通配
# 进度行形如「检索图片  42% (66/157)  IMG_2031.jpg」，尾部的文件名是可选的
_PROGRESS_RE = re.compile(r"^([\u4e00-\u9fa5]+)\s+\d+%\s+\((\d+)/(\d+)\)(.*)$")
_MAX_LOG = 3000


# ------------------------------------------------------------------ 任务管理

class TaskRunner:
    """以子进程方式执行 scan / archive，并把输出流收集成可增量拉取的日志。"""

    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.task = ""
        self.running = False
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.progress: Optional[Dict] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._entries: List[List] = []
        self._seq = 0
        self._lock = threading.Lock()

    def _push(self, text: str) -> None:
        with self._lock:
            self._seq += 1
            stamp = time.strftime("%H:%M:%S")
            self._entries.append([self._seq, stamp, text])
            if len(self._entries) > _MAX_LOG:
                self._entries = self._entries[-_MAX_LOG:]
        match = _PROGRESS_RE.match(text.strip())
        if match:
            self.progress = {
                "phase": match.group(1),
                "current": int(match.group(2)),
                "total": int(match.group(3)),
            }

    def start(self, cmd: List[str], task: str) -> None:
        if self.running:
            raise HTTPException(status_code=409, detail="已有任务正在运行")
        self.task = task
        self.running = True
        self.started_at = time.time()
        self.ended_at = None
        self.exit_code = None
        self.progress = None
        with self._lock:
            self._entries = []
            self._seq = 0

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["FMT_PLAIN"] = "1"          # 让 scan.py 关闭 tqdm 刷屏

        # Windows 控制台默认是 GBK，必须显式指定编码与解析错误策略
        self.proc = subprocess.Popen(
            cmd, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        try:
            if self.proc and self.proc.stdout:
                for line in self.proc.stdout:
                    self._push(line.rstrip("\r\n"))
        except Exception as exc:  # pragma: no cover
            self._push(f"[读取进程输出失败] {exc}")
        try:
            code = self.proc.wait() if self.proc else -1
        except Exception:
            code = -1
        self.exit_code = code
        self.running = False
        self.ended_at = time.time()
        self._push(f"[任务结束] 退出码 {code}")

    def stop(self) -> bool:
        """终止任务进程及其子进程树（ ProcessPoolExecutor 的 worker 也要一并清掉）。"""
        if not self.running or not self.proc:
            return False
        try:
            if os.name == "nt":
                self._push("[用户请求] 正在停止任务...")
                subprocess.run(
                    ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                    capture_output=True, timeout=20,
                )
            else:
                self.proc.terminate()
        except Exception as exc:
            self._push(f"[停止失败] {exc}")
        return True

    def snapshot(self, since: int = 0) -> Dict:
        with self._lock:
            lines = [entry for entry in self._entries if entry[0] > since]
            last = self._seq
        elapsed = 0.0
        if self.started_at:
            end = self.ended_at or time.time()
            elapsed = round(end - self.started_at, 1)
        return {
            "running": self.running,
            "task": self.task,
            "exit_code": self.exit_code,
            "elapsed": elapsed,
            "progress": self.progress,
            "lines": lines,
            "seq": last,
        }


runner = TaskRunner()


# ------------------------------------------------------------------ 配置处理

class RunRequest(BaseModel):
    config: Dict
    options: Dict = {}


def _base_config() -> tuple:
    """以项目根目录的 config.yaml 为基础读取配置。

    返回 (配置对象, 是否成功读到文件)。读不到时退化为内置默认值，
    这样即使配置文件被误删，界面依然可用。
    """
    path = str(CONFIG_PATH)
    if os.path.isfile(path):
        return load_config(path, {}), True
    return load_config(None, {}), False


def _write_runtime_config(payload: Dict, output_dir: str) -> Path:
    """把界面提交的参数落成一份完整配置文件，交给命令行脚本消费。

    基础值来自 config.yaml，界面提交的参数覆盖其上。这样界面上没显示出来的
    配置项（例如 reference.* 系列门槛）也能沿用 config.yaml 里的设定。
    """
    base, _ = _base_config()
    from core.config import deep_merge
    merged = deep_merge(base, payload or {})
    out = Path(output_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "_webui_config.yaml"
    target.write_text(
        yaml.safe_dump(merged, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target


# ------------------------------------------------------------------ 目录浏览

@app.get("/")
def index():
    html = WEB_DIR / "index.html"
    if not html.is_file():
        raise HTTPException(status_code=500, detail=f"缺少界面文件 {html}")
    return FileResponse(str(html))


@app.get("/api/config")
def get_config():
    """返回 config.yaml 合并内置默认值后的完整配置。

    界面启动时用它填充表单，改动 config.yaml 后点「重新载入配置」即可同步。
    """
    cfg, loaded = _base_config()
    return {
        "loaded": loaded,
        "path": str(CONFIG_PATH),
        "config": cfg,
    }


@app.get("/api/drives")
def drives():
    """列出可用磁盘，作为目录选择器的根节点。"""
    found = []
    for letter in string.ascii_uppercase:
        path = f"{letter}:\\"
        if os.path.exists(path):
            found.append({"name": f"{letter}:", "path": path})
    return {"drives": found}


@app.get("/api/browse")
def browse(path: str = ""):
    """列出某个目录下的子文件夹。

    浏览器访问不到本地文件系统，所以选目录只能由后端代劳。
    这里只列目录（不列文件），避免上万张照片把界面拖垮。
    """
    if not path:
        return {
            "current": "",
            "parent": None,
            "dirs": [{"name": d["name"], "path": d["path"]} for d in drives()["drives"]],
        }
    target = Path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"路径不存在：{path}")
    if not target.is_dir():
        target = target.parent

    dirs = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            if child.name.startswith(".") or child.name.startswith("$"):
                continue
            dirs.append({"name": child.name, "path": str(child)})
            if len(dirs) >= 400:
                break
    except PermissionError:
        raise HTTPException(status_code=403, detail="没有访问该目录的权限")

    parent = str(target.parent) if target.parent != target else None
    return {"current": str(target), "parent": parent, "dirs": dirs}


# ------------------------------------------------------------------ 任务预估

@app.post("/api/inspect")
def inspect(req: Dict):
    """统计目录规模并估算耗时，让用户在开跑前有心理预期。"""
    # 注意入参是完整配置对象，目录在 paths 段里，不是扁平结构。
    # 先并入 config.yaml 的基础值，避免界面未提交的项被当成默认值。
    from core.config import deep_merge
    base, _ = _base_config()
    req = deep_merge(base, req or {})

    source = ((req.get("paths") or {}).get("source_dir") or "").strip()
    if not source or not os.path.isdir(source):
        raise HTTPException(status_code=400, detail="待整理相册目录无效")

    from core.io_utils import iter_media_files, VIDEO_EXTS, HEIC_EXTS
    from core.media import probe_video_info

    interval_ms = int(((req.get("scan") or {}).get("video") or {}).get("frame_interval_ms", 2000))
    workers = int((req.get("runtime") or {}).get("workers", 6))
    exclude = list((req.get("scan") or {}).get("exclude_patterns", ["_thumb"]))

    images = videos = heics = 0
    skipped_thumbs = 0
    total_bytes = 0
    video_paths: List[str] = []
    for path, kind in iter_media_files(source, True, True, exclude=exclude):
        try:
            total_bytes += os.path.getsize(path)
        except OSError:
            pass
        if kind == "video":
            videos += 1
            video_paths.append(path)
        elif kind == "heic":
            heics += 1
        else:
            images += 1

    # 再跑一遍不做过滤，好让用户知道这套规则替我省掉了多少文件
    if exclude:
        kept = {(p, k) for p, k in iter_media_files(source, True, True, exclude=exclude)}
        skipped_thumbs = sum(
            1 for p, k in iter_media_files(source, True, True)
            if (p, k) not in kept
        )

    # 视频抽帧成本取决于时长，抽样探测以便估算得准一些
    seconds_per_video, sampled = 0.0, 0
    for path in video_paths[:12]:
        fps, frames = probe_video_info(path)
        if fps and frames:
            seconds_per_video += frames / fps
            sampled += 1
    if sampled:
        seconds_per_video /= sampled
    else:
        seconds_per_video = 180.0

    # 下面三个常数是在本机（Ryzen Zen3 12 线程、无 GPU、det_size=640）实测反解出来的，
    # 不是拍脑袋的值。反解方法：同样 151 个文件（120 图 + 31 视频）分别在 500ms 与 100ms
    # 抽帧下各跑一遍，耗时 259s 与 976s；两者之差即是多出来的 3596 帧的成本，
    # 由此得出「6 进程并行时每帧墙钟时间约 0.20 秒」。
    # 原先按 0.30 估算且直接除以 workers，结果偏乐观约 4~5 倍，这里已修正。
    per_image = 0.25    # 单张图片的墙钟耗时（6 进程基准）
    per_frame = 0.20    # 单个视频帧的墙钟耗时（6 进程基准）
    ref_build = 45.0    # 构建参考模板的固定开销，参考样本含视频时会更高

    # 每个视频至少取 1 帧，短片不会算成 0
    per_video_frames = int(seconds_per_video / (interval_ms / 1000.0)) + 1 if videos else 0
    frames_total = videos * max(1, per_video_frames)

    scale = 6.0 / max(1, workers)   # 以 6 进程为基准换算到当前并行度
    raw_seconds = (ref_build + images * per_image + frames_total * per_frame) * scale

    size_label = _human_size(total_bytes)
    est_label = _human_duration(raw_seconds)

    return {
        "images": images, "videos": videos, "heics": heics,
        "total_files": images + videos + heics,
        "total_gb": round(total_bytes / (1024 ** 3), 2),
        "total_size": size_label,
        "sampled_videos": sampled,
        "skipped_thumbs": skipped_thumbs,
        "exclude_patterns": exclude,
        "avg_video_seconds": round(seconds_per_video, 1),
        "frames_to_scan": frames_total,
        "workers": workers,
        "estimate": est_label,
        "has_unsupported": heics > 0,
    }


def _human_size(num_bytes: int) -> str:
    """小到 KB、大到 TB 都能读得顺的单位换算。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _human_duration(seconds: float) -> str:
    """耗时换算：不足 1 分钟显示秒，超过 1.5 小时显示小时。"""
    if seconds < 60:
        return f"{max(1, round(seconds))} 秒"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} 分钟"
    return f"{minutes / 60:.1f} 小时"


# ------------------------------------------------------------------ 执行接口

@app.post("/api/scan")
def api_scan(req: RunRequest):
    payload = req.config or {}
    options = req.options or {}
    paths = payload.get("paths") or {}
    for key in ("reference_dir", "source_dir", "output_dir"):
        if not paths.get(key):
            raise HTTPException(status_code=400, detail=f"请填写 {key} 对应的目录")

    cfg_path = _write_runtime_config(payload, paths["output_dir"])
    cmd = [sys.executable, str(ROOT / "scan.py"), "--config", str(cfg_path)]
    if options.get("no_resume"):
        cmd.append("--no-resume")
    if options.get("limit"):
        cmd += ["--limit", str(int(options["limit"]))]
    runner.start(cmd, "scan")
    return {"ok": True, "config_path": str(cfg_path)}


@app.post("/api/archive")
def api_archive(req: RunRequest):
    payload = req.config or {}
    options = req.options or {}
    paths = payload.get("paths") or {}
    if not paths.get("output_dir"):
        raise HTTPException(status_code=400, detail="请填写输出目录")

    cfg_path = _write_runtime_config(payload, paths["output_dir"])
    cmd = [sys.executable, str(ROOT / "archive.py"), "--config", str(cfg_path)]
    if options.get("dry_run"):
        cmd.append("--dry-run")
    if options.get("archive_mode"):
        cmd += ["--mode", str(options["archive_mode"])]
    runner.start(cmd, "archive")
    return {"ok": True, "config_path": str(cfg_path)}


@app.post("/api/stop")
def api_stop():
    stopped = runner.stop()
    return {"ok": stopped, "message": "" if stopped else "当前没有正在运行的任务"}


@app.get("/api/status")
def api_status(since: int = 0):
    return runner.snapshot(since)


@app.post("/api/open")
def api_open(req: Dict):
    """在资源管理器中打开指定目录（仅限本地使用）。"""
    path = req.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="路径为空")
    target = Path(path)
    if not target.exists():
        target = target.parent
    if target.exists():
        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
    return {"ok": True}


# ------------------------------------------------------------------ 结果读取

@app.get("/api/report")
def api_report(out: str = ""):
    """读取扫描报告，生成分档统计与高分预览列表。"""
    if not out:
        raise HTTPException(status_code=400, detail="未指定输出目录")
    report = Path(out) / "scan_report.csv"
    if not report.is_file():
        return {"exists": False}

    import csv
    from core.config import load_config as _load
    from core.similarity import bin_label

    # 优先用扫描时落下的运行配置；没有（例如用命令行跑的）就回落到 config.yaml
    cfg_path = Path(out) / "_webui_config.yaml"
    cfg = _load(str(cfg_path) if cfg_path.is_file() else str(CONFIG_PATH), {})

    rows = []
    with open(report, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    scored = [r for r in rows if r.get("status") == "scored" and r.get("score")]
    for row in scored:
        row["score"] = float(row["score"])

    width = int(cfg.get("bins", {}).get("width", 10))
    counts: Dict[str, int] = {}
    for row in scored:
        label = bin_label(row["score"], cfg)
        counts[label] = counts.get(label, 0) + 1

    histogram = []
    for low in range(100 - width, -1, -width):
        label = f"{low:02d}-{low + width:02d}"
        histogram.append({"bin": label, "count": counts.get(label, 0)})

    min_score = float(cfg.get("bins", {}).get("min_score", 30))
    scored.sort(key=lambda r: -r["score"])
    top = [{
        "name": os.path.basename(r["path"]),
        "path": r["path"],
        "score": round(r["score"], 1),
        "cos": r.get("cos", ""),
        "kind": r.get("kind", ""),
        "ts": r.get("ts", ""),
        "ts_source": r.get("ts_source", ""),
        "bin": bin_label(r["score"], cfg),
    } for r in scored[:30]]

    return {
        "exists": True,
        "total": len(rows),
        "scored": len(scored),
        "above_min": sum(1 for r in scored if r["score"] >= min_score),
        "min_score": min_score,
        "histogram": histogram,
        "top": top,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="face-match-tool 图形界面")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}/"
    print("=" * 56)
    print("  face-match-tool 图形界面")
    print("=" * 56)
    print(f"  访问地址: {url}")
    print("  关闭此窗口即停止服务，正在运行的任务会一并结束。")
    print("=" * 56)

    if not args.no_browser:
        threading.Timer(1.2, lambda: _open_browser(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def _open_browser(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    main()
