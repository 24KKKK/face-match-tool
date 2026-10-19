# -*- coding: utf-8 -*-
"""把 face-match-tool 打包成一个「免装 Python」的绿色文件夹。

做法：把当前这台机器上能跑的完整 Python 解释器（含 insightface/onnxruntime/cv2
等所有依赖）连同项目代码、buffalo_l 模型、ffprobe 一起复制到一个目录里，再写一个
启动器 .bat。对方只要解压、双击 .bat 即可，不需要装任何东西、也不需要联网。

为什么不用 PyInstaller 打成单个 exe：onnxruntime / insightface 这类带原生 DLL 的
库用 PyInstaller 打包极易出现「打包成功但启动报错 / 找不到 DLL」的问题，而且打包后
体积反而更大、启动更慢。整体复制解释器虽然目录大一点，但 100% 可靠。

使用：
    python pack_portable.py                 # 输出到 人脸识别分类\人脸识别归档图形化工具绿色版
    python pack_portable.py "E:\share"      # 指定输出根目录
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# ---------- 源路径（本机已有、已验证可运行的环境） ----------
PROJECT = Path(__file__).resolve().parent                 # face-match-tool 根目录
SRC_PY = Path(r"D:\myprogram\Python\Python312")           # 已装好全部依赖的解释器
SRC_MODEL = Path.home() / ".insightface" / "models" / "buffalo_l"

# ---------- 输出路径 ----------
# 默认放在项目（face-match-tool）的同级目录，新建一个子文件夹承载绿色版打包结果
DEFAULT_OUT = PROJECT.parent / "人脸识别归档图形化工具绿色版"
OUT_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
OUT = OUT_ROOT  # 项目文件直接铺在 OUT 根下，python 解释器放在 OUT/python

# 复制 Python 时要跳过的目录（运行时不需要，能省几百 MB）
PY_SKIP_DIRS = {
    "Lib\\test", "Lib\\tests", "Lib\\site-packages\\pip",
    "Lib\\site-packages\\setuptools", "Lib\\site-packages\\wheel",
    "include", "libs", "Doc", "tcl", "share", "Scripts",
}
# 项目里不要带进绿色版的内容
PROJ_SKIP = {"__pycache__", "_testdata", ".git", ".workbuddy"}
PROJ_SKIP_PREFIX = ("_backup",)  # 形如 _backup_20261016
# 不要把绿色版输出目录自身再复制进来（否则会递归嵌套）
PROJ_SKIP.add(OUT.name)


def log(msg: str) -> None:
    print(msg, flush=True)


def _ignore_py(path: str, names):
    p = Path(path)
    rel = str(p.relative_to(SRC_PY)).replace("/", "\\")
    skip = set()
    for n in names:
        full = str((p / n).relative_to(SRC_PY)).replace("/", "\\")
        if full in PY_SKIP_DIRS or n in PY_SKIP_DIRS:
            skip.add(n)
            continue
        # 跳过单个大文件：我们不需要源码包、文档等
        if n.endswith((".pyc",)):
            skip.add(n)
    return skip


def _ignore_proj(path: str, names):
    skip = set()
    for n in names:
        if n in PROJ_SKIP or any(n.startswith(p) for p in PROJ_SKIP_PREFIX):
            skip.add(n)
            continue
        if n.endswith(".pyc") or n == "__pycache__":
            skip.add(n)
    return skip


def copy_python() -> None:
    dst = OUT / "python"
    if dst.exists():
        log(f"[清理] 已存在的 {dst}")
        shutil.rmtree(dst)
    log(f"[复制] Python 解释器 {SRC_PY} -> {dst} (已剔除测试/文档，约省 200MB)")
    shutil.copytree(SRC_PY, dst, ignore=_ignore_py, symlinks=False)
    # 清掉缓存，防止搬运后用到旧机器上的绝对路径残留
    for pc in dst.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)


def copy_project() -> None:
    if OUT.exists():
        # 只删我们可能要重写的文件，其余保留
        for f in ("webui.py", "scan.py", "archive.py", "config.yaml"):
            (OUT / f).unlink(missing_ok=True)
    log(f"[复制] 项目代码 {PROJECT} -> {OUT}")
    shutil.copytree(PROJECT, OUT, ignore=_ignore_proj, dirs_exist_ok=True,
                    copy_function=shutil.copy2)
    # 清缓存
    for pc in OUT.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)


def copy_model() -> None:
    dst = OUT / "models" / "buffalo_l"
    if dst.exists():
        shutil.rmtree(dst)
    if not SRC_MODEL.is_dir():
        log(f"[警告] 找不到模型 {SRC_MODEL}，绿色版将依赖对方首次运行时联网下载！")
        return
    log(f"[复制] 模型 {SRC_MODEL} -> {dst}")
    shutil.copytree(SRC_MODEL, dst)
    log(f"[完成] 模型就位，对方无需联网")


def patch_config() -> None:
    """把打包版默认输出目录改到绿色目录内的『归档输出』，避免对方没有 D:\\图片分类。"""
    cfg = OUT / "config.yaml"
    if not cfg.is_file():
        log("[警告] 未找到 config.yaml，跳过默认输出目录改写")
        return
    text = cfg.read_text(encoding="utf-8")
    text = re.sub(r'output_dir:\s*""', "output_dir: 归档输出", text, count=1)
    cfg.write_text(text, encoding="utf-8")
    log("[改写] config.yaml 默认输出目录 -> ./归档输出")


def write_launcher() -> None:
    bat = OUT / "启动工具.bat"
    content = (
        "@echo off\r\n"
        "chcp 65001 >nul 2>nul\r\n"
        "setlocal\r\n"
        'cd /d "%~dp0"\r\n'
        'set "PY=%~dp0python\\python.exe"\r\n'
        'if not exist "%PY%" (\r\n'
        "  echo Python 未找到，请确认整个文件夹已完整解压。\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        "echo Starting 人脸识别归档图形化工具 ...\r\n"
        "echo 浏览器将自动打开 http://127.0.0.1:8777/\r\n"
        "echo 保持此窗口打开；关闭它即停止工具。\r\n"
        "echo.\r\n"
        '"%PY%" webui.py\r\n'
        "if errorlevel 1 pause\r\n"
    )
    bat.write_text(content, encoding="utf-8")
    log(f"[生成] 启动器 {bat}")


def write_readme() -> None:
    readme = OUT / "使用说明.txt"
    content = (
        "人脸识别归档图形化工具（绿色版 / 免安装）\n"
        "========================================\n\n"
        "【怎么用】\n"
        "1. 把整个文件夹解压到任意位置（路径不要含中文/空格更稳妥，但非必须）。\n"
        "2. 双击「启动工具.bat」。\n"
        "3. 浏览器会自动打开 http://127.0.0.1:8777/ ，按界面提示选择：\n"
        "     参考样本目录（孩子的清晰照片/视频）\n"
        "     待整理相册目录（要筛查的相册）\n"
        "     输出目录（默认在本文件夹内的「归档输出」）\n"
        "4. 点「开始扫描」，完成后点「开始归档」即可。\n\n"
        "【说明】\n"
        "- 本文件夹自带 Python 运行环境与 buffalo_l 模型，对方电脑无需安装任何东西、无需联网。\n"
        "- 首次启动会解压/加载模型，约 10~20 秒，请稍候。\n"
        "- 关闭「启动工具.bat」黑窗口即停止工具；正在跑的任务会一并结束。\n"
        "- 若 8777 端口被占用，可编辑本文件夹里的 webui.py 末尾 PORT 值，或先关掉占用端口的程序。\n\n"
        "【性能】\n"
        "- 纯 CPU 推理。本机（12 核）约 0.2 秒/视频帧、0.25 秒/图片。\n"
        "- 想更快：调大抽帧间隔（毫秒），或开启图片-视频配对（封面图达标则视频免抽帧）。\n"
    )
    readme.write_text(content, encoding="utf-8")
    log(f"[生成] 使用说明 {readme}")


def main() -> int:
    if not SRC_PY.is_dir():
        log(f"[错误] 源 Python 不存在：{SRC_PY}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    copy_python()
    copy_project()
    copy_model()
    patch_config()
    write_launcher()
    write_readme()
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    log("=" * 50)
    log(f"绿色版已生成：{OUT}")
    log(f"总大小约 {total / 1024**3:.2f} GB")
    log("把它整个文件夹压缩发给对方，对方解压后双击「启动工具.bat」即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
