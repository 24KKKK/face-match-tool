# -*- coding: utf-8 -*-
"""生成端到端测试数据（使用真实人脸，才能验证检测->打分->分档的完整链路）。

刻意使用中文目录名与中文文件名 —— 这是 Windows 上最典型的翻车点。
覆盖场景：EXIF 时间 / 无 EXIF / 异常 EXIF / EXIF 竖拍方向 / 视频抽帧 / 不支持格式。
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).parent / "_testdata"
REF_DIR = ROOT / "参考样本-小明"
SRC_DIR = ROOT / "待检索相册" / "2019 国庆假期"
ASSET = Path(__file__).parent / "_asset_lena.jpg"


def load_face() -> Image.Image:
    if not ASSET.exists():
        raise SystemExit(f"缺少测试素材 {ASSET}，请先确保该文件存在")
    return Image.open(ASSET).convert("RGB")


def _ensure_clean():
    if ROOT.exists():
        shutil.rmtree(ROOT, ignore_errors=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(parents=True, exist_ok=True)


def save_with_exif(img: Image.Image, path: Path, dt_text: str | None = None, orientation: int | None = None):
    exif = img.getexif()
    if dt_text:
        exif[36867] = dt_text      # DateTimeOriginal
        exif[36868] = dt_text
    if orientation:
        exif[274] = orientation    # Orientation
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=88, exif=exif)


def write_mp4_from_image(base: Image.Image, path: Path, frames: int = 75, size=(512, 512), fps: int = 25):
    """用真实人脸生成视频，用于验证抽帧链路。VideoWriter 不支持中文路径，故先写临时文件。"""
    tmp_dir = tempfile.mkdtemp(prefix="fmt_gen_")
    tmp_path = os.path.join(tmp_dir, "clip.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    for i in range(frames):
        # 前 15 帧用别的颜色（无人脸），后面才出现人脸，模拟"孩子中途入镜"
        if i < 15:
            frame = np.full((size[1], size[0], 3), (60, 120, 180), np.uint8)
        else:
            scale = 0.85 + 0.15 * (i % 10) / 10.0
            w, h = base.size
            side = int(min(w, h) * scale)
            crop = base.resize((side, side))
            canvas = Image.new("RGB", (512, 512), (40, 40, 40))
            canvas.paste(crop, ((512 - side) // 2, (512 - side) // 2))
            frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(tmp_path, path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def main():
    _ensure_clean()
    face = load_face()

    # ---------- 参考样本：同一个人的不同形态 ----------
    save_with_exif(face, REF_DIR / "参考1 正面.jpg", "2023:03:12 10:20:30")

    bright = ImageEnhance.Brightness(face).enhance(1.25)
    save_with_exif(bright, REF_DIR / "参考2 亮一点.jpg", "2023:03:12 10:21:15")

    small = face.resize((int(face.width * 0.7), int(face.height * 0.7)))
    save_with_exif(small, REF_DIR / "参考3 远景.jpg", "2023:03:12 10:22:00")

    # 竖拍：图像内容旋转 90 度存储 + EXIF 标记，检验方向修正是否被正确应用
    rotated = face.rotate(90, expand=True)
    save_with_exif(rotated, REF_DIR / "参考4 竖拍.jpg", "2023:03:12 10:23:00", orientation=6)

    write_mp4_from_image(face, REF_DIR / "参考视频.mp4", frames=60)

    # ---------- 待检索：混合命中与未命中 ----------
    save_with_exif(face, SRC_DIR / "命中A 同一个人.jpg", "2024:05:01 14:25:30")
    save_with_exif(ImageEnhance.Brightness(face).enhance(0.85),
                   SRC_DIR / "命中B 暗一点.jpg", "2024:05:01 14:26:10")
    save_with_exif(face.resize((300, 300)), SRC_DIR / "命中C 小图.jpg", None)          # 无 EXIF
    save_with_exif(face, SRC_DIR / "命中D 异常时间.jpg", "0000:00:00 00:00:00")        # 坏 EXIF
    save_with_exif(face.rotate(90, expand=True), SRC_DIR / "命中E 竖拍.jpg",
                   "2024:07:15 09:08:07", orientation=6)
    save_with_exif(face, SRC_DIR / "照片 中文名 带空格.jpg", "2024:08:20 18:30:00")

    # 无关图片：纯色，应当检不出人脸
    Image.new("RGB", (400, 400), (30, 90, 160)).save(SRC_DIR / "无关 风景.png")
    Image.new("RGB", (500, 300), (200, 60, 60)).save(SRC_DIR / "无关 墙.jpg")

    write_mp4_from_image(face, SRC_DIR / "视频 孩子入镜.mp4", frames=100)
    write_mp4_from_image(face, SRC_DIR / "视频 002 中文名.mp4", frames=80)

    (SRC_DIR / "说明.txt").write_text("这不是媒体文件，应被忽略", encoding="utf-8")

    print(f"测试数据已生成：{ROOT}")
    print(f"  参考：{REF_DIR.relative_to(ROOT)}  检索：{SRC_DIR.relative_to(ROOT)}")
    for path in sorted(SRC_DIR.rglob("*")):
        if path.is_file():
            print(f"    - {path.name}")


if __name__ == "__main__":
    main()
