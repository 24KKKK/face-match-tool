# -*- coding: utf-8 -*-
"""配置加载与默认值合并。

用法：
    from core.config import load_config
    cfg = load_config("config.yaml", cli_overrides)
所有模块只依赖这个 dict，方便后续扩展。
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# 全部默认值：即使 config.yaml 缺失或字段不全，程序也能正常启动
DEFAULTS: Dict[str, Any] = {
    "paths": {
        "reference_dir": "",
        "source_dir": "",
        "output_dir": "",
        # 可选：手动指定 ffprobe 路径。留空则依次查找 _vendor/ffprobe.exe 与系统 PATH
        "ffprobe_path": "",
    },
    "model": {
        # insightface 模型包名。buffalo_l = RetinaFace 检测 + ArcFace(R50) 识别
        "name": "buffalo_l",
        # 检测输入边长，越大越能召回小脸，但更慢。低配可用 [320, 320]
        "det_size": [640, 640],
        "providers": ["CPUExecutionProvider"],
        "ctx_id": -1,  # -1 表示 CPU
        # 单个进程内的推理线程上限。0 = 不限。
        # onnxruntime 默认给每个进程开满 CPU 逻辑核数的线程，
        # 多进程不开限制会严重过度订阅，实测本机 6 进程时吞吐只有最优配置的 43%。
        # 经验取值：进程数 × intra_op_num_threads ≈ CPU 逻辑核数。
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
    },
    "reference": {
        # 参考样本的质量过滤门槛：脏样本会把模板向量带偏
        "min_det_score": 0.60,
        "min_face_px": 80,      # 人脸框短边最小像素
        "max_abs_yaw": 35,      # 左右侧脸角度上限
        "max_abs_pitch": 35,    # 俯仰角度上限
        "reject_outliers": True,  # 剔除与均值差异过大的离群样本
        "video_frame_interval_ms": 1000,  # 参考视频抽帧间隔，可比待检索更密
    },
    "scan": {
        "image": {"enabled": True},
        # 文件名含这些关键词的跳过不扫。缩略图（如 xxx_thumb.jpg）往往占
        # 相册文件数的绝大多数，但分辨率不足，检出率极低，扫了近乎浪费。
        "exclude_patterns": ["_thumb"],
        "video": {
            "enabled": True,
            "frame_interval_ms": 2000,  # 视频抽帧间隔（毫秒）
            "max_frames_per_video": 3000,  # 单视频最多抽帧数，防止长片拖死
            "aggregate": "max",  # max | top3_mean：视频级分数聚合方式
            "min_hit_frames": 1,  # 命中帧数低于此值视为未命中
        },
        # 微信等应用保存视频时会生成同名封面图（通常就是第一帧）。
        # 这类成对文件可以合并处理，是最大的一处提速来源。
        "pairing": {
            "enabled": True,          # 关闭则图片视频各自独立扫描
            "reuse_min_score": 70,    # 封面图分数达标时，视频不再抽帧，直接沿用该分数
            "unify_score": True,      # 组内分数统一取较高者，保证归档到同一分档目录
        },
    },
    "similarity": {
        # calibrated: 把余弦判别区间拉伸到 0-100%；raw: 余弦值直接乘 100
        "mapping": "calibrated",
        # 本机 329 个文件实测标定：正样本（参考自照）最低 0.599，
        # 待整理相册的余弦在 0.47~0.53 有一段天然空档，0.54 以上连成一整片
        "calib_low": 0.50,    # 低于此余弦值视为 0%
        "calib_high": 0.90,   # 达到此余弦值视为 100%
    },
    "bins": {
        "width": 10,      # 分档宽度（%）
        "min_score": 30,  # 低于此分数不输出，仅记录在报告中
    },
    "timestamp": {
        # 优先级链：exif(图片) / meta(视频容器元数据) / mtime / birthtime
        "prefer_sources": {
            "image": ["exif", "mtime", "birthtime"],
            "video": ["meta", "mtime", "birthtime"],
        },
        "fallback": "unknown",  # 全部提取失败时的占位名
        "min_valid_year": 1975,  # 早于此年份视为无效时间戳
    },
    "archive": {
        "mode": "link",  # link(硬链接优先) | copy(真实复制)
        "link_fallback_copy": True,
        "naming": "{date}_{time}__s{score}__{stem}{ext}",
        "keep_extension_case": True,
    },
    "runtime": {
        # 并行进程数。本机 12 逻辑核实测峰值在 3（配 2 线程/进程），
        # 6 进程且不限线程时反而比单进程慢 31%，切勿盲目调大。
        "workers": 3,
        "log_level": "INFO",
    },
}


# 有分数、可参与统计与归档的结果状态。
# paired = 沿用配对封面图分数的视频，它同样有分数值，
# 必须与 scored 同等对待，否则配对的视频会在摘要与归档阶段被整体漏掉。
ALIVE_STATUSES = ("scored", "paired")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归把 override 合并进 base，返回新对象（不修改入参）。"""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None = None, cli_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """读取 YAML 配置并与默认值、命令行覆盖值做三层合并。"""
    cfg = copy.deepcopy(DEFAULTS)

    if path and os.path.isfile(path):
        if yaml is None:
            raise RuntimeError("缺少 PyYAML，请先执行：pip install pyyaml")
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        if not isinstance(user_cfg, dict):
            raise ValueError(f"配置文件内容格式不合法：{path}")
        cfg = deep_merge(cfg, user_cfg)

    if cli_overrides:
        cfg = deep_merge(cfg, cli_overrides)

    return cfg


def get_output_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """根据输出根目录派生工作文件路径。"""
    out_root = cfg["paths"]["output_dir"]
    return {
        "root": out_root,
        "db": os.path.join(out_root, "_state.db"),
        "report": os.path.join(out_root, "scan_report.csv"),
        "dataset": os.path.join(out_root, "dataset"),
        "ref_report": os.path.join(out_root, "reference_samples.csv"),
    }
