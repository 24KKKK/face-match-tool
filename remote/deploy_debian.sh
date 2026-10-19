#!/usr/bin/env bash
# ============================================================
# face-match-tool  Debian 端环境部署脚本
#
# 用法（在 Debian 上执行）：
#   把整个 face-match-tool 目录拷到 Debian 任意位置，然后：
#     cd face-match-tool/remote
#     bash deploy_debian.sh
#
# 说明：
#   Debian 的 i5-2450M 比 Windows 主力机慢 6~8 倍，且只有 4GB 内存。
#   这里的配置按「慢但稳定」来调，不要盲目加大并发。
# ============================================================
set -euo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/face-match-env"
USE_MIRROR="${USE_MIRROR:-1}"

echo "============================================================"
echo " face-match-tool  Debian 环境部署"
echo "============================================================"
echo " 工具目录: ${TOOL_DIR}"
echo " 虚拟环境: ${VENV_DIR}"
echo ""

# ---------- 1. 检查 Python ----------
if ! command -v python3 >/dev/null 2>&1; then
    echo "[错误] 未找到 python3，请先安装：sudo apt install python3 python3-venv"
    exit 1
fi
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "[1/5] Python 版本 ${PY_VER}"

# ---------- 2. 系统依赖 ----------
echo "[2/5] 安装系统依赖（ffmpeg 用于读取视频真实拍摄时间）"
if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update -qq && apt-get install -y -qq ffmpeg python3-venv python3-pip
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg python3-venv python3-pip
    else
        echo "      跳过（无 apt 权限，可手动执行 sudo apt install ffmpeg）"
    fi
else
    echo "      跳过（非 Debian 系系统）"
fi

if command -v ffprobe >/dev/null 2>&1; then
    echo "      ffprobe 已就绪: $(command -v ffprobe)"
else
    echo "      [警告] 未找到 ffprobe，视频拍摄时间将退用文件修改时间"
fi

# ---------- 3. 虚拟环境 ----------
echo "[3/5] 创建虚拟环境"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "      已创建 ${VENV_DIR}"
else
    echo "      已存在，复用"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip -q

# ---------- 4. Python 依赖 ----------
echo "[4/5] 安装 Python 依赖（耗时较长，请耐心）"
PIP_FLAGS=""
if [ "${USE_MIRROR}" = "1" ]; then
    PIP_FLAGS="-i https://pypi.tuna.tsinghua.edu.cn/simple"
    echo "      使用清华镜像加速"
fi

pip install -q ${PIP_FLAGS} numpy opencv-python-headless onnxruntime pillow pyyaml tqdm
pip install -q ${PIP_FLAGS} insightface || {
    echo "      [警告] insightface 安装失败，尝试不带镜像重试..."
    pip install -q insightface
}

# HEIC 支持（iPhone 照片），失败不阻断
pip install -q ${PIP_FLAGS} pillow-heif 2>/dev/null || echo "      pillow-heif 未安装（不影响常规 JPG/MP4）"

# ---------- 5. 模型 ----------
echo "[5/5] 预下载人脸模型（约 281MB）"
python - <<'PY'
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("      模型就绪")
except Exception as exc:
    print(f"      [警告] 模型下载失败: {exc}")
    print("      首次运行 scan.py 时会自动重试；也可手动下载 buffalo_l.zip")
    print("      放到 ~/.insightface/models/ 下解压")
PY

echo ""
echo "============================================================"
echo " 部署完成"
echo "============================================================"
echo ""
echo " 后续使用（记得先激活环境）："
echo "   source ${VENV_DIR}/bin/activate"
echo "   cd ${TOOL_DIR}"
echo "   python scan.py --ref <参考目录> --src <相册目录> --out <输出目录> --workers 2"
echo ""
echo " 重要：Debian 只有 4GB 内存，每个识别进程约占 500~800MB，"
echo "       并发不要超过 2，否则会频繁触发内存回收甚至被系统杀掉。"
echo "       预计速度为 Windows 主力机的 1/6 ~ 1/8，适合夜间挂机跑。"
