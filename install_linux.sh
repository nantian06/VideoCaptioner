#!/usr/bin/env bash
# 本地中文字幕版（VideoCaptioner 定制版）Linux 安装脚本
#
# 用法：
#   ./install_linux.sh          自动：检测到 N 卡驱动就装 GPU 版，否则装 CPU 版
#   ./install_linux.sh --gpu    强制装 GPU 版（sherpa-onnx CUDA 版 + CUDA12/cuDNN9 运行库，约 3GB）
#   ./install_linux.sh --cpu    强制装 CPU 版（也可以用来从 GPU 版换回 CPU 版）
#
# 可选环境变量：
#   VC_CUDA_WHEEL=/path/to/sherpa_onnx-...whl   用自己下载好的 CUDA 版 wheel（网络下不动时用）
#   UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple   换 PyPI 镜像
#
# 装完后用 ./run_gui.sh 启动界面，./vc.sh 用命令行。

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"

SHERPA_VERSION="1.13.8"
CUDA_WHEEL_NAME="sherpa_onnx-${SHERPA_VERSION}+cuda12.cudnn9.onnxruntime1.28.2-cp312-cp312-linux_x86_64.whl"
CUDA_WHEEL_PATH="csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/${SHERPA_VERSION}/${CUDA_WHEEL_NAME}"
CUDA_WHEEL_URLS=(
    "https://huggingface.co/${CUDA_WHEEL_PATH}"
    "https://hf-mirror.com/${CUDA_WHEEL_PATH}"
)
NVIDIA_PKGS=(
    nvidia-cuda-runtime-cu12
    nvidia-cuda-nvrtc-cu12
    nvidia-nvjitlink-cu12
    nvidia-cublas-cu12
    nvidia-cufft-cu12
    nvidia-curand-cu12
    "nvidia-cudnn-cu12>=9,<10"
)

info() { echo -e "\033[0;34m[信息]\033[0m $*"; }
ok()   { echo -e "\033[0;32m[完成]\033[0m $*"; }
warn() { echo -e "\033[1;33m[注意]\033[0m $*"; }
err()  { echo -e "\033[0;31m[错误]\033[0m $*"; }

MODE="auto"
for arg in "$@"; do
    case "$arg" in
        --gpu) MODE="gpu" ;;
        --cpu) MODE="cpu" ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) err "不认识的参数：$arg（可用 --gpu / --cpu）"; exit 1 ;;
    esac
done

if [ "$MODE" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        MODE="gpu"
        info "检测到 N 卡：$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)，安装 GPU 版"
    else
        MODE="cpu"
        info "没检测到 N 卡驱动（nvidia-smi），安装 CPU 版"
    fi
fi

if [ "$(uname -m)" != "x86_64" ] && [ "$MODE" = "gpu" ]; then
    warn "GPU 版 wheel 只提供 x86_64，当前是 $(uname -m)，改装 CPU 版"
    MODE="cpu"
fi

# ---------- 1. 系统依赖 ----------
apt_install() {
    if command -v apt-get >/dev/null 2>&1; then
        local missing=()
        for p in "$@"; do
            dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
        done
        if [ ${#missing[@]} -gt 0 ]; then
            info "安装系统包：${missing[*]}（需要 sudo 密码）"
            sudo apt-get update -y && sudo apt-get install -y "${missing[@]}" \
                || warn "系统包安装失败，可以稍后手动执行：sudo apt-get install ${missing[*]}"
        fi
        return 0
    fi
    return 1
}

if ! command -v ffmpeg >/dev/null 2>&1; then
    info "没找到 ffmpeg，尝试安装（识别、合成视频都要用）"
    apt_install ffmpeg \
        || { command -v dnf >/dev/null 2>&1 && sudo dnf install -y ffmpeg; } \
        || { command -v pacman >/dev/null 2>&1 && sudo pacman -S --noconfirm ffmpeg; } \
        || warn "请手动安装 ffmpeg"
fi

# Qt 界面在 Debian/Ubuntu 上需要的 xcb 库（缺了会报 "Could not load the Qt platform plugin xcb"）
apt_install libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 \
    libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libxcb-cursor0 fonts-noto-cjk || true

# ---------- 2. uv ----------
if ! command -v uv >/dev/null 2>&1; then
    export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    info "安装 uv（Python 包管理器）"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        wget -qO- https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv：$(uv --version)"

# ---------- 3. Python 环境（固定 3.12，CUDA 版 wheel 只有 cp312） ----------
info "创建 Python 3.12 环境并安装依赖（第一次会比较久）"
uv sync --python 3.12
ok "基础依赖已装好（含 CPU 版 sherpa-onnx）"

# ---------- 4. GPU 版 ----------
install_gpu() {
    local wheel=""
    if [ -n "${VC_CUDA_WHEEL:-}" ]; then
        wheel="$VC_CUDA_WHEEL"
        [ -f "$wheel" ] || { err "VC_CUDA_WHEEL 指向的文件不存在：$wheel"; return 1; }
    else
        mkdir -p "$ROOT/.cache"
        wheel="$ROOT/.cache/$CUDA_WHEEL_NAME"
        if [ ! -s "$wheel" ]; then
            for url in "${CUDA_WHEEL_URLS[@]}"; do
                info "下载 CUDA 版 sherpa-onnx：$url"
                if curl -fL --retry 2 --connect-timeout 20 -o "$wheel.part" "$url"; then
                    mv "$wheel.part" "$wheel"
                    break
                fi
                warn "这个地址下载失败，换下一个"
            done
        fi
        [ -s "$wheel" ] || { err "CUDA 版 wheel 下载失败"; return 1; }
    fi

    info "安装 CUDA 12 / cuDNN 9 运行库（pip 包，约 2-3GB，不需要系统装 CUDA Toolkit）"
    uv pip install --python "$PY" "${NVIDIA_PKGS[@]}" || return 1

    info "替换为 CUDA 版 sherpa-onnx"
    uv pip uninstall --python "$PY" sherpa-onnx sherpa-onnx-core || true
    uv pip install --python "$PY" --no-deps "$wheel" || return 1
    "$PY" -c "import sherpa_onnx" || return 1
    echo gpu > "$ROOT/.venv/.vc_variant"
    return 0
}

if [ "$MODE" = "gpu" ]; then
    if install_gpu; then
        ok "GPU 版已安装"
    else
        warn "GPU 版安装失败，恢复为 CPU 版（软件照样能用，只是识别慢一些）"
        uv pip uninstall --python "$PY" sherpa-onnx || true
        uv sync --python 3.12 --reinstall-package sherpa-onnx --reinstall-package sherpa-onnx-core
        echo cpu > "$ROOT/.venv/.vc_variant"
    fi
else
    echo cpu > "$ROOT/.venv/.vc_variant"
fi

chmod +x "$ROOT/run_gui.sh" "$ROOT/vc.sh" 2>/dev/null || true

# ---------- 5. 自检 ----------
echo
info "环境自检："
"$ROOT/.venv/bin/videocaptioner" doctor || true
echo
ok "安装完成。启动界面：./run_gui.sh    命令行：./vc.sh transcribe 视频.mp4"
echo "   第一次识别时会自动下载模型（SenseVoice 约 160MB），也可以在 设置 → 模型管理 里提前下载。"
warn "以后不要直接运行 uv sync / uv run，那会把 GPU 版换回 CPU 版；要更新请重新运行本脚本。"
