#!/usr/bin/env bash
# 启动图形界面（先运行 ./install_linux.sh 安装）
cd "$(dirname "$0")"
if [ ! -x .venv/bin/videocaptioner-gui ]; then
    echo "还没安装，请先运行 ./install_linux.sh"
    exit 1
fi
exec .venv/bin/videocaptioner-gui "$@"
