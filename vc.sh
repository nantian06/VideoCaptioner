#!/usr/bin/env bash
# 命令行入口，例如：./vc.sh transcribe 直播.mp4 --local-model funasr-nano
DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -x "$DIR/.venv/bin/videocaptioner" ]; then
    echo "还没安装，请先运行 ./install_linux.sh"
    exit 1
fi
exec "$DIR/.venv/bin/videocaptioner" "$@"
