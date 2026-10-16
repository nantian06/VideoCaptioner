@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: 本地中文字幕版（VideoCaptioner 定制版）Windows 安装脚本
::
:: 用法（双击即可，自动判断）：
::   install_windows.bat          自动：检测到 N 卡驱动就装 GPU 版，否则装 CPU 版
::   install_windows.bat --gpu    强制 GPU 版（sherpa-onnx CUDA 版 + CUDA12/cuDNN9 运行库，约 3GB）
::   install_windows.bat --cpu    强制 CPU 版（也可用来从 GPU 版换回）
::
:: 可选环境变量：
::   VC_CUDA_WHEEL=D:\下载\sherpa_onnx-...whl   用自己下载好的 CUDA 版 wheel
::   UV_DEFAULT_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple   换 PyPI 镜像

cd /d "%~dp0"
set "ROOT=%CD%"
set "PY=%ROOT%\.venv\Scripts\python.exe"

set "SHERPA_VERSION=1.13.8"
set "CUDA_WHEEL_NAME=sherpa_onnx-%SHERPA_VERSION%+cuda12.cudnn9-cp312-cp312-win_amd64.whl"
set "CUDA_WHEEL_PATH=csukuangfj2/sherpa-onnx-wheels/resolve/main/cuda/%SHERPA_VERSION%/%CUDA_WHEEL_NAME%"

set "MODE=auto"
if /i "%~1"=="--gpu" set "MODE=gpu"
if /i "%~1"=="--cpu" set "MODE=cpu"

if "%MODE%"=="auto" (
    set "MODE=cpu"
    where nvidia-smi >nul 2>&1
    if !errorlevel! equ 0 (
        nvidia-smi -L >nul 2>&1
        if !errorlevel! equ 0 set "MODE=gpu"
    )
    if "!MODE!"=="gpu" (
        echo [信息] 检测到 N 卡，安装 GPU 版
    ) else (
        echo [信息] 没检测到 N 卡驱动（nvidia-smi），安装 CPU 版
    )
)

:: ---------- 1. uv ----------
where uv >nul 2>&1
if !errorlevel! neq 0 (
    set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv\bin;%PATH%"
)
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo [信息] 安装 uv（Python 包管理器）
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv\bin;!PATH!"
)
where uv >nul 2>&1
if !errorlevel! neq 0 (
    echo [错误] uv 安装失败，请手动安装：https://docs.astral.sh/uv/
    pause
    exit /b 1
)

:: ---------- 2. ffmpeg ----------
where ffmpeg >nul 2>&1
if !errorlevel! neq 0 (
    if exist "%ROOT%\resource\bin\ffmpeg.exe" (
        echo [信息] 使用自带的 ffmpeg
    ) else (
        echo [信息] 没找到 ffmpeg，尝试用 winget 安装
        winget install --id Gyan.FFmpeg -e --source winget --accept-package-agreements --accept-source-agreements
        if !errorlevel! neq 0 echo [注意] ffmpeg 自动安装失败，请手动安装：winget install Gyan.FFmpeg
    )
)

:: ---------- 3. Python 环境 ----------
echo [信息] 创建 Python 3.12 环境并安装依赖（第一次会比较久）
uv sync --python 3.12
if !errorlevel! neq 0 (
    echo [错误] 依赖安装失败，请检查网络后重试
    pause
    exit /b 1
)

:: ---------- 4. GPU 版 ----------
if "%MODE%"=="gpu" (
    call :install_gpu
    if !errorlevel! neq 0 (
        echo [注意] GPU 版安装失败，恢复为 CPU 版（软件照样能用，只是识别慢一些）
        uv pip uninstall --python "%PY%" sherpa-onnx
        uv sync --python 3.12 --reinstall-package sherpa-onnx --reinstall-package sherpa-onnx-core
        > "%ROOT%\.venv\.vc_variant" echo cpu
    ) else (
        echo [完成] GPU 版已安装
    )
) else (
    > "%ROOT%\.venv\.vc_variant" echo cpu
)

:: ---------- 5. 自检 ----------
echo.
echo [信息] 环境自检：
"%ROOT%\.venv\Scripts\videocaptioner.exe" doctor
echo.
echo [完成] 安装完成。双击 run_gui.bat 启动界面，命令行用 vc.bat
echo        第一次识别时会自动下载模型（SenseVoice 约 160MB），也可以在 设置 - 模型管理 里提前下载。
echo [注意] 以后不要直接运行 uv sync / uv run，那会把 GPU 版换回 CPU 版；要更新请重新运行本脚本。
pause
exit /b 0

:install_gpu
set "WHEEL="
if defined VC_CUDA_WHEEL (
    if not exist "%VC_CUDA_WHEEL%" (
        echo [错误] VC_CUDA_WHEEL 指向的文件不存在：%VC_CUDA_WHEEL%
        exit /b 1
    )
    set "WHEEL=%VC_CUDA_WHEEL%"
    goto :have_wheel
)
if not exist "%ROOT%\.cache" mkdir "%ROOT%\.cache"
set "WHEEL=%ROOT%\.cache\%CUDA_WHEEL_NAME%"
if exist "!WHEEL!" goto :have_wheel
for %%H in (huggingface.co hf-mirror.com) do (
    if not exist "!WHEEL!" (
        echo [信息] 下载 CUDA 版 sherpa-onnx：https://%%H/%CUDA_WHEEL_PATH%
        curl.exe -fL --retry 2 --connect-timeout 20 -o "!WHEEL!.part" "https://%%H/%CUDA_WHEEL_PATH%"
        if !errorlevel! equ 0 (
            move /y "!WHEEL!.part" "!WHEEL!" >nul
        ) else (
            echo [注意] 这个地址下载失败，换下一个
        )
    )
)
if not exist "!WHEEL!" (
    echo [错误] CUDA 版 wheel 下载失败
    exit /b 1
)

:have_wheel
echo [信息] 安装 CUDA 12 / cuDNN 9 运行库（pip 包，约 2-3GB，不需要系统装 CUDA Toolkit）
uv pip install --python "%PY%" nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 "nvidia-cudnn-cu12>=9,<10"
if !errorlevel! neq 0 exit /b 1
echo [信息] 替换为 CUDA 版 sherpa-onnx
uv pip uninstall --python "%PY%" sherpa-onnx sherpa-onnx-core
uv pip install --python "%PY%" --no-deps "!WHEEL!"
if !errorlevel! neq 0 exit /b 1
"%PY%" -c "import sherpa_onnx"
if !errorlevel! neq 0 exit /b 1
> "%ROOT%\.venv\.vc_variant" echo gpu
exit /b 0
