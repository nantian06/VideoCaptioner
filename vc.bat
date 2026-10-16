@echo off
chcp 65001 >nul
:: 命令行入口，例如：vc.bat transcribe 直播.mp4 --local-model funasr-nano
if not exist "%~dp0.venv\Scripts\videocaptioner.exe" (
    echo 还没安装，请先双击 install_windows.bat
    exit /b 1
)
"%~dp0.venv\Scripts\videocaptioner.exe" %*
