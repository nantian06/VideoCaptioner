"""N 卡检测、CUDA 运行库预加载与 GPU/CPU 自动回退。

sherpa-onnx 的 CUDA 版 wheel 自带 onnxruntime，但需要 CUDA 12 + cuDNN 9 运行库。
运行库可以来自系统安装，也可以来自 pip 包 nvidia-*-cu12（安装脚本会装），
后者不在系统库路径里，所以这里在加载 sherpa-onnx 之前手动预加载。

是否真的能用 GPU，用一个子进程试建一个 CUDA 会话来判断：
native 层缺库时可能直接崩溃，放在子进程里就不会拖垮主程序。
"""

from __future__ import annotations

import ctypes
import glob
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from videocaptioner.core.utils.logger import setup_logger

logger = setup_logger("local_asr_device")

IS_WINDOWS = platform.system() == "Windows"

# 按依赖顺序预加载
_CUDA_LIB_ORDER = [
    "cuda_runtime",
    "cuda_nvrtc",
    "nvjitlink",
    "cublas",
    "cufft",
    "curand",
    "cudnn",
]


@dataclass
class DeviceDecision:
    device: str  # "cuda" | "cpu"
    reason: str
    gpu_name: str = ""


_preloaded = False
_probe_cache: dict = {}


def sherpa_version() -> str:
    try:
        return importlib.metadata.version("sherpa-onnx")
    except importlib.metadata.PackageNotFoundError:
        return ""


def sherpa_has_cuda() -> bool:
    """装的是不是 sherpa-onnx 的 CUDA 版。"""
    if "cuda" in sherpa_version():
        return True
    try:
        import sherpa_onnx

        lib_dir = Path(sherpa_onnx.__file__).parent / "lib"
        pattern = "*onnxruntime_providers_cuda*"
        return any(lib_dir.glob(pattern))
    except Exception:
        return False


def nvidia_gpus() -> List[str]:
    """用 nvidia-smi 列出 N 卡（名称 + 显存）。没有驱动时返回空列表。"""
    smi = shutil.which("nvidia-smi")
    if not smi and IS_WINDOWS:
        candidate = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(candidate) if candidate.exists() else None
    if not smi:
        return []
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
        if out.returncode != 0:
            return []
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _nvidia_pip_dirs() -> List[Path]:
    """pip 安装的 nvidia-*-cu12 包里的库目录。"""
    roots = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    roots.update(p for p in sys.path if p.endswith("site-packages"))
    dirs = []
    sub = "bin" if IS_WINDOWS else "lib"
    for root in roots:
        for name in _CUDA_LIB_ORDER:
            for d in glob.glob(os.path.join(root, "nvidia", name, sub)):
                dirs.append(Path(d))
    return dirs


def preload_cuda_libs() -> None:
    """把 pip 版 CUDA/cuDNN 库加入搜索路径。重复调用无副作用。"""
    global _preloaded
    if _preloaded:
        return
    _preloaded = True
    dirs = _nvidia_pip_dirs()
    if not dirs:
        return
    if IS_WINDOWS:
        for d in dirs:
            try:
                os.add_dll_directory(str(d))  # type: ignore[attr-defined]
            except OSError:
                pass
        os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs) + os.pathsep + os.environ.get("PATH", "")
        return
    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
    for d in dirs:
        for lib in sorted(d.glob("lib*.so*")):
            try:
                ctypes.CDLL(str(lib), mode=mode)
            except OSError:
                pass  # 有些库依赖别的库，后面会被间接加载
    logger.info("已预加载 pip 版 CUDA 运行库：%s", ", ".join(str(d) for d in dirs))


_PROBE_CODE = r"""
import sys
from videocaptioner.core.asr.local.device import preload_cuda_libs
preload_cuda_libs()
import numpy as np, sherpa_onnx
cfg = sherpa_onnx.VadModelConfig()
cfg.silero_vad.model = sys.argv[1]
cfg.sample_rate = 16000
cfg.provider = "cuda"
vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=5)
vad.accept_waveform(np.zeros(16000, dtype=np.float32))
print("PROBE_OK")
"""


def probe_cuda(vad_model: Path, timeout: int = 90) -> tuple:
    """在子进程里试建 CUDA 会话。返回 (是否可用, 说明)。"""
    key = str(vad_model)
    if key in _probe_cache:
        return _probe_cache[key]
    if getattr(sys, "frozen", False):
        result = (True, "打包版跳过子进程探测")
        _probe_cache[key] = result
        return result
    try:
        out = subprocess.run(
            [sys.executable, "-c", _PROBE_CODE, str(vad_model)],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0,
        )
        log = (out.stdout or "") + (out.stderr or "")
        if out.returncode == 0 and "PROBE_OK" in log and "Fallback to cpu" not in log:
            result = (True, "CUDA 会话创建成功")
        else:
            tail = " ".join(log.strip().splitlines()[-3:])[:300]
            result = (False, f"CUDA 不可用（{tail or '子进程退出码 %s' % out.returncode}）")
    except subprocess.TimeoutExpired:
        result = (False, "CUDA 探测超时")
    except Exception as e:  # pragma: no cover
        result = (False, f"CUDA 探测出错：{e}")
    _probe_cache[key] = result
    return result


def resolve_device(requested: str, vad_model: Optional[Path]) -> DeviceDecision:
    """把用户选择的 auto/cuda/cpu 变成实际设备，必要时回退到 CPU。"""
    requested = (requested or "auto").lower()
    if requested == "cpu":
        return DeviceDecision("cpu", "已手动选择 CPU")

    gpus = nvidia_gpus()
    if not gpus:
        return DeviceDecision("cpu", "未检测到 N 卡驱动（nvidia-smi），使用 CPU")
    if not sherpa_has_cuda():
        return DeviceDecision(
            "cpu",
            f"检测到 {gpus[0]}，但装的是 CPU 版 sherpa-onnx，使用 CPU。"
            "运行安装脚本的 GPU 模式可启用加速",
        )
    preload_cuda_libs()
    if vad_model is not None and vad_model.exists():
        ok, msg = probe_cuda(vad_model)
        if not ok:
            return DeviceDecision("cpu", f"{msg}，已回退到 CPU")
    return DeviceDecision("cuda", f"使用 GPU：{gpus[0]}", gpu_name=gpus[0])


def default_num_threads() -> int:
    n = os.cpu_count() or 4
    return max(1, min(8, n - 1 if n > 2 else n))
