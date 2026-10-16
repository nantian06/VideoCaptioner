"""本地 ASR 模型注册表与下载器。

模型全部是 sherpa-onnx 官方转换好的 ONNX 文件，发布在
https://github.com/k2-fsa/sherpa-onnx/releases 。国内直连 GitHub 慢时，
可以走 GitHub 加速镜像，或手动下载后解压到模型目录。
"""

from __future__ import annotations

import os
import shutil
import tarfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

from videocaptioner.core.utils.logger import setup_logger

logger = setup_logger("local_asr_models")

GITHUB_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"

# 下载源：key -> (显示名, URL 前缀)。前缀直接拼在 GitHub 完整地址前面。
DOWNLOAD_SOURCES: Dict[str, tuple] = {
    "github": ("GitHub 直连", ""),
    "ghfast": ("国内加速 ghfast.top", "https://ghfast.top/"),
    "ghproxy": ("国内加速 gh-proxy.com", "https://gh-proxy.com/"),
}
# "auto" 时依次尝试的顺序：先国内加速，再直连
AUTO_SOURCE_ORDER = ["ghfast", "ghproxy", "github"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str  # vad | sensevoice | funasr-nano
    title: str
    asset: str  # release 资源路径，相对 GITHUB_RELEASE
    dirname: str  # 解压后的目录名；单文件模型则为文件名
    files: Dict[str, str] = field(default_factory=dict)  # 角色 -> 相对路径
    size_mb: int = 0
    note: str = ""

    @property
    def is_archive(self) -> bool:
        return self.asset.endswith(".tar.bz2")


MODEL_SPECS: Dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="silero-vad",
            family="vad",
            title="Silero VAD（语音活动检测）",
            asset="asr-models/silero_vad.onnx",
            dirname="silero_vad.onnx",
            files={"model": ""},
            size_mb=1,
        ),
        ModelSpec(
            key="sensevoice-int8",
            family="sensevoice",
            title="SenseVoice-Small int8（CPU 推荐）",
            asset="asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
            dirname="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            files={"model": "model.int8.onnx", "tokens": "tokens.txt"},
            size_mb=160,
        ),
        ModelSpec(
            key="sensevoice-fp32",
            family="sensevoice",
            title="SenseVoice-Small fp32（GPU 推荐）",
            asset="asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            dirname="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            files={"model": "model.onnx", "tokens": "tokens.txt"},
            size_mb=1000,
        ),
        ModelSpec(
            key="funasr-nano-int8",
            family="funasr-nano",
            title="Fun-ASR-Nano int8（CPU 可用，较慢）",
            asset="asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2",
            dirname="sherpa-onnx-funasr-nano-int8-2025-12-30",
            files={
                "encoder_adaptor": "encoder_adaptor.int8.onnx",
                "llm": "llm.int8.onnx",
                "embedding": "embedding.int8.onnx",
                "tokenizer": "Qwen3-0.6B",
            },
            size_mb=810,
        ),
        ModelSpec(
            key="funasr-nano-fp16",
            family="funasr-nano",
            title="Fun-ASR-Nano fp16（GPU 推荐）",
            asset="asr-models/sherpa-onnx-funasr-nano-fp16-2025-12-30.tar.bz2",
            dirname="sherpa-onnx-funasr-nano-fp16-2025-12-30",
            files={
                "encoder_adaptor": "encoder_adaptor.int8.onnx",
                "llm": "llm.fp16.onnx",
                "embedding": "embedding.int8.onnx",
                "tokenizer": "Qwen3-0.6B",
            },
            size_mb=1030,
        ),
    ]
}


def default_models_root() -> Path:
    from videocaptioner.config import MODEL_PATH

    return Path(MODEL_PATH) / "local_asr"


def get_models_root(custom: Optional[str] = None) -> Path:
    root = Path(custom).expanduser() if custom else default_models_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_path(spec: ModelSpec, role: str, root: Path) -> Path:
    base = root / spec.dirname
    rel = spec.files.get(role, "")
    return base / rel if rel else base


def is_installed(spec: ModelSpec, root: Path) -> bool:
    return all(model_path(spec, role, root).exists() for role in spec.files)


def pick_variant(family: str, device: str, root: Path) -> ModelSpec:
    """按设备挑选模型精度：GPU 用 fp32/fp16，CPU 用 int8。

    如果首选精度没下载、但另一种精度已经在本地，就用本地那份，避免重复下载。
    """
    if family == "vad":
        return MODEL_SPECS["silero-vad"]
    cpu_key = f"{family}-int8"
    gpu_key = "sensevoice-fp32" if family == "sensevoice" else "funasr-nano-fp16"
    preferred, other = (gpu_key, cpu_key) if device == "cuda" else (cpu_key, gpu_key)
    if not is_installed(MODEL_SPECS[preferred], root) and is_installed(
        MODEL_SPECS[other], root
    ):
        return MODEL_SPECS[other]
    return MODEL_SPECS[preferred]


def missing_specs(specs: List[ModelSpec], root: Path) -> List[ModelSpec]:
    return [s for s in specs if not is_installed(s, root)]


class DownloadCancelled(Exception):
    pass


ProgressFn = Callable[[int, str], None]


def _source_urls(spec: ModelSpec, source: str, custom_prefix: str = "") -> List[tuple]:
    url = GITHUB_RELEASE + spec.asset
    if source == "custom" and custom_prefix:
        prefix = custom_prefix if custom_prefix.endswith("/") else custom_prefix + "/"
        return [("自定义镜像", prefix + url), ("GitHub 直连", url)]
    order = AUTO_SOURCE_ORDER if source in ("", "auto") else [source]
    return [(DOWNLOAD_SOURCES[k][0], DOWNLOAD_SOURCES[k][1] + url) for k in order]


def _download_file(
    url: str,
    dest: Path,
    progress: ProgressFn,
    label: str,
    stop_event: Optional[threading.Event],
    min_speed: int = 30 * 1024,
    slow_grace: float = 25.0,
) -> None:
    """断点续传下载。持续过慢会抛出 TimeoutError，让调用方换下一个源。"""
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 30)) as resp:
        if resp.status_code == 416:  # 已经下载完整
            part.rename(dest)
            return
        resp.raise_for_status()
        if done and resp.status_code != 206:
            done = 0  # 服务器不支持续传，从头开始
        total = int(resp.headers.get("Content-Length", 0)) + done
        mode = "ab" if done else "wb"
        window_start, window_bytes = time.time(), 0
        last_report = 0.0
        with open(part, mode) as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if stop_event is not None and stop_event.is_set():
                    raise DownloadCancelled()
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                window_bytes += len(chunk)
                now = time.time()
                if now - last_report > 0.5:
                    last_report = now
                    pct = int(done * 100 / total) if total else 0
                    speed = window_bytes / max(now - window_start, 1e-6)
                    progress(
                        min(pct, 99),
                        f"下载 {label}：{done / 1e6:.0f}/{total / 1e6:.0f} MB，"
                        f"{speed / 1e6:.1f} MB/s",
                    )
                if now - window_start > slow_grace:
                    if window_bytes / (now - window_start) < min_speed:
                        raise TimeoutError("下载速度过慢")
                    window_start, window_bytes = now, 0
    if total and done < total:
        raise IOError(f"下载不完整：{done}/{total}")
    part.rename(dest)


def _extract(archive: Path, spec: ModelSpec, root: Path) -> None:
    """只解压需要的文件（跳过 test_wavs 和用不到的另一精度模型）。"""
    wanted = {f"{spec.dirname}/{rel}" for rel in spec.files.values()}
    tmp_dir = root / (spec.dirname + ".extracting")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    with tarfile.open(archive, "r:bz2") as tar:
        members = [
            m
            for m in tar.getmembers()
            if any(m.name == w or m.name.startswith(w + "/") for w in wanted)
            or m.name.endswith(("LICENSE", "README.md"))
        ]
        for m in members:
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise IOError(f"压缩包里有不安全的路径：{m.name}")
        tar.extractall(tmp_dir, members=members)
    final = root / spec.dirname
    shutil.rmtree(final, ignore_errors=True)
    (tmp_dir / spec.dirname).rename(final)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_model(
    spec: ModelSpec,
    root: Path,
    progress: Optional[ProgressFn] = None,
    source: str = "auto",
    custom_prefix: str = "",
    stop_event: Optional[threading.Event] = None,
) -> Path:
    """下载并解压一个模型，返回模型所在路径。已存在则直接返回。"""
    progress = progress or (lambda p, m: None)
    if is_installed(spec, root):
        return root / spec.dirname
    root.mkdir(parents=True, exist_ok=True)
    filename = Path(spec.asset).name
    dest = root / filename if spec.is_archive else root / spec.dirname
    errors = []
    for name, url in _source_urls(spec, source, custom_prefix):
        logger.info("下载模型 %s，来源：%s %s", spec.key, name, url)
        progress(0, f"连接 {name} 下载 {spec.title}…")
        try:
            _download_file(url, dest, progress, spec.title, stop_event)
            break
        except DownloadCancelled:
            raise
        except Exception as e:  # 换下一个源（已下载的部分会续传）
            logger.warning("从 %s 下载失败：%s", name, e)
            errors.append(f"{name}: {e}")
    else:
        raise RuntimeError(
            "模型下载失败，可以换一个下载源，或按 README 手动下载后放到 "
            f"{root}。\n" + "\n".join(errors)
        )
    if spec.is_archive:
        progress(99, f"解压 {spec.title}…")
        _extract(dest, spec, root)
        dest.unlink(missing_ok=True)
    progress(100, f"{spec.title} 已就绪")
    return root / spec.dirname


def ensure_models(
    specs: List[ModelSpec],
    root: Path,
    progress: Optional[ProgressFn] = None,
    source: str = "auto",
    custom_prefix: str = "",
    stop_event: Optional[threading.Event] = None,
) -> None:
    for spec in missing_specs(specs, root):
        download_model(spec, root, progress, source, custom_prefix, stop_event)


def disk_usage_mb(root: Path) -> int:
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            try:
                total += (Path(dirpath) / f).stat().st_size
            except OSError:
                pass
    return total // (1024 * 1024)
