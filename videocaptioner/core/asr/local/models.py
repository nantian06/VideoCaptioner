"""本地 ASR 模型注册表与下载器。

模型全部是 sherpa-onnx 官方转换好的 ONNX 文件，发布在
https://github.com/k2-fsa/sherpa-onnx/releases 。国内直连 GitHub 慢时，
可以走 GitHub 加速镜像，或手动下载后解压到模型目录。
"""

from __future__ import annotations

import hashlib
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
    sha256: str = ""  # 下载资源（asset）的 SHA256；空表示未登记，跳过校验
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
            sha256="9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
        ),
        ModelSpec(
            key="sensevoice-int8",
            family="sensevoice",
            title="SenseVoice-Small int8（CPU 推荐）",
            asset="asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
            dirname="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
            files={"model": "model.int8.onnx", "tokens": "tokens.txt"},
            size_mb=160,
            sha256="7d1efa2138a65b0b488df37f8b89e3d91a60676e416f515b952358d83dfd347e",
        ),
        ModelSpec(
            key="sensevoice-fp32",
            family="sensevoice",
            title="SenseVoice-Small fp32（GPU 推荐）",
            asset="asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2",
            dirname="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            files={"model": "model.onnx", "tokens": "tokens.txt"},
            size_mb=1000,
            sha256="f6b2a72ebcb1ac7a764d4cfccd886e6bcb2a95c4657c2199d0ba95ed4b9ea71a",
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
            sha256="eb43d7ccc2e86b243f6a03b7df361033dda66db9523d1a92bf6aca2b50c9476b",
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
            sha256="a07a996361aa2f8b2c4f47861fe01953b5509664efa3392b734580b1eeb362e3",
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
    """所有角色都就位：文件非空、目录非空，断链等异常状态视为未安装。

    注意角色可能是目录（如 Fun-ASR-Nano 的 tokenizer 是 Qwen3-0.6B 目录），
    Windows 上目录 st_size 恒为 0，不能用字节数判断，必须分开处理。
    """
    for role in spec.files:
        p = model_path(spec, role, root)
        try:
            if p.is_dir():
                if next(p.iterdir(), None) is None:
                    return False  # 空目录多半是解压中断的残留
            elif not p.exists() or p.stat().st_size == 0:
                return False  # 空文件多半是坏下载残留
        except OSError:
            return False
    return True


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


class HashMismatchError(IOError):
    """下载文件的 SHA256 与登记值不符：内容被篡改或损坏。"""


def _verify_hash(dest: Path, spec: ModelSpec) -> None:
    """校验下载产物的 SHA256；未登记哈希的 spec 直接跳过。"""
    if not spec.sha256:
        return
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != spec.sha256:
        raise HashMismatchError(
            f"{dest.name} SHA256 校验失败：期望 {spec.sha256[:16]}…，实际 {actual[:16]}…"
        )


ProgressFn = Callable[[int, str], None]


def _source_urls(spec: ModelSpec, source: str, custom_prefix: str = "") -> List[tuple]:
    url = GITHUB_RELEASE + spec.asset
    if source == "custom" and custom_prefix:
        prefix = custom_prefix if custom_prefix.endswith("/") else custom_prefix + "/"
        if not prefix.lower().startswith("https://"):
            # 明文 HTTP 会被中间人替换模型文件；自定义镜像只接受 https
            raise ValueError(f"自定义镜像必须是 https:// 地址，当前是：{prefix}")
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
        if not members:
            raise IOError(f"压缩包里找不到需要的文件（{spec.title}），可能下载错了资源")
        for m in members:
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise IOError(f"压缩包里有不安全的路径：{m.name}")
            if m.issym() or m.islnk():
                # 链接成员可让后续写入跟随到压缩包外，一律拒绝
                raise IOError(f"压缩包里有链接成员，已拒绝解压：{m.name}")
        try:
            tar.extractall(tmp_dir, members=members, filter="data")
        except TypeError:  # Python < 3.10.12 没有 filter 参数；上方手动过滤已覆盖
            tar.extractall(tmp_dir, members=members)
    final = root / spec.dirname
    shutil.rmtree(final, ignore_errors=True)
    (tmp_dir / spec.dirname).rename(final)
    shutil.rmtree(tmp_dir, ignore_errors=True)


_download_locks: Dict[str, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def _spec_lock(key: str) -> threading.Lock:
    with _download_locks_guard:
        if key not in _download_locks:
            _download_locks[key] = threading.Lock()
        return _download_locks[key]


def download_model(
    spec: ModelSpec,
    root: Path,
    progress: Optional[ProgressFn] = None,
    source: str = "auto",
    custom_prefix: str = "",
    stop_event: Optional[threading.Event] = None,
) -> Path:
    """下载并解压一个模型，返回模型所在路径。已存在则直接返回。

    同一模型的并发调用在锁上排队，避免两个线程写坏同一个 .part 文件；
    排队期间持续响应 stop_event，持有方异常退出（如被强制终止）时
    等待方仍能被取消，不会钉死在锁上。
    """
    progress = progress or (lambda p, m: None)
    lock = _spec_lock(spec.key)
    while not lock.acquire(timeout=0.5):
        if stop_event is not None and stop_event.is_set():
            raise DownloadCancelled()
    try:
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
                _verify_hash(dest, spec)
                break
            except DownloadCancelled:
                raise
            except HashMismatchError as e:
                # 这个源给的内容就是坏的：清掉产物和 .part，不让坏数据续传到下一个源
                logger.error("从 %s 下载的 %s 校验失败：%s", name, dest.name, e)
                errors.append(f"{name}: {e}")
                dest.unlink(missing_ok=True)
                dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
            except Exception as e:  # 换下一个源（已下载的部分会续传，最后统一校验兜底）
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
    finally:
        lock.release()


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
