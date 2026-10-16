"""本地识别引擎：ffmpeg 解码 → Silero VAD 切段 → SenseVoice / Fun-ASR-Nano 识别。

整段音频不会一次性读进内存：ffmpeg 边解码边喂给 VAD，只保留最近一段音频，
所以几个小时的直播录像也能跑。
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from videocaptioner.core.utils.logger import setup_logger

from . import models as M
from .align import Unit, align_text, fix_units_itn, split_text, tokens_to_units
from .device import DeviceDecision, default_num_threads, preload_cuda_libs, resolve_device

logger = setup_logger("local_asr_engine")

SAMPLE_RATE = 16000
SV_LANGS = {"zh", "en", "ja", "ko", "yue"}

ProgressFn = Callable[[int, str], None]


class TranscribeCancelled(Exception):
    pass


@dataclass
class EngineOptions:
    model: str = "sensevoice"  # sensevoice | funasr-nano
    device: str = "auto"  # auto | cuda | cpu
    language: str = ""  # "" 自动；zh/en/...
    num_threads: int = 0
    hotwords: str = ""  # 逗号分隔，仅 Fun-ASR-Nano 生效
    models_dir: str = ""
    download_source: str = "auto"
    custom_mirror: str = ""
    # VAD 参数
    vad_threshold: float = 0.5
    min_silence: float = 0.4
    max_speech: float = 15.0
    pad: float = 0.2


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    units: List[Unit] = field(default_factory=list)


def _import_sherpa():
    try:
        import sherpa_onnx
    except ImportError as e:
        raise RuntimeError(
            "没有安装 sherpa-onnx，本地识别不可用。请重新运行安装脚本（install_linux.sh / install_windows.bat）。"
        ) from e
    return sherpa_onnx


def audio_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            timeout=60,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


class _Recognizers:
    """缓存已加载的模型，批量处理时不必每个文件都重新加载。"""

    _lock = threading.Lock()
    _cache: Dict[tuple, object] = {}

    @classmethod
    def get(cls, key: tuple, factory):
        with cls._lock:
            if key not in cls._cache:
                # 显存/内存有限：换模型时丢掉旧的
                if len(cls._cache) >= 3:
                    cls._cache.clear()
                cls._cache[key] = factory()
            return cls._cache[key]

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._cache.clear()


class LocalEngine:
    def __init__(self, opts: EngineOptions):
        self.opts = opts
        self.root = M.get_models_root(opts.models_dir)
        self.decision: Optional[DeviceDecision] = None
        self.vad_spec = M.MODEL_SPECS["silero-vad"]
        self.sv_spec: Optional[M.ModelSpec] = None
        self.nano_spec: Optional[M.ModelSpec] = None

    # ---------------- 准备：选设备、下载模型、加载模型 ----------------
    def prepare(self, progress: ProgressFn, stop_event: Optional[threading.Event] = None):
        o = self.opts
        # VAD 很小，先下载，用来做 GPU 探测
        M.ensure_models([self.vad_spec], self.root, progress, o.download_source, o.custom_mirror, stop_event)
        vad_path = M.model_path(self.vad_spec, "model", self.root)
        self.decision = resolve_device(o.device, vad_path)
        logger.info("设备：%s（%s）", self.decision.device, self.decision.reason)
        progress(0, self.decision.reason)

        dev = self.decision.device
        self.sv_spec = M.pick_variant("sensevoice", dev, self.root)
        needed = [self.sv_spec]
        if o.model == "funasr-nano":
            self.nano_spec = M.pick_variant("funasr-nano", dev, self.root)
            needed.append(self.nano_spec)
        M.ensure_models(needed, self.root, progress, o.download_source, o.custom_mirror, stop_event)

        progress(0, "加载模型…")
        t = time.time()
        try:
            self._load(dev)
        except Exception as e:
            if dev != "cuda":
                raise
            logger.warning("GPU 加载模型失败，回退到 CPU：%s", e)
            self.decision = DeviceDecision("cpu", f"GPU 加载模型失败（{e}），已回退到 CPU")
            progress(0, self.decision.reason)
            self._load("cpu")
        logger.info("模型加载完成，用时 %.1fs", time.time() - t)

    def _load(self, dev: str):
        sherpa = _import_sherpa()
        if dev == "cuda":
            preload_cuda_libs()
        threads = self.opts.num_threads or default_num_threads()
        # SenseVoice 的自动语种检测会把中英夹杂的直播误判成日语（「礼拜二」→「礼バ二」）。
        # 用 zh 模式英文照样能识别，所以「自动」按中文处理；纯日/韩/粤语请明确选语言。
        lang = self.opts.language if self.opts.language in SV_LANGS else "zh"
        assert self.sv_spec is not None
        sv = self.sv_spec
        sv_key = ("sv", sv.key, dev, threads, lang)
        self.sv = _Recognizers.get(
            sv_key,
            lambda: sherpa.OfflineRecognizer.from_sense_voice(
                model=str(M.model_path(sv, "model", self.root)),
                tokens=str(M.model_path(sv, "tokens", self.root)),
                num_threads=threads,
                use_itn=True,
                language=lang,
                provider=dev,
            ),
        )
        self.nano = None
        if self.nano_spec is not None:
            nano = self.nano_spec
            hot = ",".join(w.strip() for w in self.opts.hotwords.replace("，", ",").split(",") if w.strip())
            nano_key = ("nano", nano.key, dev, threads, hot)
            self.nano = _Recognizers.get(
                nano_key,
                lambda: sherpa.OfflineRecognizer.from_funasr_nano(
                    encoder_adaptor=str(M.model_path(nano, "encoder_adaptor", self.root)),
                    llm=str(M.model_path(nano, "llm", self.root)),
                    embedding=str(M.model_path(nano, "embedding", self.root)),
                    tokenizer=str(M.model_path(nano, "tokenizer", self.root)),
                    num_threads=threads,
                    provider=dev,
                    hotwords=hot,
                ),
            )
        self._sherpa = sherpa

    def _make_vad(self):
        sherpa = self._sherpa
        cfg = sherpa.VadModelConfig()
        cfg.silero_vad.model = str(M.model_path(self.vad_spec, "model", self.root))
        cfg.silero_vad.threshold = self.opts.vad_threshold
        cfg.silero_vad.min_silence_duration = self.opts.min_silence
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = self.opts.max_speech
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = 1
        cfg.provider = "cpu"  # VAD 很轻，放 CPU 更稳
        return sherpa.VoiceActivityDetector(cfg, buffer_size_in_seconds=self.opts.max_speech * 4 + 30)

    # ---------------- 识别 ----------------
    def transcribe(
        self,
        audio_path: str,
        progress: Optional[ProgressFn] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> List[Utterance]:
        progress = progress or (lambda p, m: None)
        if self.decision is None:
            self.prepare(progress, stop_event)
        total = audio_duration(audio_path)
        vad = self._make_vad()
        ring = _Ring(seconds=self.opts.max_speech * 4 + 60)
        pending: List[Tuple[float, float, np.ndarray]] = []
        results: List[Utterance] = []
        batch = 8 if self.nano is None else 1
        started = time.time()
        dev_name = "GPU" if self.decision and self.decision.device == "cuda" else "CPU"

        def run_pending(force: bool = False):
            while pending and (force or len(pending) >= batch):
                chunk = pending[:batch]
                del pending[:batch]
                results.extend(self._decode(chunk))
                done_s = chunk[-1][1]
                if total > 0:
                    pct = int(min(done_s / total, 1.0) * 100)
                    speed = done_s / max(time.time() - started, 1e-6)
                    progress(min(pct, 99), f"本地识别中（{dev_name}）{pct}%，{speed:.0f}x 实时")

        def take_segments():
            while not vad.empty():
                seg = vad.front
                s0 = seg.start / SAMPLE_RATE
                s1 = s0 + len(seg.samples) / SAMPLE_RATE
                vad.pop()
                a = max(0.0, s0 - self.opts.pad)
                b = s1 + self.opts.pad
                samples = ring.slice(a, b)
                if samples is None or len(samples) == 0:
                    samples = np.asarray(seg.samples, dtype=np.float32)
                    a = s0
                pending.append((a, a + len(samples) / SAMPLE_RATE, samples))

        proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", audio_path, "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        assert proc.stdout is not None
        window = 512  # silero 的窗口大小
        leftover = np.zeros(0, dtype=np.float32)
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    raise TranscribeCancelled()
                raw = proc.stdout.read(SAMPLE_RATE * 2)  # 1 秒
                if not raw:
                    break
                x = np.frombuffer(raw[: len(raw) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
                ring.push(x)
                buf = np.concatenate([leftover, x])
                n = len(buf) // window * window
                for i in range(0, n, window):
                    vad.accept_waveform(buf[i : i + window])
                leftover = buf[n:]
                take_segments()
                run_pending()
            if len(leftover):
                vad.accept_waveform(leftover)
            vad.flush()
            take_segments()
            run_pending(force=True)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        if proc.returncode not in (0, None, -9) and not results:
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            raise RuntimeError(f"ffmpeg 解码音频失败：{err[-300:]}")
        elapsed = time.time() - started
        logger.info(
            "识别完成：%d 段，音频 %.0fs，用时 %.0fs（%s）", len(results), total, elapsed, dev_name
        )
        progress(100, f"本地识别完成，用时 {elapsed:.0f} 秒")
        return results

    def _decode(self, chunk: List[Tuple[float, float, np.ndarray]]) -> List[Utterance]:
        streams = []
        for _, _, samples in chunk:
            s = self.sv.create_stream()
            s.accept_waveform(SAMPLE_RATE, samples)
            streams.append(s)
        self.sv.decode_streams(streams)
        out = []
        for (t0, t1, samples), s in zip(chunk, streams):
            r = s.result
            dur = t1 - t0
            ref = tokens_to_units(list(r.tokens), list(r.timestamps), dur)
            ref = fix_units_itn(ref)
            if self.nano is not None:
                ns = self.nano.create_stream()
                ns.accept_waveform(SAMPLE_RATE, samples)
                self.nano.decode_stream(ns)
                text = ns.result.text.strip()
                units = align_text(text, ref, 0.0, dur) if text else []
            else:
                units = ref
            if not any(not u.is_punct for u in units):
                continue
            for u in units:
                u.start += t0
                u.end += t0
            text = "".join(u.text for u in units)
            out.append(Utterance(units[0].start, max(u.end for u in units), text, units))
        return out


class _Ring:
    """保存最近 N 秒音频，用来给 VAD 切出的段落前后补一点余量。"""

    def __init__(self, seconds: float):
        self.cap = int(seconds * SAMPLE_RATE)
        self.blocks: deque = deque()
        self.size = 0
        self.end = 0  # 已推入的总样本数

    def push(self, x: np.ndarray):
        self.blocks.append(x)
        self.size += len(x)
        self.end += len(x)
        while self.size - len(self.blocks[0]) >= self.cap:
            self.size -= len(self.blocks.popleft())

    def slice(self, a: float, b: float) -> Optional[np.ndarray]:
        start_avail = self.end - self.size
        ia = max(int(a * SAMPLE_RATE), start_avail)
        ib = min(int(b * SAMPLE_RATE), self.end)
        if ib <= ia:
            return None
        data = np.concatenate(list(self.blocks))
        return data[ia - start_avail : ib - start_avail].copy()


def plain_units(text: str, start: float, end: float) -> List[Unit]:
    from .align import _spread

    return _spread(split_text(text), start, end)
