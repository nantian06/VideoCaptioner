"""本地中文 ASR（sherpa-onnx：SenseVoice / Fun-ASR-Nano），接入 VideoCaptioner 的 ASR 体系。"""

import hashlib
import os
import threading
from typing import Callable, List, Optional

from ..utils.logger import setup_logger
from .asr_data import ASRDataSeg
from .base import BaseASR
from .local.align import Unit, build_sentences, join_units
from .local.engine import EngineOptions, LocalEngine, audio_duration

logger = setup_logger("local_asr")

_SAMPLE_BYTES = 4 * 1024 * 1024


class LocalASR(BaseASR):
    """离线识别。不上传任何音频；模型首次使用时自动下载。"""

    SUPPORTED_SOUND_FORMAT = ["flac", "m4a", "mp3", "wav", "mp4", "mkv", "flv", "ts", "aac", "ogg", "opus", "webm", "mov"]

    def __init__(
        self,
        audio_input: str,
        use_cache: bool = True,
        need_word_time_stamp: bool = False,
        model: str = "sensevoice",
        device: str = "auto",
        language: str = "",
        num_threads: int = 0,
        hotwords: str = "",
        max_chars: int = 25,
        models_dir: str = "",
        download_source: str = "auto",
        custom_mirror: str = "",
        vad_threshold: float = 0.5,
        stop_event: Optional[threading.Event] = None,
    ):
        self.need_word_time_stamp = need_word_time_stamp
        self.max_chars = max_chars
        self.stop_event = stop_event
        self.opts = EngineOptions(
            model=model,
            device=device,
            language=language,
            num_threads=num_threads,
            hotwords=hotwords,
            models_dir=models_dir,
            download_source=download_source,
            custom_mirror=custom_mirror,
            vad_threshold=vad_threshold,
        )
        super().__init__(audio_input, use_cache, need_word_time_stamp)

    # 长录音不整段读进内存：用文件大小 + mtime + 头中尾采样做缓存 key。
    # mtime 让「同大小、采样区未被改写」的原地修改（流复制剪切、元数据注入）
    # 不再命中旧缓存；中段字节级改写且 mtime 不变的极端场景由采样兜底
    def _set_data(self):
        path = self.audio_input
        if not isinstance(path, str) or not os.path.exists(path):
            raise ValueError(f"音频文件不存在：{path}")
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext and ext not in self.SUPPORTED_SOUND_FORMAT:
            # 不硬拒绝：ffmpeg 支持的格式比登记表宽；但要让用户知道后续报错的可能来源
            logger.warning("扩展名 .%s 不在已验证列表里，若解码失败请先转成 wav/mp3", ext)
        size = os.path.getsize(path)
        h = hashlib.md5(f"{size}:{os.stat(path).st_mtime_ns}".encode())
        with open(path, "rb") as f:
            h.update(f.read(_SAMPLE_BYTES))
            if size > _SAMPLE_BYTES * 2:
                f.seek(size // 2)
                h.update(f.read(_SAMPLE_BYTES))
                f.seek(-_SAMPLE_BYTES, os.SEEK_END)
                h.update(f.read(_SAMPLE_BYTES))
        self.file_binary = None
        self.crc32_hex = h.hexdigest()[:16]

    def _get_audio_duration(self) -> float:
        duration = audio_duration(str(self.audio_input))
        if not duration:
            # ffprobe 不在 PATH 时进度条会退化成恒 0%/99%，至少留痕
            logger.warning("无法获取音频时长（ffprobe 不可用？），进度按 60s 估算")
            return 60.0
        return duration

    def _get_key(self) -> str:
        o = self.opts
        return f"{self.crc32_hex}-{o.model}-{o.language}-{o.hotwords}-{o.vad_threshold}-v2"

    def _run(self, callback: Optional[Callable[[int, str], None]] = None, **kwargs) -> dict:
        callback = callback or (lambda p, m: None)
        engine = LocalEngine(self.opts)
        engine.prepare(callback, self.stop_event)
        utts = engine.transcribe(str(self.audio_input), callback, self.stop_event)
        decision = engine.decision
        return {
            "device": decision.device if decision else "cpu",
            "device_reason": decision.reason if decision else "",
            "utterances": [
                {
                    "start": u.start,
                    "end": u.end,
                    "units": [[x.text, round(x.start, 3), round(x.end, 3), x.kind] for x in u.units],
                }
                for u in utts
            ],
        }

    def _make_segments(self, resp_data: dict) -> List[ASRDataSeg]:
        units = [
            Unit(t, s, e, k)
            for utt in resp_data.get("utterances", [])
            for t, s, e, k in utt["units"]
        ]
        if self.need_word_time_stamp:
            # 字/词级：交给后续断句（LLM 或规则）处理
            return [
                ASRDataSeg(u.text, int(u.start * 1000), int(max(u.end, u.start + 0.05) * 1000))
                for u in units
                if not u.is_punct
            ]
        segs = []
        for line in build_sentences(units, max_len=self.max_chars):
            words = [u for u in line if not u.is_punct]
            if not words:
                continue
            segs.append(
                ASRDataSeg(join_units(line), int(words[0].start * 1000), int(words[-1].end * 1000))
            )
        return segs
