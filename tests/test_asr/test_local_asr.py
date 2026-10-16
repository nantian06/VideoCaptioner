"""本地中文识别（sherpa-onnx）的单元测试。

对齐、组句等纯逻辑不需要模型；端到端测试需要设置环境变量
VC_LOCAL_ASR_MODELS 指向已下载的模型目录，否则跳过。
"""

import os
from pathlib import Path

import pytest

from videocaptioner.core.asr.local.align import (
    Unit,
    align_text,
    build_sentences,
    fix_units_itn,
    join_units,
    split_text,
    tokens_to_units,
)


def test_split_text_mixed():
    parts = split_text("昨天是MONDAY，today is礼拜二。")
    assert [p for p, _ in parts] == ["昨", "天", "是", "MONDAY", "，", "today", "is", "礼", "拜", "二", "。"]
    assert dict(parts)["MONDAY"] == "word"
    assert dict(parts)["，"] == "punct"


def test_tokens_to_units_glues_bpe_pieces():
    tokens = ["昨", "天", "是", "mon", "day", "，", "to", "day", " is", "礼", "拜", "2"]
    starts = [0.6, 0.84, 1.14, 1.74, 2.04, 2.34, 3.72, 4.02, 4.38, 5.16, 5.46, 5.64]
    units = fix_units_itn(tokens_to_units(tokens, starts, 6.0))
    assert [u.text for u in units] == ["昨", "天", "是", "monday", "，", "today", "is", "礼", "拜", "二"]
    monday = units[3]
    assert monday.start == pytest.approx(1.74)
    assert monday.end == pytest.approx(2.34)


def test_cap_long_gaps():
    units = tokens_to_units(["好", "的"], [0.0, 5.0], 6.0)
    assert units[0].end - units[0].start <= 0.6 + 1e-6


def test_align_text_uses_reference_times():
    ref = tokens_to_units(["这", "是", "平", "凡", "的"], [0.1, 0.3, 0.5, 0.7, 0.9], 1.2)
    units = align_text("这是频繁的。", ref, 0.0, 1.2)
    assert [u.text for u in units] == ["这", "是", "频", "繁", "的", "。"]
    assert units[0].start == pytest.approx(0.1)
    # 被替换的两个字落在原来「平凡」的时间范围里
    assert units[2].start == pytest.approx(0.5)
    assert units[3].end == pytest.approx(0.9)
    # 标点不占时长
    assert units[-1].start == units[-1].end


def _units(text, step=0.2, gap_at=None, gap=1.0):
    out, t = [], 0.0
    for i, (txt, kind) in enumerate(split_text(text)):
        if gap_at is not None and i == gap_at:
            t += gap
        dur = 0.0 if kind == "punct" else step
        out.append(Unit(txt, t, t + dur, kind))
        t += dur
    return out


def test_build_sentences_breaks_on_period_and_strips_trailing_punct():
    sents = build_sentences(_units("今天天气很好。我们出去玩吧！"), max_len=25)
    assert [join_units(s) for s in sents] == ["今天天气很好", "我们出去玩吧！"]


def test_build_sentences_splits_long_text_at_comma():
    text = "这是一个非常非常长的句子，后面还跟着很多很多很多的内容需要切开才行"
    sents = build_sentences(_units(text), max_len=20)
    assert all(sum(1 for u in s if u.kind == "cjk") <= 20 for s in sents)
    assert join_units(sents[0]) == "这是一个非常非常长的句子"


def test_build_sentences_breaks_on_pause():
    sents = build_sentences(_units("你好大家好欢迎来到直播间", gap_at=5, gap=1.5), max_len=25)
    assert [join_units(s) for s in sents] == ["你好大家好", "欢迎来到直播间"]


def test_join_units_spacing():
    units = _units("I feel that.You know这个brand")
    assert join_units(units) == "I feel that. You know这个brand"


def test_local_asr_make_segments_word_and_sentence(tmp_path: Path):
    from videocaptioner.core.asr.local_asr import LocalASR

    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF0000WAVE")
    resp = {
        "utterances": [
            {
                "start": 0.0,
                "end": 1.0,
                "units": [["你", 0.0, 0.2, "cjk"], ["好", 0.2, 0.4, "cjk"], ["。", 0.4, 0.4, "punct"], ["OK", 0.5, 0.9, "word"]],
            }
        ]
    }
    asr = LocalASR.__new__(LocalASR)
    asr.need_word_time_stamp = True
    asr.max_chars = 25
    words = asr._make_segments(resp)
    assert [w.text for w in words] == ["你", "好", "OK"]
    asr.need_word_time_stamp = False
    sents = asr._make_segments(resp)
    assert [s.text for s in sents] == ["你好", "OK"]
    assert sents[1].start_time == 500 and sents[1].end_time == 900


MODELS = os.environ.get("VC_LOCAL_ASR_MODELS", "")


@pytest.mark.skipif(not MODELS, reason="需要 VC_LOCAL_ASR_MODELS 指向本地模型目录")
def test_engine_end_to_end_sensevoice():
    from videocaptioner.core.asr.local.engine import EngineOptions, LocalEngine

    wav = Path(MODELS) / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17" / "test_wavs" / "zh.wav"
    engine = LocalEngine(EngineOptions(model="sensevoice", device="cpu", language="zh", models_dir=MODELS))
    utts = engine.transcribe(str(wav))
    text = "".join(u.text for u in utts)
    assert "时间早上" in text
    assert utts[0].units[0].start >= 0


def test_resolve_device_fallbacks(monkeypatch, tmp_path):
    from videocaptioner.core.asr.local import device as D

    monkeypatch.setattr(D, "nvidia_gpus", lambda: [])
    assert D.resolve_device("auto", None).device == "cpu"
    assert D.resolve_device("cpu", None).device == "cpu"

    # 有 N 卡但装的是 CPU 版 sherpa-onnx
    monkeypatch.setattr(D, "nvidia_gpus", lambda: ["NVIDIA GeForce RTX 4060, 8188 MiB"])
    monkeypatch.setattr(D, "sherpa_has_cuda", lambda: False)
    d = D.resolve_device("cuda", None)
    assert d.device == "cpu" and "CPU 版" in d.reason

    # CUDA 版，但运行库缺失（探测失败）→ 回退 CPU
    vad = tmp_path / "vad.onnx"
    vad.write_bytes(b"x")
    monkeypatch.setattr(D, "sherpa_has_cuda", lambda: True)
    monkeypatch.setattr(D, "preload_cuda_libs", lambda: None)
    monkeypatch.setattr(D, "probe_cuda", lambda p: (False, "CUDA 不可用（libcudnn.so.9 not found）"))
    d = D.resolve_device("auto", vad)
    assert d.device == "cpu" and "回退" in d.reason

    # 探测成功 → GPU
    monkeypatch.setattr(D, "probe_cuda", lambda p: (True, "ok"))
    d = D.resolve_device("auto", vad)
    assert d.device == "cuda" and "RTX 4060" in d.gpu_name


def test_probe_cuda_on_cpu_build_reports_unavailable():
    """在 CPU 版 sherpa-onnx 上真实跑一次子进程探测：应判定 CUDA 不可用而不是崩溃。"""
    if not MODELS:
        pytest.skip("需要 VC_LOCAL_ASR_MODELS")
    from videocaptioner.core.asr.local import device as D

    if D.sherpa_has_cuda():
        pytest.skip("装的是 CUDA 版")
    ok, msg = D.probe_cuda(Path(MODELS) / "silero_vad.onnx")
    assert ok is False
