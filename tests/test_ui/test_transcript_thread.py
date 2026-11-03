"""TranscriptThread 取消语义与本地引擎直读原文件的测试。

用 offscreen 平台直接同步调用 run()，不真正起线程。
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from videocaptioner.core.asr.local.engine import TranscribeCancelled
from videocaptioner.core.entities import (
    TranscribeConfig,
    TranscribeModelEnum,
    TranscribeOutputFormatEnum,
)

from videocaptioner.ui.thread import transcript_thread as tt


def _make_task(tmp_path: Path, model: TranscribeModelEnum) -> SimpleNamespace:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fake")
    config = TranscribeConfig(
        transcribe_model=model,
        output_format=TranscribeOutputFormatEnum.SRT,
    )
    return SimpleNamespace(
        file_path=str(video),
        transcribe_config=config,
        output_path=str(tmp_path / "out.srt"),
        need_next_task=False,
        selected_audio_track_index=0,
        started_at=None,
    )


class Recorder:
    """收集线程信号用的列表。"""

    def __init__(self):
        self.cancelled = []
        self.finished = []
        self.errors = []

    def wire(self, thread: tt.TranscriptThread):
        thread.cancelled.connect(lambda: self.cancelled.append(1))
        thread.finished.connect(lambda _t: self.finished.append(1))
        thread.error.connect(lambda m: self.errors.append(m))


@pytest.fixture(autouse=True)
def _no_real_transcribe(monkeypatch):
    """所有用例都不真正跑识别，只记录传入的音频路径。"""
    calls = {}

    def fake_transcribe(audio_path, config, callback=None, stop_event=None):
        calls["audio_path"] = audio_path
        raise TranscribeCancelled()

    monkeypatch.setattr(tt, "transcribe", fake_transcribe)
    calls["audio_path"] = None
    return calls


# ---------- 取消语义（#6）----------


def test_cancelled_exception_reports_cancelled(tmp_path):
    thread = tt.TranscriptThread(_make_task(tmp_path, TranscribeModelEnum.LOCAL))
    rec = Recorder()
    rec.wire(thread)
    thread.run()
    assert rec.cancelled and not rec.errors and not rec.finished


def test_real_error_after_cancel_click_still_reports_error(tmp_path, monkeypatch):
    """用户点了取消之后发生的真实错误（如模型损坏）必须上报，不再吞成「已取消」。"""
    thread = tt.TranscriptThread(_make_task(tmp_path, TranscribeModelEnum.LOCAL))
    thread.stop_event.set()  # 模拟取消按钮已点过
    rec = Recorder()
    rec.wire(thread)

    def broken_transcribe(audio_path, config, callback=None, stop_event=None):
        raise RuntimeError("sherpa-onnx 模型损坏")

    monkeypatch.setattr(tt, "transcribe", broken_transcribe)
    thread.run()
    assert rec.errors and "模型损坏" in rec.errors[0]
    assert not rec.cancelled and not rec.finished


def test_cancel_clicked_but_task_completes_reports_cancelled(tmp_path):
    """点过取消后任务正常跑完：按取消处理，不冒充「转录完成」。"""
    task = _make_task(tmp_path, TranscribeModelEnum.LOCAL)
    thread = tt.TranscriptThread(task)
    thread.stop_event.set()
    rec = Recorder()
    rec.wire(thread)

    def succeed_transcribe(audio_path, config, callback=None, stop_event=None):
        from videocaptioner.core.asr.asr_data import ASRData

        return ASRData([])

    monkeypatch_transcribe = succeed_transcribe  # noqa: F841
    import unittest.mock as mock

    with mock.patch.object(tt, "transcribe", succeed_transcribe):
        thread.run()
    assert rec.cancelled and not rec.finished and not rec.errors


# ---------- 本地引擎跳过整段转码（#7）----------


def test_local_engine_reads_source_directly(tmp_path, monkeypatch, _no_real_transcribe):
    converted = []
    monkeypatch.setattr(tt, "video2audio", lambda *a, **k: converted.append(a) or True)
    task = _make_task(tmp_path, TranscribeModelEnum.LOCAL)
    thread = tt.TranscriptThread(task)
    rec = Recorder()
    rec.wire(thread)
    thread.run()
    assert not converted, "本地引擎不应做整段 video2audio 转码"
    assert _no_real_transcribe["audio_path"] == task.file_path, "本地引擎应直接读原文件"


def test_cloud_engine_converts_and_uses_wav(tmp_path, monkeypatch, _no_real_transcribe):
    """云端引擎先转音频，且必须把转换出的临时 WAV 传给识别，不能拿原视频。"""
    converted = []
    monkeypatch.setattr(
        tt, "video2audio", lambda *a, **k: converted.append(a) or True
    )
    task = _make_task(tmp_path, TranscribeModelEnum.BIJIAN)
    thread = tt.TranscriptThread(task)
    rec = Recorder()
    rec.wire(thread)
    thread.run()
    assert len(converted) == 1, "云端引擎仍需要先转音频"
    used = _no_real_transcribe["audio_path"]
    assert used != task.file_path, "识别输入不能还是原视频文件"
    assert used.endswith(".wav"), "识别输入应是转换出的临时 WAV"


def test_local_with_selected_audio_track_still_converts(
    tmp_path, monkeypatch, _no_real_transcribe
):
    """用户显式选了非默认音轨时，本地识别也要走转码来落实音轨选择。"""
    converted = []
    captured_kwargs = {}
    monkeypatch.setattr(
        tt,
        "video2audio",
        lambda *a, **k: captured_kwargs.update(k) or converted.append(a) or True,
    )
    task = _make_task(tmp_path, TranscribeModelEnum.LOCAL)
    task.selected_audio_track_index = 1
    thread = tt.TranscriptThread(task)
    rec = Recorder()
    rec.wire(thread)
    thread.run()
    assert len(converted) == 1, "选了非默认音轨就必须转码"
    assert captured_kwargs.get("audio_track_index") == 1
    assert _no_real_transcribe["audio_path"].endswith(".wav")


def test_download_cancelled_reports_cancelled(tmp_path, monkeypatch):
    """取消本地识别时，模型可能正在下载（抛 DownloadCancelled），应按取消处理。"""
    thread = tt.TranscriptThread(_make_task(tmp_path, TranscribeModelEnum.LOCAL))
    thread.stop_event.set()
    rec = Recorder()
    rec.wire(thread)

    def cancelled_transcribe(audio_path, config, callback=None, stop_event=None):
        from videocaptioner.core.asr.local.models import DownloadCancelled

        raise DownloadCancelled()

    monkeypatch.setattr(tt, "transcribe", cancelled_transcribe)
    thread.run()
    assert rec.cancelled and not rec.errors and not rec.finished


# ---------- 协作式停止（#5）----------


def test_stop_is_cooperative_without_terminate(tmp_path):
    """stop() 只设取消标志并有限等待，绝不能 self.terminate() 强杀线程。"""
    import inspect

    source = inspect.getsource(tt.TranscriptThread.stop)
    assert "self.terminate" not in source
    thread = tt.TranscriptThread(_make_task(tmp_path, TranscribeModelEnum.LOCAL))
    thread.stop(timeout_ms=1)  # 未启动的线程应立即返回，不报错
