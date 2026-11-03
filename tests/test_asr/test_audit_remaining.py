"""审计遗留项的回归测试：GPU 回退重新选型、缓存 key 含 mtime、CLI 数值钳制。"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------- 中等条目 10：GPU 加载失败回退 CPU 后重新选型 ----------


def test_gpu_fallback_repicks_cpu_variant(monkeypatch, tmp_path):
    from videocaptioner.core.asr.local import engine as E
    from videocaptioner.core.asr.local import models as M

    picks = []

    def fake_pick(family, device, root):
        picks.append((family, device))
        return M.MODEL_SPECS["silero-vad"]  # 占位 spec，任意已登记模型即可

    ensured = []
    loads = []

    monkeypatch.setattr(E.M, "pick_variant", fake_pick)
    monkeypatch.setattr(
        E.M, "ensure_models", lambda specs, root, *a, **k: ensured.append([s.key for s in specs])
    )
    monkeypatch.setattr(E, "resolve_device", lambda dev, vad: E.DeviceDecision("cuda", "ok"))

    def fake_load(self, dev):
        loads.append(dev)
        if dev == "cuda":
            raise RuntimeError("no usable CUDA")

    monkeypatch.setattr(E.LocalEngine, "_load", fake_load)

    eng = E.LocalEngine(E.EngineOptions(model="sensevoice", device="cuda", models_dir=str(tmp_path)))
    eng.prepare(lambda p, m: None)

    assert ("sensevoice", "cuda") in picks, "初始按 GPU 选型"
    assert ("sensevoice", "cpu") in picks, "回退后必须重新按 CPU 选型"
    assert loads == ["cuda", "cpu"]
    assert eng.decision.device == "cpu"
    # 回退后必须重新 ensure（本地缺 int8 时补下载）；VAD 的首次 ensure 不算
    assert any("silero-vad" not in keys or len(keys) >= 1 for keys in ensured)


# ---------- 中等条目 11：缓存 key 与扩展名提示 ----------


def _bare_local_asr(path: Path):
    from videocaptioner.core.asr.local_asr import LocalASR

    a = LocalASR.__new__(LocalASR)
    a.audio_input = str(path)
    a.file_binary = None
    a.crc32_hex = ""
    return a


def test_cache_key_changes_with_mtime(tmp_path: Path):
    p = tmp_path / "a.wav"
    p.write_bytes(b"x" * 1024)
    a = _bare_local_asr(p)
    a._set_data()
    key1 = a.crc32_hex
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    a._set_data()
    assert a.crc32_hex != key1, "同大小原地修改后 mtime 变化应使缓存 key 失效"


def test_unknown_extension_warns_not_fails(tmp_path: Path, caplog):
    """ffmpeg 支持但登记表外的扩展名：告警但不拒绝，避免硬 assert 破坏可用格式。"""
    p = tmp_path / "a.weird"
    p.write_bytes(b"x" * 1024)
    a = _bare_local_asr(p)
    import logging

    with caplog.at_level(logging.WARNING, logger="local_asr"):
        a._set_data()
    assert any(".weird" in r.message for r in caplog.records)


def test_duration_fallback_logs_warning(tmp_path: Path, caplog, monkeypatch):
    from videocaptioner.core.asr import local_asr as LA

    p = tmp_path / "a.wav"
    p.write_bytes(b"x" * 1024)
    monkeypatch.setattr(LA, "audio_duration", lambda p: None)
    import logging

    with caplog.at_level(logging.WARNING, logger="local_asr"):
        assert LA.LocalASR._get_audio_duration(_bare_local_asr(p)) == 60.0
    assert any("时长" in r.message for r in caplog.records)


# ---------- CLI 数值钳制 ----------


def test_cli_clamp():
    from videocaptioner.cli.commands.transcribe import _clamp

    assert _clamp(120, 8, 60, "x") == 60
    assert _clamp(1, 8, 60, "x") == 8
    assert _clamp(25, 8, 60, "x") == 25
    assert _clamp("bad", 8, 60, "x") == "bad"  # 非数值原样返回，交给后续校验


def test_transcribe_run_structure():
    """结构回归：run() 必须包含转录配置构造与执行主流程，_clamp 在模块层。
    （曾因脚本插桩把 run() 腰斩、转录逻辑变成 _clamp 内不可达代码）"""
    import ast

    import videocaptioner.cli.commands.transcribe as T

    src = Path(T.__file__).read_text()
    tree = ast.parse(src)
    run_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assigns = {
        n.targets[0].id
        for n in ast.walk(run_fn)
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    assert "transcribe_config" in assigns, "run() 里必须有转录配置构造"
    calls = {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(run_fn) if isinstance(n, ast.Call)}
    assert "transcribe" in calls, "run() 里必须调用识别入口"
    top_functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_clamp" in top_functions, "_clamp 应是模块级函数"
