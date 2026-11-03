"""模型下载与解压的安全校验测试：哈希校验、tar 成员过滤、安装状态判断。"""

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from videocaptioner.core.asr.local import models as M


GOOD = b"good model content"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
DIRNAME = "fake-model-2026-01-01"


def _spec(sha256: str = GOOD_SHA, is_archive: bool = False) -> M.ModelSpec:
    asset = f"asr-models/{DIRNAME}.tar.bz2" if is_archive else f"asr-models/{DIRNAME}.onnx"
    return M.ModelSpec(
        key="fake",
        family="vad",
        title="测试模型",
        asset=asset,
        dirname=DIRNAME if is_archive else f"{DIRNAME}.onnx",
        files={"model": ""} if not is_archive else {"model": "model.onnx"},
        sha256=sha256,
    )


def _add_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name)
    ti.size = len(data)
    tar.addfile(ti, io.BytesIO(data))


# ---------- _verify_hash ----------


def test_verify_hash_accepts_matching(tmp_path: Path):
    dest = tmp_path / "m.onnx"
    dest.write_bytes(GOOD)
    M._verify_hash(dest, _spec())  # 不抛异常即通过


def test_verify_hash_rejects_tampered(tmp_path: Path):
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"tampered!")
    with pytest.raises(M.HashMismatchError):
        M._verify_hash(dest, _spec())


def test_verify_hash_skipped_without_registered_sha(tmp_path: Path):
    dest = tmp_path / "m.onnx"
    dest.write_bytes(b"anything")
    M._verify_hash(dest, _spec(sha256=""))


# ---------- download_model：校验失败换源、坏数据不残留 ----------


def test_download_model_retries_next_source_after_hash_mismatch(
    monkeypatch, tmp_path: Path
):
    spec = _spec()
    calls = []

    def fake_download(url, dest, progress, label, stop_event):
        calls.append(url)
        # 第一个源给坏内容，第二个源给好内容
        dest.write_bytes(GOOD if len(calls) > 1 else b"evil content")

    monkeypatch.setattr(M, "_download_file", fake_download)
    out = M.download_model(spec, tmp_path)
    assert len(calls) == 2, "校验失败后应尝试下一个源"
    assert out.read_bytes() == GOOD
    assert not list(tmp_path.glob("*.part")), "坏数据的 .part 应被清理"


def test_download_model_raises_when_all_sources_tampered(monkeypatch, tmp_path: Path):
    def fake_download(url, dest, progress, label, stop_event):
        dest.write_bytes(b"evil content")

    monkeypatch.setattr(M, "_download_file", fake_download)
    with pytest.raises(RuntimeError, match="校验失败"):
        M.download_model(_spec(), tmp_path)
    assert not list(tmp_path.glob("*.part"))


def test_download_model_concurrent_calls_single_flight(monkeypatch, tmp_path: Path):
    """同一模型的并发下载应串行，第二个调用直接命中已安装。"""
    import threading

    n = {"downloads": 0}

    def fake_download(url, dest, progress, label, stop_event):
        n["downloads"] += 1
        dest.write_bytes(GOOD)

    monkeypatch.setattr(M, "_download_file", fake_download)
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        M.download_model(_spec(), tmp_path)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert n["downloads"] == 1


# ---------- _source_urls：自定义镜像必须 https ----------


def test_source_urls_custom_requires_https():
    with pytest.raises(ValueError, match="https"):
        M._source_urls(_spec(), "custom", "http://insecure.example.com/")


def test_source_urls_custom_https_ok():
    urls = M._source_urls(_spec(), "custom", "https://mirror.example.com/")
    assert urls[0][1].startswith("https://mirror.example.com/")
    assert urls[1][1] == M.GITHUB_RELEASE + _spec().asset


# ---------- is_installed：空文件视为未安装 ----------


def test_is_installed_rejects_empty_file(tmp_path: Path):
    spec = _spec()
    p = M.model_path(spec, "model", tmp_path)
    p.write_bytes(b"")
    assert M.is_installed(spec, tmp_path) is False
    p.write_bytes(GOOD)
    assert M.is_installed(spec, tmp_path) is True


# ---------- _extract：tar 成员过滤 ----------


def _make_archive(path: Path, entries) -> None:
    with tarfile.open(path, "w:bz2") as t:
        for e in entries:
            if e[1] == "sym":
                ti = tarfile.TarInfo(e[0])
                ti.type = tarfile.SYMTYPE
                ti.linkname = e[2]
                t.addfile(ti)
            else:
                _add_file(t, e[0], e[2])


def test_extract_accepts_clean_archive(tmp_path: Path):
    spec = _spec(is_archive=True)
    archive = tmp_path / "a.tar.bz2"
    _make_archive(
        archive,
        [
            (f"{DIRNAME}/model.onnx", "file", GOOD),
            (f"{DIRNAME}/tokens.txt", "file", b"tokens"),
            (f"{DIRNAME}/test_wavs/zh.wav", "file", b"skip me"),
        ],
    )
    M._extract(archive, spec, tmp_path)
    assert M.model_path(spec, "model", tmp_path).read_bytes() == GOOD
    assert not (tmp_path / DIRNAME / "test_wavs").exists(), "test_wavs 应被跳过"


def test_extract_rejects_symlink_member(tmp_path: Path):
    spec = _spec(is_archive=True)
    archive = tmp_path / "evil.tar.bz2"
    _make_archive(
        archive,
        [
            (f"{DIRNAME}/model.onnx", "sym", "/home/user/.bashrc"),
            (f"{DIRNAME}/model.onnx", "file", b"overwrite via link"),
        ],
    )
    with pytest.raises(IOError, match="链接成员"):
        M._extract(archive, spec, tmp_path)


def test_extract_rejects_path_traversal(tmp_path: Path):
    # 名字伪装成 wanted 前缀、再借 .. 跳出，必须被安全检查拦下
    spec = _spec(is_archive=True)
    archive = tmp_path / "evil.tar.bz2"
    _make_archive(
        archive, [(f"{DIRNAME}/model.onnx/../../../escape", "file", b"pwned")]
    )
    with pytest.raises(IOError, match="不安全的路径"):
        M._extract(archive, spec, tmp_path)
    assert not (tmp_path.parent / "escape").exists()


def test_extract_reports_missing_wanted_files(tmp_path: Path):
    spec = _spec(is_archive=True)
    archive = tmp_path / "empty.tar.bz2"
    _make_archive(archive, [("unrelated/other.bin", "file", b"x")])
    with pytest.raises(IOError, match="找不到需要的文件"):
        M._extract(archive, spec, tmp_path)
