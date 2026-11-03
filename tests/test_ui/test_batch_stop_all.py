"""BatchProcessThread.stop_all 对不同 worker stop 签名的分派测试。

不真正起 QThread，用轻量 stub 验证分派与回收逻辑。
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from videocaptioner.ui.thread.batch_process_thread import BatchProcessThread


def _make_batch(*threads) -> BatchProcessThread:
    batch = BatchProcessThread.__new__(BatchProcessThread)
    batch.task_queue = __import__("queue").Queue()
    batch.current_tasks = {}
    batch.max_concurrent_tasks = 1
    batch.is_running = True
    batch.factory = None
    batch.threads = list(threads)
    batch.zombie_threads = []
    return batch


def test_stop_with_timeout_support():
    calls = []
    t = SimpleNamespace(
        stop=lambda timeout_ms=None: calls.append(timeout_ms),
        isRunning=lambda: False,
    )
    batch = _make_batch(t)
    batch.stop_all()
    assert calls == [1000], "支持 timeout_ms 的 worker 应带超时调用"
    assert batch.is_running is False and not batch.threads


def test_stop_without_timeout_support():
    calls = []
    t = SimpleNamespace(stop=lambda: calls.append(1), isRunning=lambda: False)
    batch = _make_batch(t)
    batch.stop_all()  # 不应抛 TypeError（SubtitleThread 场景）
    assert calls == [1]


def test_worker_without_stop_is_retired_if_running():
    """没有 stop 的 worker（如 VideoSynthesisThread）若在运行，应进僵尸列表收尾。"""
    t = SimpleNamespace(isRunning=lambda: True)
    batch = _make_batch(t)
    batch.stop_all()
    assert batch.zombie_threads == [t]
    assert not batch.threads


def test_stopped_worker_not_retired():
    t = SimpleNamespace(stop=lambda: None, isRunning=lambda: False)
    batch = _make_batch(t)
    batch.stop_all()
    assert batch.zombie_threads == []
