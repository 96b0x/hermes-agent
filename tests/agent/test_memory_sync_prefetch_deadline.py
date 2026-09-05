"""Current-turn providers must not be discarded by the legacy fast-cache deadline."""
import threading
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider
from plugins.memory.hindsight import HindsightMemoryProvider


def test_manager_honors_opted_in_synchronous_budget():
    done = threading.Event()

    def prefetch(query, *, session_id=''):
        done.wait(0.05)
        return 'current-turn evidence'

    provider = SimpleNamespace(name='sync', prefetch=prefetch, prefetch_timeout_seconds=1.0)
    manager = MemoryManager(external_prefetch_timeout=0.001)
    try:
        assert manager._prefetch_provider(provider, 'question') == 'current-turn evidence'
    finally:
        done.set()
        manager.shutdown_all()


def test_hindsight_budget_covers_request_and_compiler():
    provider = HindsightMemoryProvider()
    provider._recall_sync = True
    provider._timeout = 120
    assert provider.prefetch_timeout_seconds >= 150
    provider._recall_sync = False
    assert provider.prefetch_timeout_seconds is None


def test_base_provider_keeps_manager_default():
    assert isinstance(MemoryProvider.prefetch_timeout_seconds, property)


@pytest.mark.parametrize('invalid', [None, -1, 0, float('nan'), float('inf'), 'forever'])
def test_invalid_provider_budget_does_not_disable_timeout(invalid):
    done = threading.Event()
    provider = SimpleNamespace(name='slow', prefetch=lambda *a, **k: done.wait(1), prefetch_timeout_seconds=invalid)
    manager = MemoryManager(external_prefetch_timeout=0.001)
    try:
        assert manager._prefetch_provider(provider, 'question') == ''
    finally:
        done.set()
        manager.shutdown_all()
