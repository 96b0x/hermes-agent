"""Current-turn recall budgets, scope propagation, and stale completion fences."""

import contextvars
import threading
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, call_auxiliary_task, spawn_context_thread


class _Provider(MemoryProvider):
    name = "current-turn-probe"

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []


def test_provider_budget_extends_host_wait_and_copies_scope(monkeypatch):
    scope = contextvars.ContextVar("memory_test_scope", default="unbound")
    token = scope.set("profile-a")
    observed = []
    real_spawn = spawn_context_thread

    def spawn(target, **kwargs):
        thread = real_spawn(target, **kwargs)
        join = thread.join

        def record_join(timeout=None):
            observed.append(timeout)
            return join(timeout)

        thread.join = record_join
        return thread

    monkeypatch.setattr("agent.memory_manager.spawn_context_thread", spawn)
    provider = SimpleNamespace(name="probe", prefetch=lambda *a, **k: scope.get(), prefetch_timeout_seconds=3.0)
    manager = MemoryManager(external_prefetch_timeout=0.01)
    try:
        assert manager._prefetch_provider(provider, "current question") == "profile-a"
        assert observed == [3.0]
    finally:
        scope.reset(token)
        manager.shutdown_all()


@pytest.mark.parametrize("budget", [None, -1, 0, float("nan"), float("inf"), "forever", True])
def test_invalid_provider_budget_keeps_bounded_host_wait(budget):
    release = threading.Event()
    provider = SimpleNamespace(
        name="probe", prefetch=lambda *a, **k: release.wait(5) and "late evidence",
        prefetch_timeout_seconds=budget,
    )
    manager = MemoryManager(external_prefetch_timeout=0.001)
    try:
        assert manager._prefetch_provider(provider, "current question") == ""
    finally:
        release.set()
        for thread in manager._external_prefetch_threads.values():
            thread.join(5)
        manager.shutdown_all()


@pytest.mark.parametrize("boundary", ["switch", "rewind", "turn", "shutdown", "async_boundary"])
def test_pending_recall_cannot_cross_generation_boundary(boundary):
    started, release = threading.Event(), threading.Event()
    results = []
    manager = MemoryManager(external_prefetch_timeout=5)

    class Provider(_Provider):
        def prefetch(self, query, *, session_id="", messages=None):
            started.set()
            assert release.wait(5)
            return "evidence for " + session_id

    manager.add_provider(Provider())
    caller = spawn_context_thread(
        lambda: results.append(manager.prefetch_all("question", session_id="old")), name="test-recall-caller")
    caller.start()
    try:
        assert started.wait(5)
        if boundary == "turn":
            manager.on_turn_start(2, "next turn")
        elif boundary == "shutdown":
            manager.shutdown_all()
        elif boundary == "async_boundary":
            manager.commit_session_boundary_async([], new_session_id="new")
        else:
            manager.on_session_switch("old" if boundary == "rewind" else "new", rewound=boundary == "rewind")
        release.set()
        caller.join(5)
        assert not caller.is_alive()
        assert results == [""]
    finally:
        release.set()
        caller.join(5)
        manager.shutdown_all()


def test_history_redaction_learns_from_filtered_tool_messages_without_mutating_input():
    seen = []

    class Provider(_Provider):
        def prefetch(self, query, *, session_id="", messages=None):
            seen.append((query, messages))
            return "ranked evidence"

    key = "AQ.Ab8RN6Jx2kq9Zs0VwT4yLm3PbQe7HcUfGdA1nX5oIr"
    history = [
        {"role": "tool", "content": '{"api_key": "' + key + '"}'},
        {"role": "assistant", "content": "The key was " + key},
    ]
    manager = MemoryManager()
    manager.add_provider(Provider())
    try:
        assert manager.prefetch_all("Explain " + key, messages=history) == "ranked evidence"
        assert key not in repr(seen)
        assert [m["role"] for m in seen[0][1]] == ["assistant"]
        assert key in history[1]["content"]
    finally:
        manager.shutdown_all()


def test_provider_auxiliary_task_uses_host_route_and_preserves_completion(monkeypatch):
    from agent import auxiliary_client

    seen = []
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ids": []}'))])

    def call(**kwargs):
        seen.append(kwargs)
        return completion

    monkeypatch.setattr(auxiliary_client, "call_llm", call)
    messages = [{"role": "user", "content": "Arrange retrieved evidence IDs"}]
    assert call_auxiliary_task("memory_context", messages=messages, temperature=0, max_tokens=1200) is completion
    assert seen == [{"task": "memory_context", "messages": messages, "temperature": 0,
                     "max_tokens": 1200, "timeout": None}]
