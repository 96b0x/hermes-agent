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
def test_invalid_provider_budget_keeps_bounded_host_wait(budget, monkeypatch):
    release = threading.Event()
    threads = []

    def spawn(target, **kwargs):
        thread = spawn_context_thread(target, **kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr("agent.memory_manager.spawn_context_thread", spawn)
    provider = SimpleNamespace(
        name="probe", prefetch=lambda *a, **k: release.wait(5) and "late evidence",
        prefetch_timeout_seconds=budget,
    )
    manager = MemoryManager(external_prefetch_timeout=0.001)
    try:
        assert manager._prefetch_provider(provider, "current question") == ""
    finally:
        release.set()
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
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


@pytest.mark.parametrize("hook", ["prefetch", "turn"])
def test_foreign_owner_cannot_invalidate_current_recall(monkeypatch, hook):
    manager = MemoryManager()
    manager._prefetch_generation = 7
    monkeypatch.setattr(manager, "_owner_allows", lambda _hook: False)
    if hook == "prefetch":
        assert manager.prefetch_all("foreign request") == ""
    else:
        manager.on_turn_start(2, "foreign request")
    assert manager._prefetch_generation == 7


def test_provider_budget_is_used_for_shared_gate_deadline(monkeypatch):
    manager = MemoryManager(external_prefetch_timeout=0.01)
    observed = []
    real_enter = manager._PREFETCH_GATE.enter

    def enter(key, token, timeout):
        observed.append(timeout)
        return real_enter(key, token, timeout)

    monkeypatch.setattr(manager._PREFETCH_GATE, "enter", enter)
    provider = SimpleNamespace(name="budget-gate-probe", prefetch=lambda *a, **k: "evidence",
                               prefetch_timeout_seconds=3.0)
    assert manager._prefetch_provider(provider, "question") == "evidence"
    assert observed == [3.0]
    assert manager._registry_key(provider) not in manager._PREFETCH_GATE.calls


def test_thread_start_failure_does_not_pin_shared_gate(monkeypatch):
    manager = MemoryManager()

    class CannotStart:
        def start(self):
            raise RuntimeError("thread start refused")

    monkeypatch.setattr("agent.memory_manager.spawn_context_thread", lambda *a, **k: CannotStart())
    provider = SimpleNamespace(name="start-failure-probe", prefetch=lambda *a, **k: "unused")
    with pytest.raises(RuntimeError, match="thread start refused"):
        manager._prefetch_provider(provider, "question")
    assert manager._registry_key(provider) not in manager._PREFETCH_GATE.calls


def test_relocated_context_renderer_preserves_evidence_contract():
    from agent.memory_context import build_memory_context_block, sanitize_context
    block = build_memory_context_block("- evidence")
    assert "may be stale, incomplete, or conflicting" in block
    assert "authoritative reference data" not in block
    note = block.splitlines()[1]
    assert sanitize_context(note + "\n\nVisible text") == "Visible text"
    assert sanitize_context(block) == ""
