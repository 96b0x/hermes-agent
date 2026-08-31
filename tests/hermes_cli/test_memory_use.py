"""Tests for switching among already-configured memory providers.

`hermes memory use` changes only the active-provider selector. Provider-native
configuration and persisted state must remain untouched so users can switch
away and later resume the same provider.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_cli.memory_setup as memory_setup


class _Provider:
    def __init__(self, available=True):
        self._available = available

    def is_available(self):
        return self._available


def _args(provider, force=False):
    return SimpleNamespace(memory_command="use", provider=provider, force=force)


def test_use_builtin_clears_only_active_selector(monkeypatch, capsys):
    config = {
        "memory": {
            "provider": "hindsight",
            "hindsight": {"mode": "local_embedded", "bank_id": "saved"},
            "mem0": {"mode": "oss", "collection": "also-saved"},
        }
    }
    save = MagicMock()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", save)

    assert memory_setup.cmd_use(_args("builtin")) is True

    assert config["memory"]["provider"] == ""
    assert config["memory"]["hindsight"]["bank_id"] == "saved"
    assert config["memory"]["mem0"]["collection"] == "also-saved"
    save.assert_called_once_with(config)
    assert "built-in only" in capsys.readouterr().out


def test_use_external_provider_preserves_every_provider_config(monkeypatch, capsys):
    config = {
        "memory": {
            "provider": "mem0",
            "hindsight": {"mode": "local_embedded", "bank_id": "saved"},
            "mem0": {"mode": "oss", "collection": "also-saved"},
        }
    }
    save = MagicMock()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", save)
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("hindsight", "local", _Provider(available=True))],
    )

    assert memory_setup.cmd_use(_args("hindsight")) is True

    assert config["memory"]["provider"] == "hindsight"
    assert config["memory"]["hindsight"]["bank_id"] == "saved"
    assert config["memory"]["mem0"]["collection"] == "also-saved"
    save.assert_called_once_with(config)
    output = capsys.readouterr().out
    assert "hindsight" in output
    assert "new session" in output.lower()


def test_use_unknown_provider_does_not_mutate_config(monkeypatch, capsys):
    config = {"memory": {"provider": "hindsight"}}
    save = MagicMock()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", save)
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    assert memory_setup.cmd_use(_args("not-a-provider")) is False

    assert config["memory"]["provider"] == "hindsight"
    save.assert_not_called()
    output = capsys.readouterr().out
    assert "not found" in output.lower()
    assert "hermes memory status" in output


def test_use_unavailable_provider_requires_force(monkeypatch, capsys):
    config = {"memory": {"provider": ""}}
    save = MagicMock()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", save)
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("mem0", "API key / local", _Provider(available=False))],
    )

    assert memory_setup.cmd_use(_args("mem0")) is False
    assert config["memory"]["provider"] == ""
    save.assert_not_called()
    assert "--force" in capsys.readouterr().out


def test_use_force_allows_unavailable_provider(monkeypatch):
    config = {"memory": {"provider": ""}}
    save = MagicMock()
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr("hermes_cli.config.save_config", save)
    monkeypatch.setattr(
        memory_setup,
        "_get_available_providers",
        lambda: [("mem0", "API key / local", _Provider(available=False))],
    )

    assert memory_setup.cmd_use(_args("mem0", force=True)) is True
    assert config["memory"]["provider"] == "mem0"
    save.assert_called_once_with(config)


def test_memory_command_routes_use(monkeypatch):
    use = MagicMock()
    monkeypatch.setattr(memory_setup, "cmd_use", use)
    args = _args("hindsight")

    memory_setup.memory_command(args)

    use.assert_called_once_with(args)
