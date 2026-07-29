"""Public embedding configuration must be usable without editing Python source."""

from __future__ import annotations

import json

from evidence.gap_checker import main
from tests.test_m1_gap_checker import _cli_inputs, _minimal_report


class _Checker:
    def __init__(self, report):
        self.report = report

    def run(self, **_kwargs):
        return type("Run", (), {"report": self.report})()


def test_embedding_precedence_is_defaults_then_json_then_env_then_cli(
    tmp_path,
    monkeypatch,
):
    manifest, output = _cli_inputs(tmp_path)
    config = tmp_path / "embedding.json"
    config.write_text(
        json.dumps(
            {
                "embedding": {
                    "embedding_backend": "api",
                    "embedding_api_base": "https://config.example/v1",
                    "embedding_api_model": "config-model",
                    "embedding_api_timeout": 12,
                    "embedding_api_key_env": "CONFIG_API_KEY",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVIDENCE_EMBEDDING_API_BASE", "https://env.example/v1")
    monkeypatch.setenv("CONFIG_API_KEY", "secret-from-configured-env")
    captured = {}

    def factory(settings):
        captured["settings"] = settings
        return _Checker(_minimal_report("complete"))

    code = main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--embedding-config",
            str(config),
            "--embedding-api-model",
            "cli-model",
            "--embedding-api-timeout",
            "9.5",
            "--fake-embedder",
        ],
        checker_factory=factory,
    )

    assert code == 0
    settings = captured["settings"]
    assert settings.embedding_api_base == "https://env.example/v1"
    assert settings.embedding_api_model == "cli-model"
    assert settings.embedding_api_timeout == 9.5
    assert settings.embedding_api_key == "secret-from-configured-env"


def test_cli_can_select_a_custom_api_key_environment_variable(
    tmp_path,
    monkeypatch,
):
    manifest, output = _cli_inputs(tmp_path)
    monkeypatch.setenv("CUSTOM_EMBEDDING_KEY", "custom-secret")
    captured = {}

    def factory(settings):
        captured["key"] = settings.embedding_api_key
        return _Checker(_minimal_report("complete"))

    code = main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--embedding-api-key-env",
            "CUSTOM_EMBEDDING_KEY",
            "--fake-embedder",
        ],
        checker_factory=factory,
    )
    assert code == 0
    assert captured["key"] == "custom-secret"


def test_embedding_config_rejects_a_persisted_api_key_without_leaking_it(
    tmp_path,
    capsys,
):
    manifest, output = _cli_inputs(tmp_path)
    config = tmp_path / "embedding.json"
    secret = "must-not-appear"
    config.write_text(
        json.dumps({"embedding_api_key": secret}),
        encoding="utf-8",
    )

    code = main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--embedding-config",
            str(config),
        ],
        checker_factory=lambda settings: _Checker(_minimal_report("complete")),
    )
    stderr = capsys.readouterr().err
    assert code == 2
    assert secret not in stderr
    assert "不得保存 API key" in stderr


def test_resolved_remote_endpoint_triggers_the_data_egress_warning(
    tmp_path,
    capsys,
):
    manifest, output = _cli_inputs(tmp_path)
    code = main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--embedding-api-base",
            "https://embedding.example/v1",
        ],
        checker_factory=lambda settings: _Checker(_minimal_report("complete")),
    )
    assert code == 0
    assert "材料内容" in capsys.readouterr().err


def test_embedding_timeout_must_be_positive(tmp_path, capsys):
    manifest, output = _cli_inputs(tmp_path)
    code = main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--embedding-api-timeout",
            "0",
        ],
        checker_factory=lambda settings: _Checker(_minimal_report("complete")),
    )
    assert code == 2
    assert "必须大于 0" in capsys.readouterr().err
