"""agentic-kb-lite rg CLI 行为的 M1 库接口回归测试。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from evidence.rg_executor import (
    InvalidRipgrepOutputError,
    RipgrepExecutionError,
    RipgrepExecutor,
    RipgrepNotFoundError,
    resolve_rg_path,
)


def _event(kind, path, **data):
    payload = {"path": {"text": str(path)}, **data}
    return json.dumps({"type": kind, "data": payload}, ensure_ascii=False)


def _match(path, text="前文 needle 后文\n", start=7, end=13):
    return _event(
        "match",
        path,
        lines={"text": text},
        line_number=2,
        submatches=[
            {"match": {"text": text.encode("utf-8")[start:end].decode("utf-8")},
             "start": start, "end": end}
        ],
    )


def _output(path):
    return "\n".join(
        (
            _event("begin", path),
            _event(
                "context",
                path,
                lines={"text": "context\n"},
                line_number=1,
                submatches=[],
            ),
            _match(path),
            _event("end", path),
            json.dumps({"type": "summary", "data": {"stats": {}}}),
        )
    )


class RecordingRunner:
    def __init__(self, process):
        self.process = process
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return self.process


def _executor(process):
    runner = RecordingRunner(process)
    return RipgrepExecutor(rg_path="test-rg", runner=runner), runner


def test_bundled_rg_is_preferred_on_windows(tmp_path, monkeypatch):
    bundled = tmp_path / "tools" / "rg.exe"
    bundled.parent.mkdir()
    bundled.write_bytes(b"test")
    monkeypatch.setattr("evidence.rg_executor.platform.system", lambda: "Windows")
    monkeypatch.setattr("evidence.rg_executor.shutil.which", lambda _: "PATH-rg")
    assert resolve_rg_path(tmp_path) == (str(bundled), "bundled")


def test_path_rg_is_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("evidence.rg_executor.platform.system", lambda: "Windows")
    monkeypatch.setattr("evidence.rg_executor.shutil.which", lambda _: "PATH-rg")
    assert resolve_rg_path(tmp_path) == ("PATH-rg", "system PATH")


def test_missing_rg_has_structured_context(tmp_path, monkeypatch):
    monkeypatch.setattr("evidence.rg_executor.platform.system", lambda: "Windows")
    monkeypatch.setattr("evidence.rg_executor.shutil.which", lambda _: None)
    with pytest.raises(RipgrepNotFoundError) as raised:
        resolve_rg_path(tmp_path, anchors=("锚点",), search_scope=("doc.txt",))
    assert raised.value.anchors == ("锚点",)
    assert raised.value.search_scope == ("doc.txt",)
    assert "PATH" in raised.value.stderr


def test_return_code_zero_parses_and_groups_all_event_types(tmp_path):
    target = tmp_path / "doc.txt"
    process = subprocess.CompletedProcess([], 0, _output(target), "warning")
    executor, runner = _executor(process)
    result = executor.search(["needle"], [target], context=1, globs=("*.txt",))
    assert result.returncode == 0
    assert [event.kind for event in result.events] == ["context", "match"]
    assert tuple(result.grouped_events) == (str(target),)
    assert result.grouped_events[str(target)] == result.events
    command, kwargs = runner.calls[0]
    assert command == [
        "test-rg", "--json", "-i", "-C", "1", "--glob", "*.txt",
        "--fixed-strings", "-e", "needle", "--", str(target),
    ]
    assert kwargs["shell"] is False
    assert kwargs["encoding"] == "utf-8"


def test_return_code_one_is_an_empty_success(tmp_path):
    executor, _ = _executor(subprocess.CompletedProcess([], 1, "", ""))
    result = executor.search(["absent"], [tmp_path / "doc.txt"])
    assert result.returncode == 1
    assert result.events == ()
    assert dict(result.grouped_events) == {}


def test_return_code_two_raises_and_preserves_stderr(tmp_path):
    executor, _ = _executor(
        subprocess.CompletedProcess([], 2, "", "regex parse failed")
    )
    with pytest.raises(RipgrepExecutionError) as raised:
        executor.search(["["], [tmp_path / "doc.txt"], regex=True)
    assert raised.value.returncode == 2
    assert raised.value.stderr == "regex parse failed"
    assert "regex parse failed" in str(raised.value)
    assert raised.value.anchors == ("[",)


def test_regex_mode_is_explicit_and_omits_fixed_strings(tmp_path):
    executor, runner = _executor(subprocess.CompletedProcess([], 1, "", ""))
    executor.search([r"foo\d+"], [tmp_path / "doc.txt"], regex=True)
    assert "--fixed-strings" not in runner.calls[0][0]
    assert r"foo\d+" in runner.calls[0][0]


@pytest.mark.parametrize(
    "stdout",
    (
        "not-json",
        json.dumps({"type": "unknown", "data": {}}),
        json.dumps({"type": "match", "data": {"path": {"text": "x"}}}),
    ),
)
def test_malformed_or_incomplete_json_is_not_silently_ignored(tmp_path, stdout):
    executor, _ = _executor(subprocess.CompletedProcess([], 0, stdout, "bad output"))
    with pytest.raises(InvalidRipgrepOutputError) as raised:
        executor.search(["needle"], [tmp_path / "doc.txt"])
    assert raised.value.returncode == 0
    assert raised.value.stderr == "bad output"


def test_incomplete_event_group_is_not_silently_accepted(tmp_path):
    target = tmp_path / "doc.txt"
    stdout = "\n".join((_event("begin", target), _match(target)))
    executor, _ = _executor(subprocess.CompletedProcess([], 0, stdout, ""))
    with pytest.raises(InvalidRipgrepOutputError, match="before the current group ended"):
        executor.search(["needle"], [target])


def test_shell_metacharacters_remain_one_argv_value(tmp_path):
    malicious = 'needle; Remove-Item -Recurse * | $(whoami) "quoted"'
    executor, runner = _executor(subprocess.CompletedProcess([], 1, "", ""))
    executor.search([malicious], [tmp_path / "doc.txt"])
    command, kwargs = runner.calls[0]
    assert command[command.index("-e") + 1] == malicious
    assert kwargs["shell"] is False
    assert isinstance(command, list)


def test_file_not_found_during_execution_is_typed(tmp_path):
    def missing(*args, **kwargs):
        raise FileNotFoundError("gone")

    executor = RipgrepExecutor(rg_path="gone-rg", runner=missing)
    with pytest.raises(RipgrepNotFoundError) as raised:
        executor.search(["needle"], [tmp_path / "doc.txt"])
    assert raised.value.returncode is None
    assert raised.value.command[0] == "gone-rg"
