"""严格的 ripgrep JSON 库接口。

接口保留 bundled-first、PATH fallback、显式参数列表、返回码、JSON event 与 context grouping，
并使用 typed errors 代替进程级退出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import platform
from pathlib import Path
import shutil
import subprocess
from types import MappingProxyType
from typing import Callable, Mapping, Sequence


def _strings(values: object, field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings; got {values!r}")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of strings; got {values!r}") from exc
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must contain at least one item")
    for index, value in enumerate(result):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string; got {value!r}")
    return result


class RipgrepError(RuntimeError):
    """所有 rg 库错误的结构化基类。"""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str] = (),
        returncode: int | None = None,
        stderr: str = "",
        anchors: Sequence[str] = (),
        search_scope: Sequence[str] = (),
    ) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr
        self.anchors = tuple(anchors)
        self.search_scope = tuple(search_scope)
        details = (
            f"command={self.command!r}, returncode={returncode!r}, "
            f"stderr={stderr!r}, anchors={self.anchors!r}, "
            f"search_scope={self.search_scope!r}"
        )
        super().__init__(f"{message}; {details}")


class RipgrepNotFoundError(RipgrepError):
    """bundled 与 PATH 均找不到 rg，或执行文件在调用时消失。"""


class RipgrepExecutionError(RipgrepError):
    """rg 返回码 >=2 或无法启动。"""


class InvalidRipgrepOutputError(RipgrepError):
    """rg JSON 非法、事件结构不完整或 event grouping 不一致。"""


@dataclass(frozen=True)
class RipgrepSubmatch:
    """rg match event 中一段 UTF-8 byte offset 子匹配。"""

    text: str
    start_byte: int
    end_byte: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError(f"RipgrepSubmatch.text must be non-empty; got {self.text!r}")
        if type(self.start_byte) is not int or type(self.end_byte) is not int:
            raise ValueError("RipgrepSubmatch byte offsets must be integers")
        if self.start_byte < 0 or self.end_byte <= self.start_byte:
            raise ValueError(
                "RipgrepSubmatch byte range must satisfy 0 <= start < end; "
                f"got {self.start_byte}, {self.end_byte}"
            )


@dataclass(frozen=True)
class RipgrepLineEvent:
    """经验证并归属到文件的 match/context line event。"""

    kind: str
    path: str
    line_number: int
    line_text: str
    submatches: tuple[RipgrepSubmatch, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"match", "context"}:
            raise ValueError(f"invalid ripgrep line event kind: {self.kind!r}")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"RipgrepLineEvent.path must be non-empty; got {self.path!r}")
        if type(self.line_number) is not int or self.line_number < 1:
            raise ValueError(
                f"RipgrepLineEvent.line_number must be >= 1; got {self.line_number!r}"
            )
        if not isinstance(self.line_text, str):
            raise ValueError("RipgrepLineEvent.line_text must be str")
        submatches = tuple(self.submatches)
        if any(not isinstance(item, RipgrepSubmatch) for item in submatches):
            raise ValueError("RipgrepLineEvent.submatches must contain RipgrepSubmatch")
        if self.kind == "match" and not submatches:
            raise ValueError("match event must contain at least one submatch")
        if self.kind == "context" and submatches:
            raise ValueError("context event must not contain submatches")
        object.__setattr__(self, "submatches", submatches)


@dataclass(frozen=True)
class RipgrepRunResult:
    command: tuple[str, ...]
    returncode: int
    stderr: str
    rg_source: str
    events: tuple[RipgrepLineEvent, ...]
    grouped_events: Mapping[str, tuple[RipgrepLineEvent, ...]] = field(
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.returncode not in (0, 1):
            raise ValueError("RipgrepRunResult returncode must be 0 or 1")
        if not isinstance(self.rg_source, str) or not self.rg_source:
            raise ValueError("RipgrepRunResult.rg_source must be non-empty")
        events = tuple(self.events)
        groups = {
            path: tuple(items) for path, items in dict(self.grouped_events).items()
        }
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "grouped_events", MappingProxyType(groups))


def resolve_rg_path(
    repository_root: str | Path | None = None,
    *,
    anchors: Sequence[str] = (),
    search_scope: Sequence[str] = (),
) -> tuple[str, str]:
    """Windows 优先 ``tools/rg.exe``，否则使用 PATH；缺失时抛结构化错误。"""

    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    if platform.system() == "Windows":
        bundled = root / "tools" / "rg.exe"
        if bundled.is_file():
            return str(bundled), "bundled"
    system_rg = shutil.which("rg")
    if system_rg:
        return system_rg, "system PATH"
    raise RipgrepNotFoundError(
        "ripgrep executable was not found",
        command=(),
        returncode=None,
        stderr="bundled tools/rg.exe and system PATH lookup both failed",
        anchors=anchors,
        search_scope=search_scope,
    )


def check_rg_proc(
    process: subprocess.CompletedProcess[str],
    *,
    command: Sequence[str],
    anchors: Sequence[str],
    search_scope: Sequence[str],
) -> None:
    """保留上游三态返回码：0 命中、1 无命中、>=2 执行错误。"""

    if process.returncode in (0, 1):
        return
    raise RipgrepExecutionError(
        "ripgrep execution failed",
        command=command,
        returncode=process.returncode,
        stderr=process.stderr or "",
        anchors=anchors,
        search_scope=search_scope,
    )


def _output_error(
    message: str,
    *,
    command: Sequence[str],
    returncode: int,
    stderr: str,
    anchors: Sequence[str],
    search_scope: Sequence[str],
) -> InvalidRipgrepOutputError:
    return InvalidRipgrepOutputError(
        message,
        command=command,
        returncode=returncode,
        stderr=stderr,
        anchors=anchors,
        search_scope=search_scope,
    )


def parse_rg_json_events(
    stdout: str,
    *,
    command: Sequence[str],
    returncode: int,
    stderr: str,
    anchors: Sequence[str],
    search_scope: Sequence[str],
) -> tuple[tuple[RipgrepLineEvent, ...], Mapping[str, tuple[RipgrepLineEvent, ...]]]:
    """解析并分组 begin/match/context/end，坏 JSON 或 schema 漂移立即失败。"""

    if not isinstance(stdout, str):
        raise _output_error(
            "ripgrep stdout must be UTF-8 text",
            command=command,
            returncode=returncode,
            stderr=stderr,
            anchors=anchors,
            search_scope=search_scope,
        )
    events: list[RipgrepLineEvent] = []
    groups: dict[str, list[RipgrepLineEvent]] = {}
    current_path: str | None = None
    for line_index, raw in enumerate(stdout.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _output_error(
                f"malformed ripgrep JSON event at output line {line_index}: {exc}",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            ) from exc
        if not isinstance(value, Mapping):
            raise _output_error(
                f"ripgrep event at output line {line_index} is not an object",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            )
        event_type = value.get("type")
        data = value.get("data")
        if event_type == "summary":
            continue
        if event_type not in {"begin", "match", "context", "end"} or not isinstance(
            data, Mapping
        ):
            raise _output_error(
                f"unsupported or incomplete ripgrep event at output line {line_index}",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            )
        path_value = data.get("path")
        path = path_value.get("text") if isinstance(path_value, Mapping) else None
        if event_type == "begin":
            if current_path is not None:
                raise _output_error(
                    "ripgrep begin event appeared before the current group ended",
                    command=command,
                    returncode=returncode,
                    stderr=stderr,
                    anchors=anchors,
                    search_scope=search_scope,
                )
            if not isinstance(path, str) or not path:
                raise _output_error(
                    "ripgrep begin event has no text path",
                    command=command,
                    returncode=returncode,
                    stderr=stderr,
                    anchors=anchors,
                    search_scope=search_scope,
                )
            current_path = path
            groups.setdefault(path, [])
            continue
        if event_type == "end":
            if current_path is None or not isinstance(path, str) or not path:
                raise _output_error(
                    "ripgrep end event has no active group or text path",
                    command=command,
                    returncode=returncode,
                    stderr=stderr,
                    anchors=anchors,
                    search_scope=search_scope,
                )
            if path != current_path:
                raise _output_error(
                    "ripgrep end event path does not match current group",
                    command=command,
                    returncode=returncode,
                    stderr=stderr,
                    anchors=anchors,
                    search_scope=search_scope,
                )
            current_path = None
            continue
        if not isinstance(path, str) or not path:
            raise _output_error(
                f"ripgrep {event_type} event has no text path",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            )
        if current_path is None:
            raise _output_error(
                f"ripgrep {event_type} event appeared outside a begin/end group",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            )
        if path != current_path:
            raise _output_error(
                f"ripgrep {event_type} event path does not match current group",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            )
        lines = data.get("lines")
        line_text = lines.get("text") if isinstance(lines, Mapping) else None
        line_number = data.get("line_number")
        if not isinstance(line_text, str) or type(line_number) is not int or line_number < 1:
            raise _output_error(
                f"ripgrep {event_type} event has invalid line data",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            )
        parsed_submatches: list[RipgrepSubmatch] = []
        raw_submatches = data.get("submatches", [])
        if not isinstance(raw_submatches, list):
            raise _output_error(
                "ripgrep submatches must be a list",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            )
        for raw_submatch in raw_submatches:
            if not isinstance(raw_submatch, Mapping):
                raise _output_error(
                    "ripgrep submatch is not an object",
                    command=command,
                    returncode=returncode,
                    stderr=stderr,
                    anchors=anchors,
                    search_scope=search_scope,
                )
            match = raw_submatch.get("match")
            match_text = match.get("text") if isinstance(match, Mapping) else None
            try:
                parsed_submatches.append(
                    RipgrepSubmatch(
                        text=match_text,  # type: ignore[arg-type]
                        start_byte=raw_submatch.get("start"),  # type: ignore[arg-type]
                        end_byte=raw_submatch.get("end"),  # type: ignore[arg-type]
                    )
                )
            except ValueError as exc:
                raise _output_error(
                    f"invalid ripgrep submatch: {exc}",
                    command=command,
                    returncode=returncode,
                    stderr=stderr,
                    anchors=anchors,
                    search_scope=search_scope,
                ) from exc
        try:
            event = RipgrepLineEvent(
                kind=event_type,
                path=path,
                line_number=line_number,
                line_text=line_text,
                submatches=tuple(parsed_submatches),
            )
        except ValueError as exc:
            raise _output_error(
                f"invalid ripgrep line event: {exc}",
                command=command,
                returncode=returncode,
                stderr=stderr,
                anchors=anchors,
                search_scope=search_scope,
            ) from exc
        events.append(event)
        groups.setdefault(path, []).append(event)
    if current_path is not None:
        raise _output_error(
            "ripgrep output ended before the current group ended",
            command=command,
            returncode=returncode,
            stderr=stderr,
            anchors=anchors,
            search_scope=search_scope,
        )
    if returncode == 0 and not any(event.kind == "match" for event in events):
        raise _output_error(
            "ripgrep returned success without a match event",
            command=command,
            returncode=returncode,
            stderr=stderr,
            anchors=anchors,
            search_scope=search_scope,
        )
    frozen_groups = MappingProxyType(
        {path: tuple(items) for path, items in groups.items()}
    )
    return tuple(events), frozen_groups


Runner = Callable[..., subprocess.CompletedProcess[str]]


class RipgrepExecutor:
    """参数列表执行 rg；默认 fixed-string，可显式切 regex，并保留 context/glob。"""

    def __init__(
        self,
        *,
        rg_path: str | None = None,
        rg_source: str | None = None,
        repository_root: str | Path | None = None,
        runner: Runner | None = None,
    ) -> None:
        if rg_path is not None and (not isinstance(rg_path, str) or not rg_path.strip()):
            raise ValueError(f"rg_path must be None or non-empty; got {rg_path!r}")
        self._rg_path = rg_path
        self._rg_source = rg_source or ("explicit" if rg_path is not None else None)
        self.repository_root = repository_root
        self._runner = runner or subprocess.run

    @property
    def rg_source(self) -> str:
        return self._rg_source or "unresolved"

    def _resolve(
        self,
        anchors: Sequence[str],
        search_scope: Sequence[str],
    ) -> tuple[str, str]:
        if self._rg_path is not None:
            return self._rg_path, self._rg_source or "explicit"
        path, source = resolve_rg_path(
            self.repository_root,
            anchors=anchors,
            search_scope=search_scope,
        )
        self._rg_path = path
        self._rg_source = source
        return path, source

    def search(
        self,
        terms: Sequence[str],
        targets: Sequence[str | Path],
        *,
        context: int = 0,
        regex: bool = False,
        globs: Sequence[str] = (),
    ) -> RipgrepRunResult:
        anchors = _strings(terms, "terms")
        if isinstance(targets, (str, bytes, Path)):
            raise ValueError("targets must be a sequence of paths")
        try:
            target_strings = tuple(str(target) for target in targets)
        except TypeError as exc:
            raise ValueError("targets must be a sequence of paths") from exc
        _strings(target_strings, "targets")
        glob_values = _strings(globs, "globs", allow_empty=True)
        if type(context) is not int or context < 0:
            raise ValueError(f"context must be an integer >= 0; got {context!r}")
        if type(regex) is not bool:
            raise ValueError(f"regex must be bool; got {regex!r}")
        rg_path, source = self._resolve(anchors, target_strings)
        command = [rg_path, "--json", "-i", "-C", str(context)]
        for glob in glob_values:
            command.extend(["--glob", glob])
        if not regex:
            command.append("--fixed-strings")
        for term in anchors:
            command.extend(["-e", term])
        command.append("--")
        command.extend(target_strings)
        try:
            process = self._runner(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                shell=False,
            )
        except FileNotFoundError as exc:
            raise RipgrepNotFoundError(
                "ripgrep executable disappeared before execution",
                command=command,
                returncode=None,
                stderr=str(exc),
                anchors=anchors,
                search_scope=target_strings,
            ) from exc
        except OSError as exc:
            raise RipgrepExecutionError(
                "ripgrep process could not be started",
                command=command,
                returncode=None,
                stderr=str(exc),
                anchors=anchors,
                search_scope=target_strings,
            ) from exc
        check_rg_proc(
            process,
            command=command,
            anchors=anchors,
            search_scope=target_strings,
        )
        events, groups = parse_rg_json_events(
            process.stdout or "",
            command=command,
            returncode=process.returncode,
            stderr=process.stderr or "",
            anchors=anchors,
            search_scope=target_strings,
        )
        return RipgrepRunResult(
            command=tuple(command),
            returncode=process.returncode,
            stderr=process.stderr or "",
            rg_source=source,
            events=events,
            grouped_events=groups,
        )
