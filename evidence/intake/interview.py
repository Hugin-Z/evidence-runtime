"""L0 Intake 执行形态：交互模式与文件模式都产出 ``ProjectSpec``。

- **交互模式** :func:`run_interactive` —— CLI 逐问逐答（人或在场 agent 回答），
  依赖不满足的题自动跳过，枚举就地提示合法值。
- **文件模式** :func:`run_from_file` —— 预填 answers 文件直接提交（agent/自动化，也是测试形态）。

两条路都走同一个收货校验 :func:`evidence.intake.validate.validate_answers`（A3），
因此"人现场答"与"文件提交"接受的是同一套规则，不会有第二套判定。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

from .questions import QUESTIONS, Question
from .spec import ProjectSpec
from .validate import load_answers_file, validate_answers


class IntakeAborted(RuntimeError):
    """用户在提交前复核时放弃了本次 intake。"""


def _prompt_line(question: Question) -> str:
    bits = [question.text]
    if question.choices:
        bits.append(f"[{'/'.join(question.choices)}]")
    if not question.required:
        bits.append("(可留空)")
    return " ".join(bits) + " > "


def collect_interactive(
    *,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
) -> dict:
    """逐问收集 answers（不校验；校验交 :func:`validate_answers`）。可注入 reader/writer 以便测试。"""

    answers: dict[str, object] = {}
    for question in QUESTIONS:
        if not question.is_active(answers):
            continue
        if question.help:
            writer(f"  · {question.help}")
        raw = reader(_prompt_line(question)).strip()
        if raw == "":
            continue
        if question.type == "string_list":
            answers[question.id] = [item.strip() for item in raw.split(",") if item.strip()]
        else:
            answers[question.id] = raw
    return answers


def _prune_inactive(answers: dict) -> None:
    """依赖失活的答案清掉（如 section_scope 由 custom 改回 all 后，section_ids 作废）。"""

    for question in QUESTIONS:
        if question.id in answers and not question.is_active(answers):
            answers.pop(question.id, None)


def render_answers_summary(answers: Mapping[str, object]) -> str:
    """提交前复核：把已答项按问题顺序列成带编号的清单。"""

    lines = ["", "── 请复核你的回答（提交前）──"]
    for index, question in enumerate(QUESTIONS):
        if question.id not in answers:
            continue
        value = answers[question.id]
        shown = "，".join(value) if isinstance(value, list) else value
        lines.append(f"  [{index}] {question.text}  →  {shown}")
    return "\n".join(lines)


def review_answers(
    answers: dict,
    *,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
) -> dict:
    """提交前复核：展示全部答案 → 确认提交、按编号返改或放弃。

    返回复核后的 answers（可能被编辑）；用户放弃则抛 :class:`IntakeAborted`。
    """

    while True:
        writer(render_answers_summary(answers))
        choice = reader(
            "确认提交？[回车=提交 / 输入编号=改该项 / q=放弃] > "
        ).strip()
        if choice == "" or choice.lower() in {"y", "yes"}:
            return answers
        if choice.lower() in {"q", "quit"}:
            raise IntakeAborted("用户在提交前复核时放弃了本次 intake")
        try:
            question = QUESTIONS[int(choice)]
        except (ValueError, IndexError):
            writer("  无效输入：请输入清单里的编号、回车、或 q。")
            continue
        if question.help:
            writer(f"  · {question.help}")
        raw = reader(_prompt_line(question)).strip()
        if raw == "":
            answers.pop(question.id, None)                 # 清空 = 留空
        elif question.type == "string_list":
            answers[question.id] = [x.strip() for x in raw.split(",") if x.strip()]
        else:
            answers[question.id] = raw
        _prune_inactive(answers)                            # 编辑可能改变依赖，随即修剪


def run_interactive(
    *,
    reader: Callable[[str], str] = input,
    writer: Callable[[str], None] = print,
    review: bool = True,
) -> ProjectSpec:
    """交互模式：逐问逐答 →（可选）提交前复核 → 收货校验 → ProjectSpec。"""

    answers = collect_interactive(reader=reader, writer=writer)
    if review:
        answers = review_answers(answers, reader=reader, writer=writer)
    return validate_answers(answers)


def run_from_file(answers_path: str) -> ProjectSpec:
    """文件模式：读预填 answers → 收货校验 → ProjectSpec。"""

    return validate_answers(load_answers_file(answers_path))


def save_spec(spec: ProjectSpec, path: str | Path) -> Path:
    """把 ProjectSpec 落盘为 JSON（项目库内可复用可修改）。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
