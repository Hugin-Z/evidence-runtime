"""``python -m evidence.intake`` —— L0 任务入口命令行。

::

    # 交互模式：逐问逐答
    python -m evidence.intake --out spec.json

    # 文件模式：预填 answers 直接提交（agent/自动化/测试）
    python -m evidence.intake --answers answers.json --out spec.json

产出一份 ProjectSpec JSON。随后用 `python -m evidence.gap_checker --spec spec.json …`
驱动全链。**本命令零 LLM、零向量调用**——只做结构化收集与确定性校验。
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .interview import IntakeAborted, run_from_file, run_interactive, save_spec
from .validate import IntakeValidationError


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evidence.intake",
        description="L0 任务入口：结构化 intake → ProjectSpec",
    )
    parser.add_argument(
        "--answers",
        help="预填 answers 文件（JSON 对象：问题 id → 值）；不传 = 交互模式",
    )
    parser.add_argument("--out", required=True, help="ProjectSpec 落盘路径（JSON）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        if args.answers:
            spec = run_from_file(args.answers)
        else:
            spec = run_interactive()
    except IntakeAborted as exc:
        print(f"intake 已放弃：{exc}", file=sys.stderr)
        return 1
    except IntakeValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    target = save_spec(spec, args.out)
    print(f"ProjectSpec written: {target}")
    print(f"  doc_type    : {spec.doc_type}")
    print(f"  project     : {spec.project_name}（{spec.client_name}）")
    print(f"  project_id  : {spec.project_id}")
    print(f"  section     : {spec.section_scope}"
          f"{'' if not spec.section_ids else ' ' + ','.join(spec.section_ids)}")
    print(f"次一步：python -m evidence.gap_checker --spec {target} --output-dir out/gap-report")
    return 0


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
