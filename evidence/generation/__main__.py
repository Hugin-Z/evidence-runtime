"""``python -m evidence.generation`` —— L6 生成编排命令行。

::

    # 1) 先产 brief（gap_checker --brief），再起编排：写任务包 + 报告待撰写清单
    python -m evidence.generation --brief out/generation-brief.json --session-dir out/gen \
        [--spec spec.json]

    # 2) agent 或人工按 out/gen/tasks/*.task.md 撰写，回稿存
    #    out/gen/responses/<section_id>.response.json

    # 3) 重跑同一命令 → 收货校验；接受/退回（退回见 out/gen/rejections/）；全部就绪则产方案稿

**Engine 零 LLM**：本命令只做组装/派发/收货，撰写由交互 agent 完成。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .executor import InteractiveFileExecutor
from .orchestrator import Orchestrator
from .section_spec import load_section_specs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evidence.generation",
        description="L6 生成编排：任务派发 + 收货校验（交互执行方）",
    )
    parser.add_argument("--brief", required=True, help="GenerationBrief JSON（gap_checker --brief 产）")
    parser.add_argument("--session-dir", required=True, help="编排会话目录（任务包/回稿/状态）")
    parser.add_argument("--spec", help="ProjectSpec JSON（可选；取参数面注入任务包）")
    parser.add_argument("--doc-type", default="proposal-document", help="section_specs 的 doc_type")
    parser.add_argument("--out", help="全部就绪时输出的方案稿 markdown 路径")
    parser.add_argument("--reset-section", action="append", default=[], metavar="SECTION_ID",
                        help="强制重跑该 section（无视旧终态/回稿指纹）；可多次用于改稿后重新收货")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        project_params: dict = {}
        if args.spec:
            from ..intake import load_project_spec
            spec = load_project_spec(args.spec)
            project_params = {
                "project_name": spec.project_name,
                "client_name": spec.client_name,
                "industry_domain": spec.industry_domain,
                "project_number": spec.project_number,
                "budget": spec.budget,
                "duration": spec.duration,
            }
        orch = Orchestrator(
            brief=brief,
            section_specs=load_section_specs(args.doc_type),
            project_params=project_params,
            session_dir=args.session_dir,
            reset_sections=args.reset_section,
        )
        state = orch.run(InteractiveFileExecutor(args.session_dir))
    except (OSError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # 报告 + 退回清单落盘
    rej_dir = Path(args.session_dir) / "rejections"
    for sid in state.rejected():
        rej_dir.mkdir(parents=True, exist_ok=True)
        oc = state.outcomes[sid]
        (rej_dir / f"{sid}.rejection.md").write_text(
            f"# {sid} 退回（第 {oc.attempts} 次）\n\n" +
            "\n".join(f"- {c}: {d}" for c, d in oc.problems) +
            "\n\n修正后重跑本命令即可重新收货。\n",
            encoding="utf-8",
        )

    # 汇总各 section 任务包的 prompt-injection 告警；只标注，不改材料。
    tasks_dir = Path(args.session_dir) / "tasks"
    total_warn = 0
    if tasks_dir.is_dir():
        for tf in sorted(tasks_dir.glob("*.task.json")):
            try:
                ws = json.loads(tf.read_text(encoding="utf-8")).get("injection_warnings", [])
            except (OSError, ValueError):
                ws = []
            total_warn += len(ws)
    if total_warn:
        print(f"⚠ prompt-injection 告警 {total_warn} 条（材料含疑似指令；已隔离进数据区+标注，"
              f"未改材料；见 tasks/*.task.md 的告警节）。收货与判定不受其影响。")

    print(f"sections: {len(state.order)}")
    for sid in state.order:
        oc = state.outcomes[sid]
        tag = {"accepted": "✓", "placeholder": "▷", "awaiting": "…",
               "rejected": "✗", "failed": "✗✗"}.get(oc.status, "?")
        extra = ""
        if oc.status == "rejected":
            extra = f"  → rejections/{sid}.rejection.md（第 {oc.attempts} 次）"
        elif oc.status == "awaiting":
            extra = f"  → 撰写 tasks/{sid}.task.md，回稿存 responses/{sid}.response.json"
        print(f"  {tag} {sid} {oc.section_title} [{oc.status}]{extra}")

    if state.settled():
        md = orch.assemble_markdown(state)
        out = args.out or str(Path(args.session_dir) / "proposal.md")
        Path(out).write_text(md, encoding="utf-8")
        print(f"\n全部就绪（settled），方案稿：{out}")
        print("次一步：claim audit → 交付准入 → render。方案稿尚未获交付准入，勿直接当交付版。")
        return 0
    # 非终态：明确打印 section 状态并返回非零，供自动化区分。
    unsettled = state.unsettled_sections()
    print(f"\n未就绪（NOT settled）：{len(unsettled)} 个 section 未到终态——")
    for sid, status in unsettled:
        print(f"  ✗ {sid} [{status}]")
    print("撰写/修正后重跑；未就绪不得进入渲染交付版。")
    return 1


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
