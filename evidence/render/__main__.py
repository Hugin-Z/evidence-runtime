"""``python -m evidence.render`` —— L6′ 渲染命令行。

::

    # 编排会话（含 state.json）→ docx + verify 报告
    python -m evidence.render --session-dir out/gen \
        --brief out/gap-report/generation-brief.json \
        --spec spec.json --out out/proposal.docx [--audit]

交付版默认不显证据来源；``--audit`` 出审计版（每段下方显去标识来源）。
渲染后**立即 verify_docx 回验**，不过则拒收（退出码 3）并打印差异清单。

**交付准入闸**：交付版要求会话 settled，并同时传 ``--recheck``、``--audit-brief`` 与
``--audit-map``。渲染门会用当前证据工件重跑规则、覆盖与处置检查；任一不一致/未就绪 → 拒绝
（退出码 4）。可加 ``--draft`` 渲染带水印草稿版（供内部复核，非交付物）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from ..generation.orchestrator import OrchestrationState
from .docx_render import RenderError, render_proposal
from .verify_docx import load_docx, verify_docx


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evidence.render",
        description="L6′ 渲染：结构化方案稿 → DOCX + verify_docx 回验",
    )
    parser.add_argument("--session-dir", required=True, help="生成编排会话目录（含 state.json）")
    parser.add_argument("--brief", required=True, help="GenerationBrief JSON（取 ref→source）")
    parser.add_argument("--spec", help="ProjectSpec JSON（取标题页参数）")
    parser.add_argument("--out", required=True, help="输出 docx 路径")
    parser.add_argument("--audit", action="store_true", help="审计版：每段下方显去标识来源")
    parser.add_argument("--recheck", help="Claim Audit 复核报告 JSON（交付版必需；其缓存结论会被复算）")
    parser.add_argument("--audit-brief", help="Claim Audit 使用的 AuditBrief JSON（交付版必需）")
    parser.add_argument("--audit-map", help="generation-brief.audit.json（交付版必需；用于复算引用）")
    parser.add_argument(
        "--delivery-weak-policy",
        choices=["block", "allow"],
        default="block",
        help="渲染部署方的 weak 策略（默认 block）；与复核报告声明取更严者",
    )
    parser.add_argument("--draft", action="store_true",
                        help="渲染带水印草稿版（不设交付准入闸；非交付物）")
    parser.add_argument("--no-header", action="store_true",
                        help="不在页眉显示项目名（交付版默认显示）")
    parser.add_argument("--state-suffix", action="store_true",
                        help="按交付态给输出文件名加后缀（-deliverable / -shareable-draft / -draft）")
    return parser


def _params_from_spec(spec_path: str | None, brief: dict) -> dict:
    params = {"doc_type_zh": brief.get("meta", {}).get("doc_type_zh")}
    if spec_path:
        from ..intake import load_project_spec
        spec = load_project_spec(spec_path)
        params.update({
            "project_name": spec.project_name, "client_name": spec.client_name,
            "project_number": spec.project_number, "industry_domain": spec.industry_domain,
            "doc_type_zh": params.get("doc_type_zh"),
        })
    return params


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    from ..integrity import DeliveryGateError
    try:
        state_dict = json.loads((Path(args.session_dir) / "state.json").read_text(encoding="utf-8"))
        try:
            # from_dict 校验正文与哈希；只改正文会触发 HashMismatchError。
            state = OrchestrationState.from_dict(state_dict)
        except DeliveryGateError as exc:
            print(f"拒绝渲染（退出码 4 · {exc.code}）：{exc}", file=sys.stderr)
            return 4
        brief = json.loads(Path(args.brief).read_text(encoding="utf-8"))
        params = _params_from_spec(args.spec, brief)
        # 公开 CLI 与 Python API 共用 render_proposal 单门；不在 CLI 预先读 admitted 布尔。
        recheck = json.loads(Path(args.recheck).read_text(encoding="utf-8")) if args.recheck else None
        audit_brief = (
            json.loads(Path(args.audit_brief).read_text(encoding="utf-8"))
            if args.audit_brief else None
        )
        audit_map = (
            json.loads(Path(args.audit_map).read_text(encoding="utf-8"))
            if args.audit_map else None
        )
        try:
            doc = render_proposal(
                state,
                brief=brief,
                params=params,
                recheck=recheck,
                audit_brief=audit_brief,
                audit_map=audit_map,
                deployment_weak_policy=args.delivery_weak_policy,
                audit_mode=args.audit,
                draft=args.draft,
                include_header=not args.no_header,
            )
        except DeliveryGateError as exc:
            print(f"拒绝渲染交付版（退出码 4 · {exc.code}）：{exc}", file=sys.stderr)
            return 4
        except RenderError as exc:
            print(f"拒绝渲染交付版（退出码 4）：{exc}", file=sys.stderr)
            return 4
        # 校验通过后再据当前 state + recheck 推导交付态，用于命名与输出说明。
        from ..delivery import DeliveryState, delivery_state
        dstate = (DeliveryState.UNSETTLED if args.draft
                  else delivery_state(state, recheck))
        out_path = Path(args.out)
        if args.state_suffix:
            suffix = {DeliveryState.DELIVERABLE: "-deliverable",
                      DeliveryState.SHAREABLE_DRAFT: "-shareable-draft"}.get(
                          dstate, "-draft" if args.draft else "")
            out_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(out_path))
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # 渲染后立即回验（A3）；草稿版不要求 settled；审计版跳过内部术语隔离扫描
    result = verify_docx(load_docx(str(out_path)), state, params=params,
                         require_settled=not args.draft, audit_mode=args.audit)
    kind = {DeliveryState.DELIVERABLE: "交付版（DELIVERABLE）",
            DeliveryState.SHAREABLE_DRAFT: "进度草稿（SHAREABLE_DRAFT·含待补料·非交付物）"}.get(
                dstate, "草稿版（水印·非交付物）" if args.draft else "审计版")
    if args.audit and not args.draft:
        kind = "审计版（含内部诊断）"
    print(f"rendered: {out_path}  （{kind}）")
    if result.accepted:
        print("verify_docx: ✓ 内容/完整性/占位/参数/就绪/交付态/术语隔离 全部一致")
        return 0
    print("verify_docx: ✗ 拒收，差异清单：", file=sys.stderr)
    print(result.diff_summary(), file=sys.stderr)
    return 3


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(main())
