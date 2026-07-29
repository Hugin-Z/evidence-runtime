"""L6 编排器：组装 → 派发 → 回收 → 收货校验 → 接受/退回/重试。

**编排器是代码，全程确定性。** 按 crosswalk 顺序逐 section：
- ``allow`` / ``limited`` → 组装 :class:`SectionTask` 派发、收货校验、接受/退回（重试上限 N）；
- ``block`` / ``needs_resolution`` → **不派任务**，直接产占位 + 缺口说明（复用 brief 的
  missing_fields / conflicts，agent 碰不到这些 section）；
- ``out_of_evidence_scope`` → 占位，指明走 drafter L3 领域插件/intake，不经本引擎证据。

**重试上限 N**：原则值 ``MAX_RETRIES = 2``（最多 3 次尝试）。硬约束（引用存在性/逐字/占位/
绑定）是客观的——反复过不了 2 次，多半是任务包欠定义（设计缺口），不该无限磨，判 failed 停下。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ..brief import FORBIDDEN_OUTWARD_KEYS
from .discipline import receiving_discipline_note, task_constraints
from .executor import DraftNotReady, TaskExecutorProtocol
from .receiving import check_draft
from .section_spec import SectionWritingSpec, spec_for
from .task import SectionDraft, SectionTask, TaskEvidence

MAX_RETRIES = 2                                # 原则值：最多 2 次退回重试（3 次尝试）

_GATE_MEANING = {
    "allow": "证据充分，允许正常撰写",
    "limited": "证据可用但不足，必须标注不足、限制结论",
    "block": "证据不足，阻断——不派任务，产占位",
    "needs_resolution": "存在未解决冲突——不派任务，先消解",
}

_RESPONSE_FORMAT = {
    "paragraphs": "每段 {text, span_refs}；陈述事实的段落必须绑定至少一个包内 span_ref",
    "placeholder": "无证据字段写 `【待补充：字段说明】`，该段 span_refs 留空",
    "limited_note": "gate=limited 时必填，标注证据不足之处；否则可为 null",
}


class OrchestrationError(RuntimeError):
    """编排输入不合法（brief 结构错误等）。"""


def assert_task_clean(task: SectionTask) -> None:
    """手段无关红线自检：任务包递归不得含任何内部字段名。违反即抛。"""

    def walk(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                if isinstance(k, str) and k.lower() in FORBIDDEN_OUTWARD_KEYS:
                    raise OrchestrationError(f"任务包泄漏内部字段 {path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(task.to_dict(), "task")


@dataclass
class SectionOutcome:
    section_id: str
    section_title: str
    category: str                 # dispatched / placeholder
    status: str                   # accepted / rejected / awaiting / failed / placeholder
    gate: str | None = None
    attempts: int = 0
    problems: tuple[tuple[str, str], ...] = ()
    placeholder_reason: str | None = None
    draft: SectionDraft | None = None
    # 产出该 outcome 的回稿指纹；重新运行时回稿一变，旧 accepted 就不能复用。
    response_digest: str | None = None

    def problem_text(self) -> str:
        return "; ".join(f"{code}: {detail}" for code, detail in self.problems)

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "section_title": self.section_title,
            "category": self.category,
            "status": self.status,
            "gate": self.gate,
            "attempts": self.attempts,
            "problems": [list(p) for p in self.problems],
            "placeholder_reason": self.placeholder_reason,
            "draft": None if self.draft is None else self.draft.to_dict(),
            "response_digest": self.response_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SectionOutcome":
        return cls(
            section_id=str(data["section_id"]),
            section_title=str(data["section_title"]),
            category=str(data["category"]),
            status=str(data["status"]),
            gate=None if data.get("gate") is None else str(data["gate"]),
            attempts=int(data.get("attempts", 0)),  # type: ignore[arg-type]
            problems=tuple(tuple(p) for p in data.get("problems", [])),  # type: ignore[misc]
            placeholder_reason=(
                None if data.get("placeholder_reason") is None
                else str(data["placeholder_reason"])
            ),
            draft=None if data.get("draft") is None
            else SectionDraft.from_dict(data["draft"]),  # type: ignore[arg-type]
            response_digest=(
                None if data.get("response_digest") is None
                else str(data["response_digest"])
            ),
        )


@dataclass
class OrchestrationState:
    order: list[str]
    outcomes: dict[str, SectionOutcome]

    def awaiting(self) -> list[str]:
        return [s for s in self.order if self.outcomes[s].status == "awaiting"]

    def rejected(self) -> list[str]:
        return [s for s in self.order if self.outcomes[s].status == "rejected"]

    def accepted(self) -> list[str]:
        return [s for s in self.order if self.outcomes[s].status == "accepted"]

    def settled(self) -> bool:
        """交付状态不变量：所有可写 section 均 accepted。

        placeholder（block/needs_resolution/oos，本就不派）算合法终态；
        **failed 不算**——它是"派了却始终收不了货"，属未就绪。任一 awaiting/rejected/failed
        即非终态。此前把 failed 也当终态是缺陷：会让"有 section 收不了货"的会话被误判可交付。
        """

        return all(
            self.outcomes[s].status in {"accepted", "placeholder"}
            for s in self.order
        )

    def unsettled_sections(self) -> list[tuple[str, str]]:
        """非终态 section 及其状态（awaiting/rejected/failed），供 CLI/render 打印未就绪清单。"""

        return [
            (s, self.outcomes[s].status)
            for s in self.order
            if self.outcomes[s].status not in {"accepted", "placeholder"}
        ]

    def to_dict(self) -> dict:
        return {"order": list(self.order),
                "outcomes": {k: v.to_dict() for k, v in self.outcomes.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OrchestrationState":
        # 加载即校验正文与存储哈希一致；只改正文会触发 HashMismatchError。
        # 交付/审计路径一切经此加载，故这是"存储哈希不可信、一切现算"的强制点。
        from ..integrity import assert_state_hashes_consistent
        assert_state_hashes_consistent(data)
        return cls(
            order=[str(x) for x in data.get("order", [])],  # type: ignore[union-attr]
            outcomes={
                str(k): SectionOutcome.from_dict(v)  # type: ignore[arg-type]
                for k, v in dict(data.get("outcomes", {})).items()
            },
        )


class Orchestrator:
    """按 brief 逐 section 组织生成。一个 brief 一个编排器实例。"""

    def __init__(
        self,
        *,
        brief: Mapping[str, object],
        section_specs: Mapping[str, SectionWritingSpec] | None = None,
        project_params: Mapping[str, object] | None = None,
        max_retries: int = MAX_RETRIES,
        session_dir: str | Path | None = None,
        reset_sections: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(brief, Mapping) or "sections" not in brief:
            raise OrchestrationError("brief 必须含 sections")
        self.brief = brief
        self.section_specs = dict(section_specs or {})
        self.project_params = dict(project_params or {})
        self.max_retries = max_retries
        self.session_dir = Path(session_dir) if session_dir is not None else None
        self._sections = list(brief["sections"])  # type: ignore[arg-type]
        # 显式重跑这些 section（``--reset-section``），无视旧终态与回稿指纹。
        self._reset_sections = tuple(reset_sections or ())

    # ── 任务组装 ──────────────────────────────────────────────────────
    def build_task(self, section: Mapping[str, object]) -> SectionTask:
        sid = str(section["section_id"])
        gate = section.get("gate")
        evidence = tuple(
            TaskEvidence(
                span_ref=str(a["ref"]),
                content=str(a["content"]),
                source=str(a.get("source") or "未知来源"),
                supports_fields=tuple(str(x) for x in a.get("supports_fields", [])),
                slot_id=None if a.get("slot_id") is None else str(a["slot_id"]),
            )
            for a in section.get("acquired", [])  # type: ignore[union-attr]
            if a.get("status") == "acquired" and a.get("ref") and a.get("content")
        )
        missing = tuple(dict.fromkeys(
            str(m["field_id"]) for m in section.get("missing_fields", [])  # type: ignore[union-attr]
        ))
        spec = spec_for(self.section_specs, sid)
        task = SectionTask(
            section_id=sid,
            section_title=str(section.get("section_title", "")),
            gate=str(gate),
            gate_meaning=_GATE_MEANING.get(str(gate), ""),
            writing=spec.to_dict(),
            project_params=dict(self.project_params),
            evidence=evidence,
            missing_fields=missing,
            constraints=(),                       # 下面按 gate 填
            response_format=dict(_RESPONSE_FORMAT),
            limited_note_required=(gate == "limited"),
        )
        # 对进入任务包的证据扫描 prompt-injection 信号，只标注、不改材料。
        from ..injection_scan import scan_evidence
        warnings = tuple(w.to_dict() for w in scan_evidence(evidence))
        task = SectionTask(**{**task.__dict__,
                              "constraints": task_constraints(task),
                              "injection_warnings": warnings})
        assert_task_clean(task)                   # 手段无关红线
        return task

    # ── 占位（不派发的 section）────────────────────────────────────────
    def _placeholder(self, section: Mapping[str, object]) -> SectionOutcome:
        sid = str(section["section_id"])
        title = str(section.get("section_title", ""))
        gate = section.get("gate")
        if section.get("out_of_evidence_scope"):
            reason = (
                f"out_of_evidence_scope：{section.get('reason') or '该 section 不由本引擎证据背书'}"
                "，走 drafter L3 领域插件/intake。"
            )
        elif gate == "needs_resolution":
            conflicts = section.get("conflicts", [])
            reason = f"存在未解决冲突（{len(conflicts)} 处），需先消解，不生成结论。"
        else:  # block
            missing = [str(m.get("field_id")) for m in section.get("missing_fields", [])]
            reason = f"证据不足（缺字段：{missing or '全部'}），阻断生成，产占位待补料。"
        return SectionOutcome(
            section_id=sid, section_title=title, category="placeholder",
            status="placeholder", gate=None if gate is None else str(gate),
            placeholder_reason=reason,
        )

    @staticmethod
    def _dispatched_gate(gate: object) -> bool:
        return gate in {"allow", "limited"}

    # ── 主流程 ────────────────────────────────────────────────────────
    def run(
        self,
        executor: TaskExecutorProtocol,
        *,
        state: OrchestrationState | None = None,
    ) -> OrchestrationState:
        """跑一轮：新 section 组装派发；被退回且未超限的 section 重新派发（读最新回稿）。

        ``state`` 给定则由它续跑（内存线程重试次数）；否则从 session_dir 恢复（或空）。
        """

        state = state if state is not None else self._load_state()
        reset = set(self._reset_sections)
        for section in self._sections:
            sid = str(section["section_id"])
            prior = state.outcomes.get(sid)
            # 已终态默认跳过，但回稿变化时必须重新收货；``--reset-section`` 可强制重跑。
            if prior is not None and prior.status in {"accepted", "failed", "placeholder"}:
                if sid in reset:
                    pass  # 显式重置：落到重跑
                elif self._response_changed(executor, sid, prior):
                    pass  # 回稿指纹变了：落到重跑（旧 accepted 失效）
                else:
                    continue  # 已终态且回稿未变，跳过

            if section.get("out_of_evidence_scope") or not self._dispatched_gate(section.get("gate")):
                state.outcomes[sid] = self._placeholder(section)
                continue

            task = self.build_task(section)
            attempts = (prior.attempts if prior and sid not in reset else 0)
            try:
                draft = executor.execute(task)
            except DraftNotReady:
                state.outcomes[sid] = SectionOutcome(
                    section_id=sid, section_title=task.section_title,
                    category="dispatched", status="awaiting", gate=task.gate,
                    attempts=attempts,
                )
                continue

            attempts += 1
            digest = self._response_digest(executor, sid, draft)
            result = check_draft(task, draft)
            if result.accepted:
                from ..integrity import stamp_paragraphs
                # 收货接受时确定性写入段落 id + content hash（完整性指纹，非身份凭证），
                # 随 state 落盘，供 claim 绑定与新鲜度校验。
                stamped = SectionDraft(
                    section_id=draft.section_id,
                    paragraphs=stamp_paragraphs(sid, draft.paragraphs),
                    limited_note=draft.limited_note,
                )
                state.outcomes[sid] = SectionOutcome(
                    section_id=sid, section_title=task.section_title,
                    category="dispatched", status="accepted", gate=task.gate,
                    attempts=attempts, draft=stamped, response_digest=digest,
                )
            elif attempts > self.max_retries:
                state.outcomes[sid] = SectionOutcome(
                    section_id=sid, section_title=task.section_title,
                    category="dispatched", status="failed", gate=task.gate,
                    attempts=attempts, problems=result.problems, draft=draft,
                    response_digest=digest,
                )
            else:
                state.outcomes[sid] = SectionOutcome(
                    section_id=sid, section_title=task.section_title,
                    category="dispatched", status="rejected", gate=task.gate,
                    attempts=attempts, problems=result.problems, draft=draft,
                    response_digest=digest,
                )
        # 保持 order 与 brief 顺序一致
        state.order = [str(s["section_id"]) for s in self._sections]
        self._save_state(state)
        return state

    # ── 回稿变更检测 ─────────────────────────────────────────────────────
    @staticmethod
    def _response_digest(executor: TaskExecutorProtocol, sid: str,
                         draft: SectionDraft) -> str:
        """本次收到的回稿指纹。优先用执行方的原始回稿（文件字节）；退化用 draft 结构。"""
        from ..integrity import response_fingerprint
        fp = getattr(executor, "response_fingerprint", None)
        if callable(fp):
            got = fp(sid)
            if got:
                return got
        return response_fingerprint(draft.to_dict())

    def _response_changed(self, executor: TaskExecutorProtocol, sid: str,
                          prior: SectionOutcome) -> bool:
        """已终态 section 的回稿是否较产出该 outcome 时变了。

        执行方能给出回稿指纹（InteractiveFileExecutor 用回稿文件字节）时按指纹比对；
        prior 无 digest（老 state）时保守视为**未变**（不无谓重跑，保持向后兼容）。
        """
        fp = getattr(executor, "response_fingerprint", None)
        if not callable(fp) or prior.response_digest is None:
            return False
        current = fp(sid)
        return current is not None and current != prior.response_digest

    def run_until_settled(
        self, executor: TaskExecutorProtocol, *, max_passes: int = 10
    ) -> OrchestrationState:
        """给**同步执行方**（Callable/API）用：循环 run() 直到全部终态或用尽 max_passes。

        交互 file 执行方不要用它（会在同一轮把重试次数烧光）——交互靠 agent 编辑回稿后重跑。
        """

        state = self.run(executor)
        passes = 1
        while not state.settled() and passes < max_passes:
            if state.awaiting():
                break  # 同步执行方不该 awaiting；防御性退出
            if not state.rejected():
                break
            state = self.run(executor, state=state)   # 线程状态，累积重试次数
            passes += 1
        return state

    # ── 状态持久化 ────────────────────────────────────────────────────
    def _state_path(self) -> Path | None:
        return None if self.session_dir is None else self.session_dir / "state.json"

    def _load_state(self) -> OrchestrationState:
        path = self._state_path()
        if path is not None and path.is_file():
            return OrchestrationState.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
        return OrchestrationState(order=[], outcomes={})

    def _save_state(self, state: OrchestrationState) -> None:
        path = self._state_path()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(state.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ── 产物：结构化方案稿 ────────────────────────────────────────────
    def assemble_markdown(self, state: OrchestrationState) -> str:
        lines: list[str] = [f"# {self.brief.get('meta', {}).get('doc_type_zh') or '方案稿'}", ""]
        for sid in state.order:
            outcome = state.outcomes[sid]
            lines.append(f"## {sid} {outcome.section_title}")
            if outcome.status == "accepted" and outcome.draft is not None:
                for para in outcome.draft.paragraphs:
                    lines.append(para.text)
                    if para.span_refs:
                        lines.append(f"> 依据：{', '.join(para.span_refs)}")
                    lines.append("")
                if outcome.draft.limited_note:
                    lines.append(f"> ⚠ 证据不足说明：{outcome.draft.limited_note}")
                    lines.append("")
            elif outcome.status == "placeholder":
                lines.append(f"【本节暂缺】{outcome.placeholder_reason}")
                lines.append("")
            else:
                lines.append(f"【未完成：{outcome.status}】{outcome.problem_text()}")
                lines.append("")
        return "\n".join(lines)
