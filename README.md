**English** · [中文](README.zh.md)

# Evidence Runtime

**An evidence-constrained agent workflow for turning project materials into auditable deliverables.**

Evidence Runtime is built for work that starts with source material and ends with a proposal. It manages a
project library, checks whether a fixed document contract is supported, prepares constrained writing tasks,
receives writer responses, runs a claim-audit gate, and renders DOCX output.

It is neither a personal knowledge base nor an embedded language model. It can be used as a standalone
**Evidence Agent**: the repository supplies the workflow, templates, state machine, and deterministic checks;
a host agent or a human performs semantic writing and auditing.

## What it does

| Stage | Runtime | Agent or human |
|---|---|---|
| Intake and library | Builds a `ProjectSpec`; stores project-scoped PDF, DOCX, MD, and TXT files | Confirms task parameters and data permissions |
| Evidence readiness | Retrieves, locates verbatim spans, applies rules, and marks slots `filled / weak / missing` | Resolves missing material and business choices |
| Generation | Emits constrained section tasks; validates and receives responses | Writes the requested sections |
| Claim Audit | Splits sentence-level units; checks coverage; recomputes reference, content, and jurisdiction checks | Submits a claim for each factual unit and assigns its support judgment |
| Delivery | Renders drafts or gated DOCX through one public entry point and verifies the document | Performs final business review and approval |

The package currently ships a fixed evidence contract and proposal templates for Chinese public-sector and
enterprise solution documents. Contracts, crosswalks, and section specs are usable as-is and extensible through
the same protocols.

## What “preventing unsupported writing” means

The system cannot stop a writer from producing an incorrect sentence. It prevents **unaudited content from
being admitted as a deliverable**:

- every non-placeholder factual unit must be covered by an audit claim;
- references must exist, with consistent evidence content and jurisdiction;
- an audit must match the current draft, so an old report cannot approve edited text;
- cached booleans such as `accepted` or `delivery.admitted` cannot bypass recomputation;
- missing evidence remains an explicit `【待补充】` placeholder.

The default `integrity` mode detects stale reports, accidental edits, and incomplete audits. It does **not**
stop a principal with write access to the entire workspace from coordinating all artifacts and self-authorizing.
Strong authorization requires an independent auditor identity and signing infrastructure, which is not yet
implemented. See [Security and data boundaries](docs/SECURITY.md).

## Quick start

Requirements: Python 3.11 or 3.12 and [ripgrep](https://github.com/BurntSushi/ripgrep).

```bash
python -m pip install -e ".[test]"
python -m pytest -q
bash examples/demo-proposal/run_chain.sh
```

The synthetic demo runs the real intake, library, gap analysis, two-pass generation, Claim Audit, and DOCX
rendering path. It intentionally lacks some evidence and therefore produces a marked `SHAREABLE_DRAFT`.
[`demo-proposal-deliverable`](examples/demo-proposal-deliverable/) shows the fully supported
`DELIVERABLE` path. These demos verify workflow behavior, not retrieval accuracy.

## Normal workflow

The package installs six commands:

```text
evidence-intake
evidence-library
evidence-gap-check
evidence-generate
evidence-audit
evidence-render
```

Their normal order is:

```text
Task → ProjectSpec → Project library → Gap Report / GenerationBrief
     → Writer response → Claim Audit → Draft or Deliverable DOCX
```

See the [usage guide](docs/USAGE.md) for complete commands, outputs, and the two-pass handoff.

## Relationship to solution-drafter

[solution-drafter](https://github.com/Hugin-Z/solution-drafter) can act as a specialized **Writer Agent**:
it consumes section tasks from Evidence Runtime and returns structured responses. It is not a runtime dependency;
Claude Code, Codex, another agent, or a human can fill the same role.

Both projects remain independently useful today. A future unified product should compose them through the stable
task/response protocol instead of reducing Evidence Runtime to an internal library.

## Project status

Version `0.1.0` is Alpha. The main workflow and delivery gate are implemented and covered end to end. The project
is suitable for evaluation and controlled pilots, but it is not yet a general enterprise content platform.
Current limitations include a fixed document contract, no OCR, no embedded semantic model, no Web/API service,
and no signed authorization mode. See [status](docs/STATUS.md) and [limitations](docs/LIMITATIONS.md).

## Documentation

- [Documentation index](docs/README.md)
- [Usage](docs/USAGE.md)
- [Architecture and boundaries](docs/DESIGN.md)
- [Security and data boundaries](docs/SECURITY.md)
- [Evaluation](docs/EVALUATION.md)
- [Design decisions](docs/DECISIONS.md)
- [Changelog](CHANGELOG.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Data and publication boundary

Do not commit customer material, project libraries, runtime output, internal acceptance records, or development
diaries. `.library/`, `.local/`, `.internal/`, and `out/` are ignored by default. The public repository contains
synthetic demos, reproducible tests, license-reviewed benchmark records, and user-facing documentation.

If an older clone ever contained private material, deleting it from the current tree is insufficient: review and
clean the reachable Git history before publication.

## License

The code and project documentation use the [MIT License](LICENSE). Third-party datasets retain their own terms;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
