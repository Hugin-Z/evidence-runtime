#!/usr/bin/env bash
# 缺料 demo 全链脚本：外部读者可直接运行，CI 也会调用。
#
# **不复制预制 state**：真正走 生成任务包 → 放置示例回稿 → 收货（盖章）→ 审计（绑定）→ 渲染。
# 每步都**按预期返回码判定**（gap_checker 对 incomplete 返回 1 是设计行为，不是失败）。
#
# 用法：run_chain.sh [PYTHON] [WORKDIR] [DEMO_DIR]
#   PYTHON    默认 "python"（可传 venv 内的 python，用于 wheel 离仓验证）
#   WORKDIR   默认 mktemp；中间产物落此
#   DEMO_DIR  默认脚本所在目录（examples/demo-proposal）；含 materials/ answers.json expected/
set -u
PY="${1:-python}"
WORK="${2:-$(mktemp -d)}"
DEMO="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

fail() { echo "FAIL: $*" >&2; exit 1; }
# expect_rc <expected> <label> -- <cmd...>
expect_rc() {
  local want="$1" label="$2"; shift 2; [ "$1" = "--" ] && shift
  "$@"; local got=$?
  [ "$got" = "$want" ] || fail "$label: 返回码 $got，预期 $want"
  echo "  ✓ $label (rc=$got as expected)"
}

echo "demo full chain · PY=$PY · WORK=$WORK · DEMO=$DEMO"
mkdir -p "$WORK"

# L0 intake → spec
expect_rc 0 "L0 intake" -- "$PY" -m evidence.intake --answers "$DEMO/answers.json" --out "$WORK/spec.json"

# L1 library（每项目一库）
expect_rc 0 "L1 library init" -- "$PY" -m evidence.library --root "$WORK/lib" --project demo init
for f in "$DEMO"/materials/*; do
  expect_rc 0 "L1 add $(basename "$f")" -- "$PY" -m evidence.library --root "$WORK/lib" --project demo add "$f" --source-type self_produced
done

# L2–L5 判定 + brief（fake embedder；**demo 故意缺案例 → incomplete → 返回码 1 是设计行为**）
expect_rc 1 "L2-L5 gap (incomplete by design)" -- "$PY" -m evidence.gap_checker \
  --spec "$WORK/spec.json" --library-root "$WORK/lib" --output-dir "$WORK/gap" \
  --fake-embedder --brief 解决方案建议书

# L6 生成：pass1 无回稿 → awaiting → 返回码 1（未就绪）
expect_rc 1 "L6 generation pass1 (awaiting)" -- "$PY" -m evidence.generation \
  --brief "$WORK/gap/generation-brief.json" --session-dir "$WORK/gen" --spec "$WORK/spec.json"

# 放置示例回稿（agent 撰写的产物；demo 提供）
mkdir -p "$WORK/gen/responses"
cp "$DEMO"/expected/gen/responses/*.response.json "$WORK/gen/responses/" || fail "缺示例回稿"

# L6 生成：pass2 真收货 → 盖章 state → settled → 返回码 0
expect_rc 0 "L6 generation pass2 (receiving → settled)" -- "$PY" -m evidence.generation \
  --brief "$WORK/gap/generation-brief.json" --session-dir "$WORK/gen" --spec "$WORK/spec.json"

# L7 Claim Audit → 交付准入（用 committed v2 审计单；绑定到本次盖章段落）→ 返回码 0（ADMITTED）
expect_rc 0 "L7 claim audit (admitted)" -- "$PY" -m evidence.claim_audit \
  --brief "$WORK/gap/generation-brief.json" --audit-map "$WORK/gap/generation-brief.audit.json" \
  --session-dir "$WORK/gen" --sheet "$DEMO/expected/audit-sheet-v2-admitted.json" \
  --library-root "$WORK/lib" --project demo --emit-brief "$WORK/audit-brief.json" \
  --out "$WORK/recheck.json"

# L6' 渲染（新鲜 recheck：审的就是这份盖章稿）→ 返回码 0 + 产 docx。
# demo 含待补料占位 → 交付态 = SHAREABLE_DRAFT（--state-suffix 使文件名带 -shareable-draft）。
expect_rc 0 "L6' render (SHAREABLE_DRAFT · 含待补料占位)" -- "$PY" -m evidence.render \
  --session-dir "$WORK/gen" --brief "$WORK/gap/generation-brief.json" \
  --spec "$WORK/spec.json" --recheck "$WORK/recheck.json" \
  --audit-brief "$WORK/audit-brief.json" --audit-map "$WORK/gap/generation-brief.audit.json" \
  --out "$WORK/proposal.docx" --state-suffix
DOCX="$WORK/proposal-shareable-draft.docx"
[ -f "$DOCX" ] || fail "进度草稿 docx 未生成（预期 $DOCX）"
echo "  ✓ docx: $DOCX ($(wc -c < "$DOCX") bytes)"

echo "demo full chain: ALL GREEN"
