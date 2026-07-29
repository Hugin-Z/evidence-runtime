#!/usr/bin/env bash
# DELIVERABLE demo 全链脚本：材料齐备 → 全 filled、无占位 → DELIVERABLE DOCX。
# 与 examples/demo-proposal（SHAREABLE_DRAFT 草稿套）互补：这套演示"齐了什么样"。
#
# 用法：run_chain.sh [PYTHON] [WORKDIR] [DEMO_DIR]
set -u
PY="${1:-python}"
WORK="${2:-$(mktemp -d)}"
DEMO="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

fail() { echo "FAIL: $*" >&2; exit 1; }
expect_rc() {
  local want="$1" label="$2"; shift 2; [ "$1" = "--" ] && shift
  "$@"; local got=$?
  [ "$got" = "$want" ] || fail "$label: 返回码 $got，预期 $want"
  echo "  ✓ $label (rc=$got as expected)"
}

echo "deliverable demo full chain · PY=$PY · WORK=$WORK · DEMO=$DEMO"
mkdir -p "$WORK"

expect_rc 0 "L0 intake" -- "$PY" -m evidence.intake --answers "$DEMO/answers.json" --out "$WORK/spec.json"
expect_rc 0 "L1 library init" -- "$PY" -m evidence.library --root "$WORK/lib" --project demo-deliverable init
for f in "$DEMO"/materials/*; do
  expect_rc 0 "L1 add $(basename "$f")" -- "$PY" -m evidence.library --root "$WORK/lib" --project demo-deliverable add "$f" --source-type self_produced
done

# L2–L5 判定 + brief：材料齐备 → **complete → 返回码 0**（对比草稿套的 incomplete=1）
expect_rc 0 "L2-L5 gap (complete — all slots filled)" -- "$PY" -m evidence.gap_checker \
  --spec "$WORK/spec.json" --library-root "$WORK/lib" --output-dir "$WORK/gap" \
  --fake-embedder --brief 方案核心内容

expect_rc 1 "L6 generation pass1 (awaiting)" -- "$PY" -m evidence.generation \
  --brief "$WORK/gap/generation-brief.json" --session-dir "$WORK/gen" --spec "$WORK/spec.json"

mkdir -p "$WORK/gen/responses"
cp "$DEMO"/expected/gen/responses/*.response.json "$WORK/gen/responses/" || fail "缺示例回稿"

expect_rc 0 "L6 generation pass2 (receiving → settled)" -- "$PY" -m evidence.generation \
  --brief "$WORK/gap/generation-brief.json" --session-dir "$WORK/gen" --spec "$WORK/spec.json"

expect_rc 0 "L7 claim audit (admitted)" -- "$PY" -m evidence.claim_audit \
  --brief "$WORK/gap/generation-brief.json" --audit-map "$WORK/gap/generation-brief.audit.json" \
  --session-dir "$WORK/gen" --sheet "$DEMO/expected/audit-sheet.json" \
  --library-root "$WORK/lib" --project demo-deliverable --emit-brief "$WORK/audit-brief.json" \
  --out "$WORK/recheck.json"

# L6' 渲染：无占位 → 交付态 = DELIVERABLE（--state-suffix → 文件名带 -deliverable，无水印）
expect_rc 0 "L6' render (DELIVERABLE · 无占位干净交付版)" -- "$PY" -m evidence.render \
  --session-dir "$WORK/gen" --brief "$WORK/gap/generation-brief.json" \
  --spec "$WORK/spec.json" --recheck "$WORK/recheck.json" \
  --audit-brief "$WORK/audit-brief.json" --audit-map "$WORK/gap/generation-brief.audit.json" \
  --out "$WORK/proposal.docx" --state-suffix
DOCX="$WORK/proposal-deliverable.docx"
[ -f "$DOCX" ] || fail "交付版 docx 未生成（预期 $DOCX）"
echo "  ✓ docx: $DOCX ($(wc -c < "$DOCX") bytes)"

echo "deliverable demo full chain: ALL GREEN"
