# vector_corpus — 向量路测试语料

## 来源与用途
合成语料，配合 **test-only 手写概念 embedder**（见 `tests/conftest.py` 的
`FakeConceptEmbedder`）验证向量路的存在理由:**召回"语义相关但字面不同"的文档**。

> 实际被 ingest 的语料文件都在 **`data/` 子目录**；本 README 是 meta 声明，
> 刻意放在 `data/` 外，**不入库**（否则它会把自己也嵌进向量库污染断言）。

- `data/retry_doc.md`   — 字面含 `retry`/`strategy`/`timeout`（词法路也能命中）+ 可靠性概念
- `data/backoff_doc.md` — **字面不含 `retry`**，只有 `backoff`/`requeue`/`redeliver`;语义同属
  "重试/可靠性"。这是**语义召回目标**:向量路应召回它，而 ripgrep 漏掉它
- `data/gardening_doc.md` — 无关主题（园艺），远负例

## ⚠️ 这套测的是"缝接通了"，不是"真模型语义质量"
`FakeConceptEmbedder` 是**人工写定的确定性概念向量**（把
`retry≈backoff≈requeue≈timeout` 映射到同一概念维度）。这样做的唯一目的，是让
"查 retry 能召回无 literal-retry 的 backoff 文档"成为**离线可复现的机械事实**，
从而 fixture-first 可断言。

它**不代表真模型的语义质量**。真实语义检索质量由生产默认的 embedding 后端
（默认 API/LM Studio，`core/embeddings.load_embedder`；可切本地 bge-m3）决定，**不在本测试断言范围内**。

## 只断言机械事实（刻意不断言）
- 断言:某文件**是否出现**在召回结果里、scope 过滤是否生效、一致性守卫是否 raise
- **不断言**:相关度排序 / "哪条是最佳答案"（语义排序留给 `rerank`，本测试不碰）
- **不断言**:真 bge-m3 的召回质量（离线跑不动真模型，不在此测）

## 局限
朴素切块（按行 + 字符预算 + 行重叠，非 token 级、无 CJK 特化分句），
且未做 contextual enrichment。
