# lexical_corpus — 词法路测试语料（R10 限制声明）

## 来源
合成语料，按"字面命中"规格人工写定。这是**确定性变换类** fixture：给定
（语料 + 查询），ripgrep 的字面命中 `(文件, 行号, 命中 token)` 有唯一正确解。
每条断言的预期值由人工逐行阅读本目录的文件、对照字面匹配规格推导得出，
**绝不由运行 `RipgrepRetriever` 反喂**（避免与被测函数同源自指）。

> 与 `fixture-first.rule.md`"AI 不许造检索类 anchor"的关系：该禁令防的是
> 输出无唯一正确解的语义/排序 anchor。本语料只断言机械可验证的事实，
> 不在此列。见下"不覆盖"。

## 文件与可预期命中（1-based 行号）
- `alpha.md`   — `retry`@1,2；`strategy`@3；唯一 token `zqxmarker`@4
- `beta.py`    — 标识符 `configure_retry`@1；`retry`@1,2；`strategy`@1,2,3
- `gamma.md`   — `retry`@2,3（相邻两行，用于"合并成行区间"）
- `synonym_miss.md` — 语义相关但**字面不含** `retry`/`strategy`；只有 `backoff`@1,2、
  `requeue`@1,3。负例：证明词法路漏掉同义改写（这是双路 A/B 的证据资产）
- `delta.md`   — 与检索词无关，no-hit 对照
- `ignored_secret.md` + `.ignore` — 含 `retry` 但被 ignore 规则跳过，
  证明默认 ignore 行为被保留

二进制跳过、空目录、缺失目录、`rg` 未安装等场景在 `tests/test_lexical.py` 内
用 `tmp_path` / monkeypatch 确定性生成，不落盘到本语料。

## 覆盖
单关键词精确命中（文件+行号+token）、多关键词初排分（命中数/总数）、相邻命中合并、
停用词剔除、空/缺失/无命中降级、ignore 跳过、二进制跳过、`rg` 缺失降级、`lexical_top_k` 截断。

## 不覆盖（刻意不断言）
- **相关度排序 / "哪条是最佳答案"** —— 那是语义判断，留给 `rerank`，本语料不碰
- 同义词扩展 / CJK 分词 —— 词法路是"哑"检索，按设计不做
