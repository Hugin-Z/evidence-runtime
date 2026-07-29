# Third-Party Notices

The MIT license in [`LICENSE`](LICENSE) covers the project-authored software and
documentation. It does **not** relicense third-party datasets, dataset-derived
text, model weights, or services.

## Benchmark data is not bundled

`eval/bench/data/` is excluded by `.gitignore`. A source distribution or Git
checkout should therefore not contain full NFCorpus or CovidRetrieval corpora,
query files, or qrels. Users who download benchmark data are responsible for
the upstream terms that apply to their copy.

The tracked files under `eval/bench/results/` contain project-authored metrics.
NFCorpus records also contain limited query samples, identifiers, qrels, and
rankings under the terms described below. The current public CovidRetrieval
records contain project-authored measurements only: Markdown records retain
aggregates, while the smoke record retains scores and deltas under opaque
project-local pair labels. They do not contain upstream query text,
query/document identifiers, qrels, gold labels, corpus fingerprints, or
per-query rankings.

## NFCorpus

- Dataset: NFCorpus, distributed through the BEIR benchmark.
- Upstream dataset page:
  <https://huggingface.co/datasets/BeIR/nfcorpus>
- Upstream license metadata: Creative Commons Attribution-ShareAlike 4.0
  International (CC BY-SA 4.0).
- License text: <https://creativecommons.org/licenses/by-sa/4.0/>
- Relevant tracked artifacts: files whose names contain `NFCorpus` under
  `eval/bench/results/`, plus the NFCorpus entries in
  `eval/bench/results/smoke/RERANK-SMOKE-2026-07-16.txt`.
- Original dataset attribution: Vera Boteva, Demian Gholipour, Artem Sokolov,
  and Stefan Riezler, “A Full-Text Learning to Rank Dataset for Medical
  Information Retrieval,” ECIR 2016.
- Requested scholarly attribution: Nandan Thakur, Nils Reimers, Andreas
  Rücklé, Abhishek Srivastava, and Iryna Gurevych, “BEIR: A Heterogeneous
  Benchmark for Zero-shot Evaluation of Information Retrieval Models,” 2021.
  <https://openreview.net/forum?id=wCu6T5xFjeJ>

No endorsement by the dataset authors is implied. If a tracked result is an
adaptation of CC BY-SA material, the CC BY-SA attribution and share-alike
requirements apply to that adapted portion. The result records modify or
augment the cited upstream material with project-generated rankings and
metrics; this notice identifies that change.

## CovidRetrieval / Multi-CPR

- Dataset: CovidRetrieval, a C-MTEB/MTEB retrieval task derived from Multi-CPR.
- Upstream dataset pages:
  <https://huggingface.co/datasets/C-MTEB/CovidRetrieval> and
  <https://huggingface.co/datasets/mteb/CovidRetrieval>
- Source paper: Dingkun Long et al., “Multi-CPR: A Multi Domain Chinese Dataset
  for Passage Retrieval,” 2022. <https://arxiv.org/abs/2203.03367>
- Upstream license metadata checked on 2026-07-29: the MTEB dataset card reports
  the license as **unknown**, and the C-MTEB card does not state a license.
- Relevant tracked artifacts: files whose names contain `ZH-Covid` under
  `eval/bench/results/`, plus the CovidRetrieval block in
  `eval/bench/results/smoke/RERANK-SMOKE-2026-07-16.txt`. The Markdown records retain
  project-authored aggregate metrics. The smoke record retains project-local
  pair labels and project-generated per-pair scores/deltas so its aggregate
  values remain independently reproducible.

An “unknown” license is **not permission**. The repository does not claim that
the MIT license authorizes redistribution of CovidRetrieval-derived material.
On 2026-07-29 the current public tree was reduced to project-authored
measurements: query text/tokens, query/document identifiers, qrels/gold labels,
corpus fingerprints, and per-query rankings were removed from the five relevant
tracked artifacts. Future detailed rerank handoffs created with
`run_bench.py --details-out` are private working artifacts and must remain
ignored or outside the repository. The full CovidRetrieval data must not be
added without a documented redistribution basis.

## Python dependencies

Python dependencies are installed from their respective distributions and are
not copied into this repository. Their own licenses apply. Inspect the resolved
environment or wheel metadata before redistributing a bundled environment.
