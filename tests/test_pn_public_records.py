import json
import re
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_COVID_RECORDS = (
    _ROOT / "eval/bench/results/BENCH-ZH-Covid-v2-2026-07-16.md",
    _ROOT / "eval/bench/results/RERANK-ZH-Covid-v2-2026-07-16.md",
    _ROOT / "eval/bench/results/smoke/RERANK-SMOKE-2026-07-16.txt",
)
_AGGREGATE_MARKERS = {
    "BENCH-ZH-Covid-v2-2026-07-16.md": "| RRF（双路） | 0.8933 | 0.9457 | 0.7675 | 0.1800 | 0.8110 |",
    "RERANK-ZH-Covid-v2-2026-07-16.md": "| RERANK（RRF→精排） | 0.9357 | 0.9457 | 0.8710 | 0.1888 | 0.8894 |",
}
def _smoke_payload() -> dict:
    text = _COVID_RECORDS[-1].read_text(encoding="utf-8")
    matches = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    assert len(matches) == 1
    return json.loads(matches[0])


def test_covid_smoke_aggregate_recomputes_from_retained_pairs():
    covid = _smoke_payload()["datasets"]["CovidRetrieval"]
    pairs = covid["pairs"]
    aggregate = covid["aggregate"]
    deltas = [pair["delta"] for pair in pairs]

    assert aggregate["pair_count"] == len(pairs) == 3
    assert aggregate["direction_ok"] == sum(pair["direction_ok"] for pair in pairs)
    assert aggregate["direction_pass"] is all(pair["direction_ok"] for pair in pairs)
    assert aggregate["delta_min"] == min(deltas)
    assert aggregate["delta_max"] == max(deltas)
    assert aggregate["delta_mean"] == sum(deltas) / len(deltas)
    assert aggregate["delta_mean"] == 14.777403513590494


def test_covid_public_records_retain_aggregates_but_no_upstream_details():
    for path in _COVID_RECORDS:
        text = path.read_text(encoding="utf-8")
        assert "2026-07-29" in text

    for path in _COVID_RECORDS[:-1]:
        text = path.read_text(encoding="utf-8")
        assert _AGGREGATE_MARKERS[path.name] in text
        assert "| qid |" not in text
        assert "gold(rel" not in text
        assert "corpus_content_sha256" not in text
        assert "6a24c07dbfed" not in text
        assert re.search(r"\b[0-9a-f]{32}\b", text) is None

    covid = _smoke_payload()["datasets"]["CovidRetrieval"]
    assert covid["upstream_details_retained"] is False
    for pair in covid["pairs"]:
        assert "query" not in pair
        assert "qid" not in pair
        assert "rel_pid" not in pair
        assert "irr_pid" not in pair
        assert "rel_qrels_score" not in pair
        assert pair["pair_id"].startswith("covid-pair-")
        assert isinstance(pair["score_rel"], float)
        assert isinstance(pair["score_irr"], float)
        assert isinstance(pair["delta"], float)


def test_public_smoke_record_does_not_expose_a_local_model_path():
    service = _smoke_payload()["service"]
    assert service["model_file"] == "bge-reranker-v2-m3-Q8_0.gguf"
    assert "model_path" not in service


def test_future_public_bench_snapshot_omits_covid_corpus_fingerprint():
    from eval.bench.run_bench import public_dataset_config_line

    covid = public_dataset_config_line(
        dataset="covid",
        split="dev",
        corpus_count=100_001,
        query_count=949,
        corpus_digest="dataset-derived",
    )
    nfcorpus = public_dataset_config_line(
        dataset="nfcorpus",
        split="test",
        corpus_count=3_633,
        query_count=323,
        corpus_digest="licensed-digest",
    )

    assert "dataset-derived" not in covid
    assert "corpus_content_sha256" not in covid
    assert "licensed-digest" in nfcorpus
