import json

import pytest

from eval.bench.run_rerank import parse_details_file


def test_private_benchmark_handoff_round_trips(tmp_path):
    path = tmp_path / "covid.private.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "bench-private-details-v1",
                "dataset": "covid",
                "split": "dev",
                "metrics": {"rrf": {"recall@10": 0.9}},
                "queries": {
                    "private-qid": {
                        "gold": ["private-pid"],
                        "lexical": [],
                        "vector": ["private-pid"],
                        "rrf": ["private-pid"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    queries, metrics = parse_details_file(str(path))

    assert queries["private-qid"]["rrf"] == ["private-pid"]
    assert metrics["rrf"]["recall@10"] == 0.9


def test_private_benchmark_handoff_rejects_unknown_schema(tmp_path):
    path = tmp_path / "covid.private.json"
    path.write_text(
        json.dumps({"schema_version": "unknown", "queries": {}, "metrics": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bench-private-details-v1"):
        parse_details_file(str(path))
