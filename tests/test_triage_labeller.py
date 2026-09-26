"""Sharding, per-model call parameters and the cache-only merge.

No network: the merge path reads cache files the test writes itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage import labeller  # noqa: E402
from scripts.triage.corpus import read_jsonl, write_jsonl  # noqa: E402
from scripts.triage.prompts import PROMPT_VERSION  # noqa: E402


def _sha(i: int) -> str:
    """A realistic document hash — real sha256 digests, not a counter."""
    import hashlib

    return hashlib.sha256(f"doc-{i}".encode()).hexdigest()


LABEL = {
    "alert_class": "A3",
    "alert_class_confidence": "high",
    "priority": "P2",
    "primary_functions": ["Regulatory change"],
    "secondary_functions": [],
    "lines_of_defence": [2],
    "summary": "s",
    "key_dates": [],
    "applicability": [],
    "obligations_present": False,
    "rationale": "r",
}


# ── sharding ────────────────────────────────────────────────────────────────


def test_shard_is_deterministic_disjoint_and_covers_everything() -> None:
    hashes = [_sha(i) for i in range(500)]
    for n in (2, 3, 5):
        buckets = [
            [h for h in hashes if labeller.shard_of(h, n) == i] for i in range(n)
        ]
        assert sum(len(b) for b in buckets) == len(hashes)  # covers
        assert len({h for b in buckets for h in b}) == len(hashes)  # disjoint
        assert all(b for b in buckets)  # none empty
    # deterministic across calls and independent of position
    assert labeller.shard_of(hashes[7], 3) == labeller.shard_of(hashes[7], 3)


@pytest.mark.parametrize(
    "value,expected", [("0/3", (0, 3)), ("2/3", (2, 3)), (None, None)]
)
def test_parse_shard(value, expected) -> None:
    assert labeller.parse_shard(value) == expected


@pytest.mark.parametrize("bad", ["3/3", "-1/2", "5/2"])
def test_parse_shard_rejects_an_index_outside_the_range(bad) -> None:
    with pytest.raises(ValueError):
        labeller.parse_shard(bad)


# ── per-model call parameters ───────────────────────────────────────────────


def test_completion_kwargs_per_model_family() -> None:
    reasoning = labeller.completion_kwargs(cfg.label_model_spec("gpt-5.4-mini"))
    assert reasoning == {"max_completion_tokens": 1200, "reasoning_effort": "none"}
    assert "temperature" not in reasoning and "max_tokens" not in reasoning
    classic = labeller.completion_kwargs(cfg.label_model_spec("gpt-4o-mini"))
    assert classic == {"max_tokens": 800, "temperature": 0}
    # an unlisted reasoning model still gets the right shape
    assert "max_completion_tokens" in labeller.completion_kwargs(
        cfg.label_model_spec("gpt-5.9-whatever")
    )


def test_drop_unsupported_swaps_the_token_parameter_rather_than_losing_the_cap() -> (
    None
):
    kwargs = {"max_tokens": 800, "temperature": 0}
    assert (
        labeller.drop_unsupported(
            kwargs, "Unsupported value: 'temperature' does not support 0"
        )
        == "temperature"
    )
    assert kwargs == {"max_tokens": 800}
    assert (
        labeller.drop_unsupported(
            kwargs, "Unsupported parameter: 'max_tokens' is not supported"
        )
        == "max_tokens"
    )
    assert kwargs == {"max_completion_tokens": 800}  # cap kept, renamed
    assert labeller.drop_unsupported(kwargs, "something unrelated") is None


# ── cache-only merge ────────────────────────────────────────────────────────


def _corpus(tmp_path: Path, n: int = 9) -> Path:
    root = tmp_path / "corpus"
    (root / "canonical").mkdir(parents=True)
    manifest, chunks = [], []
    for i in range(1, n + 1):
        sha = _sha(i)
        (root / "canonical" / f"{i}.txt").write_text(
            "Body text. " * 40, encoding="utf-8"
        )
        manifest.append(
            {
                "id": i,
                "regulator": "HMT",
                "document_type": "policy_papers",
                "alert_class": "A3",
                "stratum": "consultation",
                "tier": "GREEN",
                "licence": "uk_ogl_v3",
                "title": f"Doc {i}",
                "url": f"https://x/{i}",
                "release_date": "2026-01-01",
                "sha256": sha,
                "chunks": 1,
                "duplicate_of": None,
            }
        )
        chunks.append(
            {
                "chunk_id": f"hmt:policy_papers#{i}/s",
                "doc_id": i,
                "document_id": f"hmt:policy_papers#{i}",
                "section": f"Doc {i}",
                "text": "Body text.",
                "start": 0,
                "end": 10,
                "source_url": f"https://x/{i}",
                "doc_hash": sha,
            }
        )
    write_jsonl(root / "manifest.jsonl", manifest)
    write_jsonl(root / "chunks.jsonl", chunks)
    return root


def _cache(labels_dir: Path, sha: str, model: str, alert_class: str) -> None:
    path = labels_dir / "cache" / sha[:2] / f"{sha}.{PROMPT_VERSION}.{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label": {**LABEL, "alert_class": alert_class},
                "usage": {"prompt_tokens": 3000, "completion_tokens": 200},
            }
        ),
        encoding="utf-8",
    )


def test_cache_only_merges_shards_and_prefers_the_stronger_model(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    labels = tmp_path / "labels"
    shas = [_sha(i) for i in range(1, 10)]
    for sha in shas:  # every document has the cheap label
        _cache(labels, sha, "gpt-4o-mini", "A3")
    for sha in shas[:4]:  # four were re-done by the strong one
        _cache(labels, sha, "gpt-5.4", "A2")

    argv = [
        "--corpus-dir",
        str(corpus),
        "--out",
        str(labels),
        "--version",
        "t",
        "--cache-only",
        "--prefer-models",
        "gpt-5.4,gpt-4o-mini",
    ]
    assert labeller.main(argv) == 0
    records = read_jsonl(labels / "triage_t.jsonl")
    assert len(records) == 9
    by_model = {r["model"] for r in records}
    assert by_model == {"gpt-5.4", "gpt-4o-mini"}
    strong = [r for r in records if r["model"] == "gpt-5.4"]
    assert len(strong) == 4 and all(r["label"]["alert_class"] == "A2" for r in strong)
    cost = json.loads((labels / "cost_t.json").read_text())
    assert cost["documents"] == 9 and cost["missing"] == 0
    assert cost["models"] == ["gpt-4o-mini", "gpt-5.4"]
    assert cost["prompt_tokens"] == 9 * 3000  # recomputed from the cache


def test_cache_only_reports_what_is_missing(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    labels = tmp_path / "labels"
    for sha in [_sha(i) for i in range(1, 5)]:  # 4 of 9 labelled
        _cache(labels, sha, "gpt-5.4-mini", "A3")
    argv = [
        "--corpus-dir",
        str(corpus),
        "--out",
        str(labels),
        "--version",
        "t",
        "--cache-only",
        "--model",
        "gpt-5.4-mini",
    ]
    assert labeller.main(argv) == 0
    assert len(read_jsonl(labels / "triage_t.jsonl")) == 4
    assert json.loads((labels / "cost_t.json").read_text())["missing"] == 5


def test_shards_write_their_own_files_and_partition_the_corpus(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    labels = tmp_path / "labels"
    for sha in [_sha(i) for i in range(1, 10)]:
        _cache(labels, sha, "gpt-5.4-mini", "A3")
    seen: list = []
    for i in range(3):
        argv = [
            "--corpus-dir",
            str(corpus),
            "--out",
            str(labels),
            "--version",
            "t",
            "--cache-only",
            "--model",
            "gpt-5.4-mini",
            "--shard",
            f"{i}/3",
        ]
        assert labeller.main(argv) == 0
        shard = read_jsonl(labels / f"triage_t.shard{i}.jsonl")
        seen.extend(r["doc_hash"] for r in shard)
    assert len(seen) == 9 and len(set(seen)) == 9  # disjoint and complete


# ── throttle ────────────────────────────────────────────────────────────────


def test_token_bucket_settles_reservations_to_actual_usage() -> None:
    import asyncio

    async def scenario():
        bucket = labeller.TokenBucket(10_000)
        entries = [await bucket.wait(4_000) for _ in range(2)]  # 8,000 reserved
        assert sum(e[1] for e in bucket.window) == 8_000
        for entry in entries:
            bucket.settle(entry, 1_000)  # they really cost 1,000 each
        assert sum(e[1] for e in bucket.window) == 2_000
        # budget freed, so more calls fit in the same minute
        assert await bucket.wait(4_000) is not None
        return True

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_token_bucket_without_a_limit_never_blocks() -> None:
    import asyncio

    async def scenario():
        bucket = labeller.TokenBucket(None)
        results = [await bucket.wait(10**9) for _ in range(3)]
        return all(r is None for r in results)

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_cost_is_priced_per_model_not_at_the_default_rate(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, n=4)
    labels = tmp_path / "labels"
    shas = [_sha(i) for i in range(1, 5)]
    for sha in shas[:2]:
        _cache(labels, sha, "gpt-4o-mini", "A3")  # $0.00015 / $0.0006
    for sha in shas[2:]:
        _cache(labels, sha, "gpt-5.4-mini", "A3")  # $0.00075 / $0.0045
    argv = [
        "--corpus-dir",
        str(corpus),
        "--out",
        str(labels),
        "--version",
        "t",
        "--cache-only",
        "--prefer-models",
        "gpt-5.4-mini,gpt-4o-mini",
        "--model",
        "gpt-4o-2024-08-06",
    ]  # a pricier default: ignored
    assert labeller.main(argv) == 0
    cost = json.loads((labels / "cost_t.json").read_text())
    cheap = 2 * (3000 * 0.00015 + 200 * 0.0006) / 1000
    dear = 2 * (3000 * 0.00075 + 200 * 0.0045) / 1000
    assert cost["estimated_usd"] == pytest.approx(cheap + dear, rel=1e-3)
    assert set(cost["by_model"]) == {"gpt-4o-mini", "gpt-5.4-mini"}
    assert cost["by_model"]["gpt-5.4-mini"]["documents"] == 2


# ── a hung call must not stop the run ───────────────────────────────────────


def test_one_document_cannot_monopolise_a_concurrency_slot() -> None:
    """The call happens inside the semaphore, so an unbounded retry ladder on
    one record starves the run."""
    assert labeller.DOC_DEADLINE_SECS <= 300


def test_a_timeout_is_retryable_like_any_other_transient_failure() -> None:
    import inspect

    src = inspect.getsource(labeller.label_one)
    retryable = src[src.index("retryable = name in"):src.index(")", src.index("retryable = name in"))]
    for name in ("TimeoutError", "RateLimitError", "APIConnectionError"):
        assert name in retryable, name


def test_retries_are_logged_rather_than_silent(caplog) -> None:
    """A run crawling along because every other call hung looked identical to
    a healthy one, because retries said nothing and the count of failures
    stayed at zero."""
    import inspect

    src = inspect.getsource(labeller.label_one)
    assert "log.warning" in src, "a retry must say so"
