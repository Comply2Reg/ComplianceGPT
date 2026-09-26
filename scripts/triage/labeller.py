#!/usr/bin/env python3
"""Label UK alert documents with OpenAI structured outputs.

    python -m scripts.triage.labeller --corpus-dir <corpus> --out data/uk/labels \\
        --version v1 [--tier GREEN] [--limit 50] [--concurrency 4] [--dry-run]
    python -m scripts.triage.labeller ... --tier GREEN AMBER \\
        --allow-amber --counsel-signoff "LEGAL-123 2026-10-01"

Licence gate: AMBER documents (FCA, PRA, BoE, DRCF, Ofcom — internal use only)
are refused unless BOTH --allow-amber and --counsel-signoff are given; the
sign-off is recorded in the cost log and on every AMBER label. There is no
environment override.

Cache: data/uk/labels/cache/{hash[:2]}/{doc_hash}.{PROMPT_VERSION}.{model}.json —
a re-run with nothing changed makes no API calls. Failures after retries go
to failures_<v>.jsonl; there is deliberately no regex fallback.

Throughput is capped per model by the account's tokens-per-minute limit, not by
our concurrency, so several processes on ONE model just collide. Split the work
instead, then merge from the cache:

    python -m scripts.triage.labeller ... --probe-limits
    for i in 0 1 2; do
        python -m scripts.triage.labeller ... --shard $i/3 --tpm <total/3> &
    done
    python -m scripts.triage.labeller ... --cache-only \
        --prefer-models gpt-5.4,gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import (  # noqa: E402
    Corpus,
    filter_rows,
    write_jsonl,
)
from scripts.triage.doc_input import DocInput, build_doc_input  # noqa: E402
from scripts.triage.prompts import PROMPT_VERSION, build_messages  # noqa: E402
from scripts.triage.schema import TriageRecord  # noqa: E402
from scripts.triage.taxonomy import CLASS_TO_FUNCTION  # noqa: E402

log = logging.getLogger("triage.labeller")
MAX_RETRIES = 6
RETRY_BASE_SECS = 3.0
# Measured prompt size (p50 3.2k over 676 calls); the completion side comes
# from the model's own cap, so the throttle paces either model family right.
EST_PROMPT_TOKENS = 3200
_DROPPED_PARAMS: set = set()


def completion_kwargs(spec: Dict) -> Dict:
    """Per-model call arguments: the gpt-5 family rejects what gpt-4o requires."""
    kwargs: Dict = {spec["token_param"]: spec["max_output"]}
    if spec.get("temperature"):
        kwargs["temperature"] = 0
    if spec.get("reasoning_effort"):
        kwargs["reasoning_effort"] = spec["reasoning_effort"]
    return kwargs


def drop_unsupported(kwargs: Dict, message: str) -> Optional[str]:
    """Remove a parameter the API just rejected, so an unlisted model still runs.

    `max_tokens` is swapped for its reasoning-model spelling rather than
    dropped: without a cap a runaway generation would be billed in full.
    """
    for key in list(kwargs):
        if key in message:
            value = kwargs.pop(key)
            if key == "max_tokens":
                kwargs["max_completion_tokens"] = value
            if key not in _DROPPED_PARAMS:
                _DROPPED_PARAMS.add(key)
                log.warning("model rejected %r; continuing without it", key)
            return key
    return None


def shard_of(doc_hash: str, shards: int) -> int:
    """Which shard a document belongs to — deterministic and order-independent.

    The whole digest is used, not a prefix: ids that share a prefix would
    otherwise all land in one shard.
    """
    try:
        return int(doc_hash, 16) % shards
    except ValueError:  # not a hex digest; still needs a stable answer
        return int(hashlib.sha256(doc_hash.encode()).hexdigest(), 16) % shards


def parse_shard(value: Optional[str]) -> Optional[tuple]:
    if not value:
        return None
    index, _, total = value.partition("/")
    i, n = int(index), int(total)
    if not 0 <= i < n:
        raise ValueError(f"--shard {value}: index must be in 0..{n - 1}")
    return i, n


# Ceiling for one labelling call, above the client's own 60s timeout. Exists
# because the client timeout has been seen not to fire, and the call holds a
# semaphore slot while it waits.
# 150s cancelled calls that were about to succeed on the longer UK documents:
# 97 responses came back 200 while only 75 were counted, because wait_for had
# already given up on them and the retry started over. Generous enough not to
# race a slow-but-healthy call, still bounded so a genuinely hung one releases
# its semaphore slot instead of deadlocking the run.
HARD_CALL_TIMEOUT = 420.0


class TokenBucket:
    """Sliding-window tokens-per-minute throttle.

    OpenAI's tier limits (gpt-4o: 30k TPM on this account; gpt-4o-mini: 200k)
    turn an unthrottled run into a wall of 429s and hung retries; pacing calls
    to the limit is faster than retrying into it.
    """

    def __init__(self, tpm: Optional[int]):
        self.tpm = tpm
        self.window: List[list] = []  # [timestamp, tokens], mutable: see settle()
        self.lock = asyncio.Lock()

    async def wait(self, tokens: int) -> Optional[list]:
        """Reserve `tokens` of this minute's budget; returns the reservation."""
        if not self.tpm:
            return None
        while True:
            async with self.lock:
                now = time.monotonic()
                self.window = [e for e in self.window if now - e[0] < 60]
                used = sum(e[1] for e in self.window)
                if used + tokens <= self.tpm:
                    entry = [now, tokens]
                    self.window.append(entry)
                    return entry
                oldest = self.window[0][0]
            await asyncio.sleep(max(0.5, 60 - (now - oldest) + 0.1))

    @staticmethod
    def settle(entry: Optional[list], actual_tokens: Optional[int]) -> None:
        """Replace a reservation with what the call really cost.

        The reservation has to assume the model spends its whole output cap;
        it rarely does (1,200 reserved, ~220 used), and without this the
        throttle would leave a quarter of the budget unspent.
        """
        if entry is not None and actual_tokens:
            entry[1] = actual_tokens


def validate_label(label: Dict, prior_alert_class: Optional[str]) -> List[str]:
    """Flags for a parsed label; nothing is rejected, everything is reported."""
    flags: List[str] = []
    cls = label.get("alert_class")
    allowed = set(CLASS_TO_FUNCTION.get(cls, {}))
    off = [
        f
        for f in label.get("primary_functions", [])
        + label.get("secondary_functions", [])
        if f not in allowed
    ]
    if off:
        flags.append("function_off_matrix")
    if not label.get("primary_functions"):
        flags.append("no_primary_function")
    if prior_alert_class and prior_alert_class != "MIXED" and cls != prior_alert_class:
        flags.append("class_disagrees_with_registry")
    if label.get("alert_class_confidence") == "low":
        flags.append("low_confidence")
    if len((label.get("summary") or "").split()) > 100:
        flags.append("summary_too_long")
    return flags


def cache_path(cache_dir: Path, doc_hash: str, model: str) -> Path:
    return cache_dir / doc_hash[:2] / f"{doc_hash}.{PROMPT_VERSION}.{model}.json"


def estimate_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> Optional[float]:
    price = cfg.label_model_spec(model).get("price_per_1k") or cfg.PRICE_PER_1K.get(
        model
    )
    if not price:
        return None
    # 6 dp, not 4: a single document costs ~$0.003, and these subtotals get
    # summed per model.
    return round(
        prompt_tokens / 1000 * price[0] + completion_tokens / 1000 * price[1], 6
    )


async def label_one(
    client,
    model: str,
    doc: DocInput,
    sem: asyncio.Semaphore,
    bucket: Optional["TokenBucket"] = None,
    call_kwargs: Optional[Dict] = None,
    est_tokens: int = EST_PROMPT_TOKENS + 800,
) -> Dict:
    messages = build_messages(doc)
    call_kwargs = call_kwargs if call_kwargs is not None else {"max_tokens": 800}
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            reservation = None
            async with sem:
                if bucket is not None:
                    reservation = await bucket.wait(est_tokens)
                # A hard ceiling on top of the client's own timeout. The
                # call is inside the semaphore, so a request that hangs
                # holds its slot forever and enough of them deadlock the
                # whole run — observed twice on the UK corpus, stalling at
                # 0% CPU with no further output. The client is configured
                # timeout=60 and that did not always fire.
                completion = await asyncio.wait_for(
                    client.beta.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=TriageRecord,
                        **call_kwargs,
                    ),
                    timeout=HARD_CALL_TIMEOUT,
                )
            choice = completion.choices[0]
            if choice.message.refusal:
                return {
                    "error": f"refusal: {choice.message.refusal}",
                    "attempts": attempt,
                }
            if choice.finish_reason == "length" or choice.message.parsed is None:
                # A reasoning model can spend the whole cap thinking; cache
                # nothing and report it rather than storing a half record.
                return {
                    "error": f"truncated (finish_reason={choice.finish_reason}); "
                    "raise max_output or lower reasoning_effort",
                    "attempts": attempt,
                }
            parsed = choice.message.parsed
            usage = completion.usage
            if bucket is not None and usage:
                bucket.settle(reservation, usage.total_tokens)
            return {
                "label": parsed.model_dump(),
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                },
                "attempts": attempt,
            }
        except Exception as exc:  # noqa: BLE001 - classified below
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            name = type(exc).__name__
            if name == "BadRequestError" and drop_unsupported(call_kwargs, str(exc)):
                continue  # same document, corrected parameter set
            retryable = name in (
                "RateLimitError",
                "APIConnectionError",
                "APITimeoutError",
                "InternalServerError",
                "APIStatusError",
                # asyncio.wait_for firing: the hard ceiling above. Transient
                # like any other timeout, so it gets the same backoff.
                "TimeoutError",
                "CancelledError",
            )
            if not retryable or attempt == MAX_RETRIES:
                break
            await asyncio.sleep(min(90, RETRY_BASE_SECS * 2**attempt + random.random()))
    return {"error": last_error or "unknown", "attempts": MAX_RETRIES}


def build_record(
    doc: DocInput,
    row: Dict,
    result: Dict,
    model: str,
    cached: bool,
    signoff: Optional[str],
) -> Dict:
    label = result["label"]
    rec = {
        "doc_id": doc.doc_id,
        "document_id": doc.document_id,
        "doc_hash": doc.doc_hash,
        "regulator": doc.regulator,
        "document_type": doc.document_type,
        "title": doc.title,
        "url": row.get("url"),
        "release_date": doc.release_date,
        "tier": doc.tier,
        "licence": row.get("licence"),
        "stratum": row.get("stratum"),
        "prior_alert_class": doc.prior_alert_class,
        "label": label,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "usage": result.get("usage"),
        "cached": cached,
        "flags": validate_label(label, doc.prior_alert_class),
        "labelled_at": result.get("labelled_at") or dt.datetime.now(dt.UTC).isoformat(),
    }
    if doc.tier == "AMBER":
        rec["licence_signoff"] = signoff
    return rec


async def run(args) -> int:
    corpus = Corpus(args.corpus_dir)
    corpus.check()
    rows = filter_rows(corpus.manifest(), args.tier, args.regulator, args.stratum)
    amber = [r for r in rows if str(r.get("tier")).upper() == "AMBER"]
    if amber and not (args.allow_amber and args.counsel_signoff):
        log.error(
            "%d AMBER documents selected; refusing without --allow-amber AND "
            "--counsel-signoff (FCA/PRA/BoE text is internal-use only)",
            len(amber),
        )
        return 2
    if args.ids_file:
        wanted = {
            int(line.split(",")[0])
            for line in Path(args.ids_file).read_text().split("\n")
            if line.strip() and line.split(",")[0].strip().isdigit()
        }
        rows = [r for r in rows if int(r["id"]) in wanted]
        log.info(
            "--ids-file %s: %d of %d documents", args.ids_file, len(rows), len(wanted)
        )
    if args.sample:
        # Stratified random sample by document type, so a smoke run sees the
        # whole corpus rather than the first N ids (all HMT policy papers).
        rng = random.Random(args.seed)
        by_type: Dict[str, List[Dict]] = {}
        for r in rows:
            by_type.setdefault(f"{r['regulator']}:{r['document_type']}", []).append(r)
        picked: List[Dict] = []
        per = max(1, args.sample // len(by_type))
        for key in sorted(by_type):
            group = by_type[key]
            picked.extend(rng.sample(group, min(per, len(group))))
        rest = [r for r in rows if r not in picked]
        rng.shuffle(rest)
        rows = (picked + rest)[: args.sample]
    shard = parse_shard(args.shard)
    if shard:
        index, total = shard
        rows = [r for r in rows if shard_of(r["sha256"], total) == index]
        log.info("--shard %s: %d documents in this shard", args.shard, len(rows))
    if args.limit:
        rows = rows[: args.limit]
    log.info("%d documents selected (tiers %s)", len(rows), ",".join(args.tier))

    out_dir: Path = args.out
    cache_dir = out_dir / "cache"
    model = args.model or os.environ.get("OPENAI_MODEL") or cfg.DEFAULT_LABEL_MODEL
    spec = cfg.label_model_spec(model)
    call_kwargs = completion_kwargs(spec)
    est_tokens = EST_PROMPT_TOKENS + int(spec["max_output"])
    # Shards write their own files; --cache-only merges them into the real ones.
    suffix = f".shard{shard[0]}" if shard else ""

    chunks = corpus.chunks_by_doc([int(r["id"]) for r in rows])
    docs: List[DocInput] = []
    for r in rows:
        try:
            canonical = corpus.canonical(int(r["id"]))
        except OSError as exc:
            log.warning("no canonical for %s: %s", r["id"], exc)
            continue
        docs.append(build_doc_input(r, canonical, chunks.get(int(r["id"]), [])))
    by_id = {int(r["id"]): r for r in rows}

    records: List[Dict] = []
    failures: List[Dict] = []
    todo: List[DocInput] = []
    prefer = args.prefer_models or [model]
    for doc in docs:
        for candidate in prefer:
            cp = cache_path(cache_dir, doc.doc_hash, candidate)
            if not cp.exists():
                continue
            cached = json.loads(cp.read_text(encoding="utf-8"))
            records.append(
                build_record(
                    doc,
                    by_id[doc.doc_id],
                    cached,
                    candidate,
                    True,
                    args.counsel_signoff,
                )
            )
            break
        else:
            todo.append(doc)
    log.info("%d cached, %d to label", len(records), len(todo))
    missing = 0
    if args.cache_only:
        missing, todo = len(todo), []
        log.info(
            "--cache-only: assembling from %s; %d documents still unlabelled",
            ",".join(prefer),
            missing,
        )

    if args.dry_run:
        est_in = sum(len(d.as_text()) // 4 + 900 for d in todo)
        usd = estimate_usd(model, est_in, len(todo) * int(spec["max_output"]) // 2)
        log.info(
            "dry run: %d calls, ~%d prompt tokens, ~%d completion tokens, ~$%s (%s)",
            len(todo),
            est_in,
            len(todo) * 400,
            usd,
            model,
        )
        return 0

    calls = 0
    tokens = Counter()
    if todo:
        if not os.environ.get("OPENAI_API_KEY"):
            log.error("OPENAI_API_KEY is not set (put it in .env; never in chat)")
            return 2
        from openai import AsyncOpenAI

        # Our own ladder handles retries; a hung socket must not block for the
        # SDK's default 10 minutes.
        client = AsyncOpenAI(max_retries=0, timeout=60.0)
        bucket = TokenBucket(args.tpm)
        sem = asyncio.Semaphore(args.concurrency)
        started = time.monotonic()

        async def one(doc: DocInput):
            result = await label_one(
                client, model, doc, sem, bucket, call_kwargs, est_tokens
            )
            result["labelled_at"] = dt.datetime.now(dt.UTC).isoformat()
            return doc, result

        for i, coro in enumerate(asyncio.as_completed([one(d) for d in todo]), 1):
            doc, result = await coro
            calls += 1
            if "label" in result:
                cp = cache_path(cache_dir, doc.doc_hash, model)
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                records.append(
                    build_record(
                        doc,
                        by_id[doc.doc_id],
                        result,
                        model,
                        False,
                        args.counsel_signoff,
                    )
                )
                u = result.get("usage") or {}
                tokens["prompt"] += u.get("prompt_tokens") or 0
                tokens["completion"] += u.get("completion_tokens") or 0
            else:
                failures.append(
                    {
                        "doc_id": doc.doc_id,
                        "document_id": doc.document_id,
                        "doc_hash": doc.doc_hash,
                        "error": result["error"],
                        "attempts": result["attempts"],
                    }
                )
            if i % 25 == 0 or i == len(todo):
                log.info(
                    "%d/%d labelled, %d failed, %.0fs",
                    i,
                    len(todo),
                    len(failures),
                    time.monotonic() - started,
                )

    records.sort(key=lambda r: r["doc_id"])
    n = write_jsonl(out_dir / f"triage_{args.version}{suffix}.jsonl", records)
    if failures:
        write_jsonl(out_dir / f"failures_{args.version}{suffix}.jsonl", failures)
    cost_path = out_dir / f"cost_{args.version}{suffix}.json"
    prior = json.loads(cost_path.read_text()) if cost_path.exists() else {}
    cost = {
        "version": args.version,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "runs": prior.get("runs", 0) + 1,
        "documents": n,
        "calls_this_run": calls,
        "cached_this_run": len(records) - calls + len(failures)
        if calls
        else len(records),
        "failures": len(failures),
        "prompt_tokens": sum(
            (r.get("usage") or {}).get("prompt_tokens") or 0 for r in records
        )
        if args.cache_only
        else prior.get("prompt_tokens", 0) + tokens["prompt"],
        "completion_tokens": sum(
            (r.get("usage") or {}).get("completion_tokens") or 0 for r in records
        )
        if args.cache_only
        else prior.get("completion_tokens", 0) + tokens["completion"],
        "models": sorted({r["model"] for r in records}),
        "missing": missing,
        "shard": args.shard,
        "tiers": sorted({r["tier"] for r in records}),
        "counsel_signoff": args.counsel_signoff,
        "updated_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    # Price each record at the rate of the model that produced it: a merged
    # file mixes models, and charging all of them the default model's rate was
    # off by 3x.
    per_model: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        usage = record.get("usage") or {}
        per_model[record["model"]]["prompt"] += usage.get("prompt_tokens") or 0
        per_model[record["model"]]["completion"] += usage.get("completion_tokens") or 0
    priced = [
        estimate_usd(name, counts["prompt"], counts["completion"])
        for name, counts in per_model.items()
    ]
    cost["estimated_usd"] = (
        round(sum(v for v in priced if v is not None), 6) if any(priced) else None
    )
    cost["by_model"] = {
        name: {
            "documents": sum(1 for r in records if r["model"] == name),
            "prompt_tokens": counts["prompt"],
            "completion_tokens": counts["completion"],
            "estimated_usd": estimate_usd(name, counts["prompt"], counts["completion"]),
        }
        for name, counts in sorted(per_model.items())
    }
    cost_path.write_text(json.dumps(cost, indent=2), encoding="utf-8")
    flags = Counter(f for r in records for f in r["flags"])
    log.info(
        "wrote %d labels (%d failures) -> %s | flags %s | cost %s",
        n,
        len(failures),
        out_dir,
        dict(flags),
        cost["estimated_usd"],
    )
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus-dir", type=Path, default=Path(cfg.TRIAGE["corpus_dir"]))
    ap.add_argument("--out", type=Path, default=Path(cfg.TRIAGE["labels_dir"]))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--tier", nargs="+", default=["GREEN"])
    ap.add_argument("--allow-amber", action="store_true")
    ap.add_argument(
        "--counsel-signoff",
        default=None,
        help="reference and date of the licence sign-off for AMBER text",
    )
    ap.add_argument("--regulator", nargs="*", default=None)
    ap.add_argument("--stratum", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        help="stratified random sample of N documents by document type",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="parallel calls; 4 tripped 429s on a fresh gpt-4o quota",
    )
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--tpm",
        type=int,
        default=None,
        help="tokens-per-minute budget to pace calls to (e.g. 28000 "
        "for a tier-1 gpt-4o account); default: no pacing",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="count documents and estimate tokens; no API calls",
    )
    ap.add_argument(
        "--shard",
        default=None,
        metavar="i/N",
        help="label only shard i of N (deterministic by document hash), so "
        "several processes can run side by side without overlapping work; "
        "each writes triage_<v>.shard<i>.jsonl",
    )
    ap.add_argument(
        "--cache-only",
        action="store_true",
        help="make no API calls: assemble the labels file from the cache "
        "(the merge step after sharded runs) and report what is missing",
    )
    ap.add_argument(
        "--prefer-models",
        type=lambda v: [m.strip() for m in v.split(",") if m.strip()],
        default=None,
        metavar="a,b,c",
        help="with --cache-only: per document take the first of these models "
        "that has a cached label (strongest first)",
    )
    ap.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="label only the doc_ids listed in this file (one per line, or the "
        "first CSV column) — for a strong-model pass over disputed rows",
    )
    ap.add_argument(
        "--probe-limits",
        action="store_true",
        help="one tiny call: print this key's rate limits for the model and a "
        "suggested --tpm, then exit",
    )
    return ap


def probe_limits(model: str) -> int:
    """Report the account's rate limits for a model from the response headers."""
    from openai import OpenAI

    spec = cfg.label_model_spec(model)
    kwargs = completion_kwargs({**spec, "max_output": 16})
    try:
        response = OpenAI(
            max_retries=0, timeout=30.0
        ).chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": "ok"}], **kwargs
        )
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        log.error("probe failed for %s: %s", model, str(exc)[:300])
        return 2
    headers = response.headers
    tpm = headers.get("x-ratelimit-limit-tokens")
    rpm = headers.get("x-ratelimit-limit-requests")
    log.info(
        "%s | requests %s/min (%s left) | tokens %s/min (%s left, reset %s)",
        model,
        rpm,
        headers.get("x-ratelimit-remaining-requests"),
        tpm,
        headers.get("x-ratelimit-remaining-tokens"),
        headers.get("x-ratelimit-reset-tokens"),
    )
    if tpm and tpm.isdigit():
        budget = int(int(tpm) * 0.9)
        per_call = EST_PROMPT_TOKENS + int(spec["max_output"])
        log.info(
            "suggested: --tpm %d (90%% of the ceiling) ≈ %d docs/min; "
            "across 3 shards use --tpm %d each",
            budget,
            budget // per_call,
            budget // 3,
        )
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg.load_env()
    if args.probe_limits:
        return probe_limits(
            args.model or os.environ.get("OPENAI_MODEL") or cfg.DEFAULT_LABEL_MODEL
        )
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
