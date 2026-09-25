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
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import random
import sys
import time
from collections import Counter
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
    price = cfg.PRICE_PER_1K.get(model)
    if not price:
        return None
    return round(
        prompt_tokens / 1000 * price[0] + completion_tokens / 1000 * price[1], 4
    )


async def label_one(client, model: str, doc: DocInput, sem: asyncio.Semaphore) -> Dict:
    messages = build_messages(doc)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                completion = await client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    response_format=TriageRecord,
                    max_tokens=800,
                    temperature=0,
                )
            choice = completion.choices[0]
            if choice.message.refusal:
                return {
                    "error": f"refusal: {choice.message.refusal}",
                    "attempts": attempt,
                }
            parsed = choice.message.parsed
            usage = completion.usage
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
            retryable = name in (
                "RateLimitError",
                "APIConnectionError",
                "APITimeoutError",
                "InternalServerError",
                "APIStatusError",
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
    if args.limit:
        rows = rows[: args.limit]
    log.info("%d documents selected (tiers %s)", len(rows), ",".join(args.tier))

    out_dir: Path = args.out
    cache_dir = out_dir / "cache"
    model = args.model or os.environ.get("OPENAI_MODEL") or cfg.TRIAGE["default_model"]

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
    for doc in docs:
        cp = cache_path(cache_dir, doc.doc_hash, model)
        if cp.exists():
            cached = json.loads(cp.read_text(encoding="utf-8"))
            records.append(
                build_record(
                    doc, by_id[doc.doc_id], cached, model, True, args.counsel_signoff
                )
            )
        else:
            todo.append(doc)
    log.info("%d cached, %d to label", len(records), len(todo))

    if args.dry_run:
        est_in = sum(len(d.as_text()) // 4 + 900 for d in todo)
        usd = estimate_usd(model, est_in, len(todo) * 400)
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

        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)
        started = time.monotonic()

        async def one(doc: DocInput):
            result = await label_one(client, model, doc, sem)
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
    n = write_jsonl(out_dir / f"triage_{args.version}.jsonl", records)
    if failures:
        write_jsonl(out_dir / f"failures_{args.version}.jsonl", failures)
    cost_path = out_dir / f"cost_{args.version}.json"
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
        "prompt_tokens": prior.get("prompt_tokens", 0) + tokens["prompt"],
        "completion_tokens": prior.get("completion_tokens", 0) + tokens["completion"],
        "tiers": sorted({r["tier"] for r in records}),
        "counsel_signoff": args.counsel_signoff,
        "updated_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    cost["estimated_usd"] = estimate_usd(
        model, cost["prompt_tokens"], cost["completion_tokens"]
    )
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
        "--dry-run",
        action="store_true",
        help="count documents and estimate tokens; no API calls",
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cfg.load_env()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
