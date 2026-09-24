#!/usr/bin/env python3
"""Feed the UK corpus chunks to the obligation pipeline on `main` — unchanged.

    python -m scripts.triage.obligation_adapter --corpus-dir <corpus> \\
        --out data/uk/obligation/chunks_uk_v1.jsonl \\
        --strata rule_final guidance --regulator-include UKLEG --tier GREEN

Emits per chunk exactly what run_extraction_stage reads (document_id,
chunk_id, text, plus title/section/regulator/jurisdiction hints). Then, from
a worktree of `main` (never merge the branches):

    git worktree add ../ComplianceGPT-main main
    cd ../ComplianceGPT-main/scripts/ObligationDataPipeline && USE_OPENAI=true \\
        python run_extract.py --version uk-v1 --chunks <abs path>/chunks_uk_v1.jsonl

Same licence gate as the labeller: AMBER chunks need --allow-amber and
--counsel-signoff.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.corpus import Corpus, filter_rows, write_jsonl  # noqa: E402

log = logging.getLogger("triage.obligation_adapter")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus-dir", type=Path, default=Path(cfg.TRIAGE["corpus_dir"]))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strata", nargs="+", default=["rule_final", "guidance"])
    ap.add_argument(
        "--regulator-include",
        nargs="*",
        default=[],
        help="regulators included regardless of stratum (UKLEG)",
    )
    ap.add_argument("--tier", nargs="+", default=["GREEN"])
    ap.add_argument("--allow-amber", action="store_true")
    ap.add_argument("--counsel-signoff", default=None)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=None, help="max documents")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    corpus = Corpus(args.corpus_dir)
    corpus.check()
    rows = corpus.manifest()
    picked = filter_rows(rows, args.tier, strata=args.strata)
    extra = (
        filter_rows(rows, args.tier, regulators=args.regulator_include)
        if args.regulator_include
        else []
    )
    seen = {r["id"] for r in picked}
    picked += [r for r in extra if r["id"] not in seen]
    if any(str(r.get("tier")).upper() == "AMBER" for r in picked) and not (
        args.allow_amber and args.counsel_signoff
    ):
        log.error(
            "AMBER documents selected; refusing without --allow-amber AND "
            "--counsel-signoff"
        )
        return 2
    if args.limit:
        picked = picked[: args.limit]
    by_id = {int(r["id"]): r for r in picked}
    log.info(
        "%d documents in strata %s (+%s)",
        len(picked),
        args.strata,
        args.regulator_include,
    )

    chunks = corpus.chunks_by_doc(by_id.keys())
    out = []
    skipped = Counter()
    for doc_id, cs in chunks.items():
        row = by_id[doc_id]
        for c in cs:
            if len(c["text"]) > args.max_chars:
                skipped["too_long"] += 1
                continue
            out.append(
                {
                    "document_id": c["document_id"],
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "title": row.get("title") or c.get("title") or "",
                    "act_name": row.get("title") or "",
                    "section_number": c.get("section") or "",
                    "regulator": c.get("regulator") or row["regulator"],
                    "jurisdiction": "GB",
                    "doc_type": row.get("document_type"),
                    "alert_class": row.get("alert_class"),
                    "source_tier": c.get("source_tier"),
                    "licence": c.get("licence"),
                    "local_path": row.get("object_key") or "",
                    "source_url": c.get("source_url"),
                }
            )
    n = write_jsonl(args.out, out)
    log.info(
        "wrote %d chunks from %d documents -> %s | skipped %s",
        n,
        len(chunks),
        args.out,
        dict(skipped),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
