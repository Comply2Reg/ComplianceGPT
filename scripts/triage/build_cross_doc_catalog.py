#!/usr/bin/env python3
"""Seed the cross-jurisdiction catalogue from the labelled corpus.

`c2r_cga_api` already has the machinery for cross-document relationships —
`ai/knowledge/graph/cross_document_rules.py` defines SIMILAR_TO, EQUIVALENT_TO,
DERIVED_FROM, INSPIRED_BY, COVERS_TOPIC and HAS_EQUIVALENT_OBLIGATION, and
`neo4j_graph_enricher` merges them into the graph. What it has never had is the
catalogue file those rules are read from; the loader logs "Catalog not found"
and returns an empty list.

This fills it from evidence rather than by hand. The labeller tags each
document with the international frameworks it implements or responds to, so a
framework naming documents in two jurisdictions is exactly a cross-border link:
a US rule and a UK one that both answer to Basel III are about the same thing.

    python -m scripts.triage.build_cross_doc_catalog \\
        --labels data/uk/labels/triage_uk_v2.jsonl data/uk/labels/triage_us_v1.jsonl \\
        --out ../../00-apps/c2r_cga_api/compliance_gap_analysis/cross_document_catalog.json

**Result: the title-term approach does not work, and this catalogue is not
installed.** Tried at several thresholds, the output is mostly noise — "Basel
III: GB 'financial services' -> US 'renewal'", "BSA/AML: GB 'laundering' ->
US 'renewal without'", where "renewal without" is Treasury title boilerplate.
A few links are right (FATF: GB 'laundering' -> US 'laundering') but the
signal-to-noise is far too low to merge into the graph, so nothing is written
to c2r_cga_api. The script is kept because it produced that evidence and
because it becomes useful the moment the blocker below is removed.

**The blocker.** The matcher can only
test properties a Regulation node actually carries: title, regulator,
jurisdiction_code, document_type. It cannot say "documents tagged Basel III",
because `frameworks` is not on the node. So the links below are expressed as
title terms that discriminate the tagged documents, which is a proxy and will
both miss and over-match. The durable fix is to write `frameworks` onto
Regulation nodes at index time and match on it directly; then this script
becomes a handful of exact rules instead of a term-frequency heuristic.

What the labels do support, independent of this heuristic, is the evidence
that cross-border links exist to be made: eleven frameworks name documents in
both jurisdictions, led by BSA/AML (GB 34 / US 1,098), FATF (38 / 449), Basel
III (35 / 187), MiFID (95 / 20) and GDPR (28 / 66). Indexing `frameworks` onto
Regulation nodes turns that into exact matches.

Everything here is COVERS_TOPIC — the safe claim that two documents concern the
same framework. EQUIVALENT_TO between a specific pair of instruments is a
judgement about legal effect and should be added by a person, not inferred.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.triage.corpus import read_jsonl  # noqa: E402

log = logging.getLogger("triage.cross_doc")

# Words that discriminate nothing: they appear across every framework.
_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "under", "amendments",
    "amendment", "final", "proposed", "rule", "rules", "notice", "notices",
    "act", "regulation", "regulations", "requirements", "requirement", "of",
    "to", "in", "on", "a", "an", "by", "or", "certain", "other", "new",
    "policy", "statement", "consultation", "paper", "report", "guidance",
    "review", "update", "updates", "changes", "change", "part", "parts",
}
_WORD = re.compile(r"[a-z][a-z'-]{2,}")


def phrases(title: str) -> Set[str]:
    """Single words and adjacent pairs, lowercased, minus the noise."""
    words = [w for w in _WORD.findall((title or "").lower()) if w not in _STOP]
    out: Set[str] = set(words)
    out.update(f"{a} {b}" for a, b in zip(words, words[1:]))
    return out


def discriminating_terms(
    tagged: List[str], untagged: List[str], *, top: int, min_docs: int,
    min_lift: float,
) -> List[str]:
    """Title terms far commoner among a framework's documents than elsewhere.

    A term is kept when it appears in at least `min_docs` tagged titles and is
    at least `min_lift` times as frequent there as in the rest of the corpus.
    """
    tag_counts: Counter = Counter()
    for t in tagged:
        tag_counts.update(phrases(t))
    un_counts: Counter = Counter()
    for t in untagged:
        un_counts.update(phrases(t))

    n_tag = max(len(tagged), 1)
    n_un = max(len(untagged), 1)
    scored: List[Tuple[float, int, str]] = []
    for term, n in tag_counts.items():
        if n < min_docs:
            continue
        rate_tag = n / n_tag
        rate_un = un_counts.get(term, 0) / n_un
        lift = rate_tag / rate_un if rate_un else float("inf")
        if lift < min_lift:
            continue
        scored.append((lift, n, term))
    # Commonest first among those that pass, so the terms chosen cover the most
    # documents rather than being the most exotic.
    scored.sort(key=lambda x: (-x[1], -x[0]))
    chosen: List[str] = []
    for _, _, term in scored:
        # Skip a term already covered by a shorter one we kept.
        if any(c in term for c in chosen):
            continue
        chosen.append(term)
        if len(chosen) >= top:
            break
    return chosen


def load_documents(paths: Iterable[Path]) -> List[Dict]:
    docs: List[Dict] = []
    for p in paths:
        for row in read_jsonl(p):
            label = row.get("label") or {}
            docs.append(
                {
                    "title": row.get("title") or "",
                    "jurisdiction": (label.get("jurisdiction") or "").upper(),
                    "frameworks": label.get("frameworks") or [],
                }
            )
    return docs


def build(docs: List[Dict], *, min_docs_per_jurisdiction: int, top_terms: int,
          min_lift: float) -> Dict:
    by_fw_jur: Dict[str, Dict[str, List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for d in docs:
        for fw in d["frameworks"]:
            if d["jurisdiction"]:
                by_fw_jur[fw][d["jurisdiction"]].append(d["title"])

    topics = []
    for fw, per_jur in sorted(by_fw_jur.items()):
        jurs = sorted(
            j for j, titles in per_jur.items()
            if len(titles) >= min_docs_per_jurisdiction and j != "ZZ"
        )
        if len(jurs) < 2:
            continue  # nothing to link across
        tagged_titles = {j: per_jur[j] for j in jurs}
        links = []
        for i, a in enumerate(jurs):
            for b in jurs[i + 1:]:
                terms_a = discriminating_terms(
                    tagged_titles[a],
                    [d["title"] for d in docs
                     if d["jurisdiction"] == a and fw not in d["frameworks"]],
                    top=top_terms, min_docs=min_docs_per_jurisdiction,
                    min_lift=min_lift,
                )
                terms_b = discriminating_terms(
                    tagged_titles[b],
                    [d["title"] for d in docs
                     if d["jurisdiction"] == b and fw not in d["frameworks"]],
                    top=top_terms, min_docs=min_docs_per_jurisdiction,
                    min_lift=min_lift,
                )
                if not terms_a or not terms_b:
                    continue
                for ta in terms_a:
                    for tb in terms_b:
                        links.append({
                            "rel": "COVERS_TOPIC",
                            "from": {"jurisdiction_code": a, "title_contains": ta},
                            "to": {"jurisdiction_code": b, "title_contains": tb},
                            "rel_topic": fw,
                        })
        if links:
            topics.append({
                "topic": fw,
                "jurisdictions": jurs,
                "documents": {j: len(tagged_titles[j]) for j in jurs},
                "links": links,
            })
    return {
        "_generated_by": "scripts/triage/build_cross_doc_catalog.py",
        "_note": (
            "COVERS_TOPIC only: two documents concern the same international "
            "framework. Title terms are a proxy for the labeller's frameworks "
            "field, which Regulation nodes do not carry; match on the field "
            "directly once it is indexed. EQUIVALENT_TO between named "
            "instruments is a legal judgement and belongs to a person."
        ),
        "topics": topics,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-docs", type=int, default=3,
                    help="documents a framework needs per jurisdiction")
    ap.add_argument("--top-terms", type=int, default=4,
                    help="title terms per jurisdiction per framework")
    ap.add_argument("--min-lift", type=float, default=3.0,
                    help="how much commoner a term must be among tagged titles")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    docs = load_documents(args.labels)
    log.info("%d labelled documents", len(docs))
    catalog = build(
        docs,
        min_docs_per_jurisdiction=args.min_docs,
        top_terms=args.top_terms,
        min_lift=args.min_lift,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    n_links = sum(len(t["links"]) for t in catalog["topics"])
    log.info("%d topics, %d links -> %s", len(catalog["topics"]), n_links, args.out)
    for t in catalog["topics"]:
        log.info("  %-16s %s  %d links", t["topic"], t["documents"], len(t["links"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
