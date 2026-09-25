#!/usr/bin/env python3
"""Label distribution, registry agreement and cost — one markdown report.

    python -m scripts.triage.report --labels data/uk/labels/triage_v1.jsonl \\
        [--weak data/uk/labels/weak_v1.jsonl] [--cost data/uk/labels/cost_v1.json] \\
        [--out data/uk/labels/report_v1.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.triage.corpus import read_jsonl  # noqa: E402
from scripts.triage.prompts import PROMPT_VERSION  # noqa: E402
from scripts.triage.taxonomy import ALERT_CLASSES, CLASS_TO_STRATUM, TARGET_MIX  # noqa: E402


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def table(headers: List[str], rows: List[List]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def cross_model(cache_dir: Path) -> Dict:
    """Every document two models have both labelled, straight from the cache.

    Two models disagreeing is a better gold-candidate signal than either one
    disagreeing with the registry: it needs no ground truth and it points at
    the documents that are genuinely ambiguous.
    """
    by_hash: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for path in cache_dir.rglob(f"*.{PROMPT_VERSION}.*.json"):
        name = path.name[: -len(".json")]
        doc_hash, _, rest = name.partition(f".{PROMPT_VERSION}.")
        try:
            by_hash[doc_hash][rest] = json.loads(path.read_text())["label"]
        except (OSError, ValueError, KeyError):
            continue
    return {h: m for h, m in by_hash.items() if len(m) > 1}


def compare_pair(overlap: Dict, a: str, b: str) -> Dict:
    both = {h: m for h, m in overlap.items() if a in m and b in m}
    if not both:
        return {}
    agree = Counter()
    confusion = Counter()
    jaccards = []
    for m in both.values():
        x, y = m[a], m[b]
        agree["alert_class"] += x["alert_class"] == y["alert_class"]
        agree["priority"] += x["priority"] == y["priority"]
        agree["obligations_present"] += (
            x["obligations_present"] == y["obligations_present"]
        )
        if x["alert_class"] != y["alert_class"]:
            confusion[(x["alert_class"], y["alert_class"])] += 1
        sx, sy = set(x["primary_functions"]), set(y["primary_functions"])
        jaccards.append(len(sx & sy) / len(sx | sy) if (sx | sy) else 1.0)
    n = len(both)
    return {
        "n": n,
        "agreement": {k: round(v / n, 4) for k, v in agree.items()},
        "function_jaccard": round(sum(jaccards) / len(jaccards), 3),
        "confusion": confusion.most_common(8),
        "disputed_hashes": [
            h for h, m in both.items() if m[a]["alert_class"] != m[b]["alert_class"]
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--weak", type=Path, default=None)
    ap.add_argument("--cost", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="label cache (default: alongside --labels) for the model-vs-model "
        "comparison and the disputed-document list",
    )
    ap.add_argument(
        "--disputed-out",
        type=Path,
        default=None,
        help="write the doc_ids where two models disagree, ready for "
        "labeller --ids-file (default: disputed_<version>.txt beside the labels)",
    )
    args = ap.parse_args(argv)

    labels = read_jsonl(args.labels)
    weak = {w["doc_hash"]: w for w in read_jsonl(args.weak)} if args.weak else {}
    cost = json.loads(args.cost.read_text()) if args.cost and args.cost.exists() else {}
    n = len(labels)
    lines = [
        f"# Triage labels report — `{args.labels.name}`",
        "",
        f"{n} labelled documents; model {cost.get('model', '?')}, prompt "
        f"{labels[0]['prompt_version'] if labels else '?'}.",
        "",
    ]

    by_class = Counter(lab["label"]["alert_class"] for lab in labels)
    by_stratum = Counter(
        CLASS_TO_STRATUM.get(lab["label"]["alert_class"], "?") for lab in labels
    )
    lines += ["## Alert class distribution", ""]
    lines += table(
        ["class", "label", "documents", "share"],
        [
            [c, ALERT_CLASSES[c], by_class.get(c, 0), pct(by_class.get(c, 0), n)]
            for c in ALERT_CLASSES
        ],
    )
    lines += ["", "## Stratum (from model class) vs target", ""]
    lines += table(
        ["stratum", "documents", "share", "target"],
        [
            [s, by_stratum.get(s, 0), pct(by_stratum.get(s, 0), n), f"{int(t * 100)}%"]
            for s, t in TARGET_MIX.items()
        ],
    )

    lines += ["", "## Priority and confidence", ""]
    lines += table(
        ["priority", "documents"],
        [
            [p, c]
            for p, c in sorted(
                Counter(lab["label"]["priority"] for lab in labels).items()
            )
        ],
    )
    lines += [""]
    lines += table(
        ["confidence", "documents"],
        [
            [p, c]
            for p, c in sorted(
                Counter(
                    lab["label"]["alert_class_confidence"] for lab in labels
                ).items()
            )
        ],
    )

    # Agreement with the registry prior (non-MIXED sources only)
    prior_rows = [
        lab
        for lab in labels
        if lab.get("prior_alert_class") and lab["prior_alert_class"] != "MIXED"
    ]
    agree = sum(
        1
        for lab in prior_rows
        if lab["label"]["alert_class"] == lab["prior_alert_class"]
    )
    lines += [
        "",
        "## Agreement with the registry class",
        "",
        f"{agree}/{len(prior_rows)} = {pct(agree, len(prior_rows))} on sources with a "
        "fixed class (MIXED sources excluded).",
        "",
    ]
    per_type: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    confusion = Counter()
    for lab in prior_rows:
        key = f"{lab['regulator']}:{lab['document_type']}"
        per_type[key][1] += 1
        if lab["label"]["alert_class"] == lab["prior_alert_class"]:
            per_type[key][0] += 1
        else:
            confusion[(lab["prior_alert_class"], lab["label"]["alert_class"])] += 1
    lines += table(
        ["source", "agree", "of", "share"],
        [
            [k, a, t, pct(a, t)]
            for k, (a, t) in sorted(
                per_type.items(), key=lambda kv: kv[1][0] / kv[1][1] if kv[1][1] else 0
            )
        ],
    )
    lines += ["", "Top disagreements (registry → model):", ""]
    lines += table(
        ["registry", "model", "documents"],
        [[a, b, c] for (a, b), c in confusion.most_common(8)],
    )

    mixed = [lab for lab in labels if lab.get("prior_alert_class") in (None, "MIXED")]
    if mixed:
        lines += [
            "",
            f"## How the {len(mixed)} unclassified (MIXED) documents resolved",
            "",
        ]
        res = Counter((lab["regulator"], lab["label"]["alert_class"]) for lab in mixed)
        lines += table(
            ["regulator", "class", "documents"],
            [[r, c, k] for (r, c), k in res.most_common(20)],
        )

    if weak:
        both = [
            lab
            for lab in labels
            if lab["doc_hash"] in weak and weak[lab["doc_hash"]].get("alert_class")
        ]
        agree_w = sum(
            1
            for lab in both
            if lab["label"]["alert_class"] == weak[lab["doc_hash"]]["alert_class"]
        )
        lines += [
            "",
            "## Agreement with weak (registry + title heuristic) labels",
            "",
            f"{agree_w}/{len(both)} = {pct(agree_w, len(both))}.",
        ]

    # Model vs model, read from the cache — no API calls, no ground truth needed.
    cache_dir = args.cache_dir or args.labels.parent / "cache"
    overlap = cross_model(cache_dir) if cache_dir.is_dir() else {}
    disputed_ids: List[int] = []
    if overlap:
        models = sorted({m for models in overlap.values() for m in models})
        lines += ["", "## Model vs model (documents labelled by both)", ""]
        id_of = {lab["doc_hash"]: lab["doc_id"] for lab in labels}
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                cmp = compare_pair(overlap, a, b)
                if not cmp:
                    continue
                lines += [f"**{a}** vs **{b}** — {cmp['n']} documents", ""]
                lines += table(
                    ["field", "agreement"],
                    [
                        [k, pct(round(v * cmp["n"]), cmp["n"])]
                        for k, v in cmp["agreement"].items()
                    ]
                    + [["primary_functions (Jaccard)", cmp["function_jaccard"]]],
                )
                if cmp["confusion"]:
                    lines += ["", f"Top class disagreements ({a} → {b}):", ""]
                    lines += table(
                        [a, b, "documents"],
                        [[x, y, c] for (x, y), c in cmp["confusion"]],
                    )
                lines += [""]
                disputed_ids.extend(
                    id_of[h] for h in cmp["disputed_hashes"] if h in id_of
                )
        disputed_ids = sorted(set(disputed_ids))
        out_ids = (
            args.disputed_out
            or args.labels.parent / f"disputed_{args.labels.stem.split('_')[-1]}.txt"
        )
        out_ids.write_text(
            "\n".join(str(i) for i in disputed_ids) + "\n", encoding="utf-8"
        )
        lines += [
            f"{len(disputed_ids)} documents where two models chose different "
            f"alert classes -> `{out_ids}` (feed to "
            "`labeller --model <strong> --ids-file`).",
            "",
        ]

    flags = Counter(f for lab in labels for f in lab.get("flags", []))
    lines += ["", "## Flags", ""]
    lines += table(["flag", "documents"], [[f, c] for f, c in flags.most_common()])
    lines += ["", "## By regulator and tier", ""]
    lines += table(
        ["regulator", "tier", "documents"],
        [
            [r, t, c]
            for (r, t), c in sorted(
                Counter((lab["regulator"], lab["tier"]) for lab in labels).items()
            )
        ],
    )
    if cost:
        lines += ["", "## Cost", "", "```", json.dumps(cost, indent=2), "```"]

    text = "\n".join(lines) + "\n"
    out = args.out or args.labels.with_name(
        args.labels.stem.replace("triage_", "report_") + ".md"
    )
    out.write_text(text, encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
