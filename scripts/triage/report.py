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
from scripts.triage.taxonomy import ALERT_CLASSES, CLASS_TO_STRATUM, TARGET_MIX  # noqa: E402


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def table(headers: List[str], rows: List[List]) -> List[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--weak", type=Path, default=None)
    ap.add_argument("--cost", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
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
