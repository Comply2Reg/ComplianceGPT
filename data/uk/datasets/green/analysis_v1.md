# Dataset analysis — green v1

Target model `qwen3-4b-instruct` (Qwen/Qwen3-4B-Instruct-2507), budget 2048 tokens; token counter: records' n_tokens.

## Records per split

| split | records | strata | classes | regulators |
|---|---|---|---|---|
| train | 1683 | {'rule_final': 940, 'consultation': 191, 'intelligence': 403, 'enforcement': 81, 'guidance': 68} | {'A11': 678, 'A3': 198, 'A2': 13, 'A6': 63, 'A1': 574, 'A14': 19, 'A9': 19, 'A10': 41, 'A4': 28, 'A13': 19, 'A8': 3, 'A12': 26, 'A7': 2} | {'HMT': 685, 'CMA': 469, 'UKLEG': 517, 'ICO': 12} |
| val | 218 | {'rule_final': 104, 'consultation': 21, 'intelligence': 76, 'enforcement': 9, 'guidance': 8} | {'A11': 131, 'A1': 25, 'A2': 8, 'A12': 4, 'A3': 23, 'A14': 7, 'A6': 8, 'A9': 2, 'A13': 2, 'A10': 8} | {'HMT': 139, 'UKLEG': 11, 'CMA': 67, 'ICO': 1} |
| test | 104 | {'rule_final': 10, 'consultation': 7, 'intelligence': 58, 'enforcement': 9, 'guidance': 20} | {'A11': 56, 'A12': 1, 'A1': 1, 'A3': 7, 'A14': 5, 'A9': 5, 'A6': 6, 'A13': 3, 'A10': 17, 'A4': 2, 'A8': 1} | {'HMT': 27, 'CMA': 76, 'ICO': 1} |

## Token lengths

| split | p50 | p95 | max | over 2048 | context trimmed |
|---|---|---|---|---|---|
| train | 820 | 1034 | 1226 | 0 | 0 |
| val | 774 | 999 | 1107 | 0 | 0 |
| test | 756 | 879 | 948 | 0 | 0 |

Whole-document tokens (for a long-context variant): p50 1659, p95 29983, max 249566 — 1572/2005 fit in 8k, 1771/2005 in 16k.

## Class coverage (train)

| class | label | train | val | test |
|---|---|---|---|---|
| A1 | Primary legislation / statutory instrument | 574 | 25 | 1 |
| A2 | Final rule / policy statement | 13 | 8 | 0 |
| A3 | Proposed rule / consultation / discussion paper  | 198 | 23 | 7 |
| A4 | Supervisory guidance (supervisory statement, sta | 28 | 0 | 2 |
| A5 | Supervisory communication (Dear CEO / portfolio  | 0 | 0 | 0 |
| A6 | Enforcement action (final, decision or warning n | 63 | 8 | 6 |
| A7 | Sanctions / watchlist delta | 2 | 0 | 0 |
| A8 | Reporting, returns or taxonomy change | 3 | 0 | 1 |
| A9 | Market or operational notice | 19 | 2 | 5 |
| A10 | Thematic review, multi-firm review or market stu | 41 | 8 | 17 |
| A11 | Intelligence: speech, blog, press release, resea | 678 | 131 | 56 |
| A12 | Codified rulebook or handbook, point-in-time | 26 | 4 | 1 |
| A13 | Court or tribunal determination | 19 | 2 | 3 |
| A14 | Perimeter, register or authorisation change (war | 19 | 7 | 5 |

Classes with fewer than 30 training examples: A2, A4, A5, A7, A8, A9, A12, A13, A14. A model cannot learn a class it has not seen; these need more sources (see docs/uk-datasets.md) or must be excluded from the evaluation claims.

## Train mix vs target

| stratum | records | share | target |
|---|---|---|---|
| rule_final | 940 | 55.9% | 35% |
| consultation | 191 | 11.3% | 20% |
| guidance | 68 | 4.0% | 15% |
| enforcement | 81 | 4.8% | 15% |
| intelligence | 403 | 23.9% | 15% |

## Diversity and repetition (train inputs)

| class | records | unique 8-gram ratio | records with an internal 10-word repeat |
|---|---|---|---|
| A1 | 574 | 0.669 | 502 |
| A2 | 13 | 0.846 | 8 |
| A3 | 198 | 0.729 | 95 |
| A4 | 28 | 0.794 | 9 |
| A6 | 63 | 0.792 | 23 |
| A7 | 2 | 0.968 | 1 |
| A8 | 3 | 0.901 | 1 |
| A9 | 19 | 0.669 | 5 |
| A10 | 41 | 0.862 | 14 |
| A11 | 678 | 0.777 | 377 |
| A12 | 26 | 0.824 | 10 |
| A13 | 19 | 0.847 | 6 |
| A14 | 19 | 0.657 | 6 |

A ratio near 1.0 means inputs rarely share phrasing; a low ratio flags a templated class (expected for enforcement notices). Internal repeats usually mean leaked navigation or a running header the furniture cleaner missed.

## Dates and label sources

- train: 1948-01-01 → 2026-07-02 (0 undated); labels {'model': 1683}
- val: 2025-11-26 → 2026-07-31 (0 undated); labels {'model': 218}
- test: 2026-07-01 → 2026-09-18 (0 undated); labels {'model': 104}

## Corpus (from coverage.json)

- documents 6642, chunks 212620
- deduped {'content_sha256': 394, 'pdf_url': 24, 'external_id': 0, 'near_duplicate': 185}
- furniture {'docs_cleaned': 5812, 'chars_removed': 1863690, 'tail_cuts': 1393}
- skipped {'garbled text': 8}
