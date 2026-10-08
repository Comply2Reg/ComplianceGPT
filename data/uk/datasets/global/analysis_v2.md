# Dataset analysis — global v2

Target model `qwen3-4b-instruct` (Qwen/Qwen3-4B-Instruct-2507), budget 2048 tokens; token counter: records' n_tokens.

## Records per split

| split | records | strata | classes | regulators |
|---|---|---|---|---|
| train | 4114 | {'rule_final': 1791, 'intelligence': 648, 'guidance': 768, 'consultation': 814, 'enforcement': 93} | {'A11': 1223, 'A2': 569, 'A1': 559, 'A3': 824, 'A7': 327, 'A14': 167, 'A9': 28, 'A8': 195, 'A12': 59, 'A6': 80, 'A4': 37, 'A5': 6, 'A10': 24, 'A13': 16} | {'HMT': 795, 'CMA': 555, 'SEC': 233, 'TREAS': 599, 'NCUA': 108, 'OFAC': 359, 'CFTC': 149, 'CFPB': 219, 'FRB': 228, 'FDIC': 132, 'FFIEC': 17, 'OCC': 136, 'UKLEG': 518, 'FINCEN': 53, 'ICO': 13} |
| val | 697 | {'rule_final': 199, 'consultation': 90, 'intelligence': 72, 'enforcement': 10, 'guidance': 326} | {'A2': 103, 'A8': 48, 'A1': 17, 'A11': 202, 'A12': 16, 'A4': 6, 'A7': 44, 'A3': 189, 'A14': 35, 'A6': 10, 'A9': 10, 'A13': 1, 'A10': 10, 'A5': 6} | {'FRB': 43, 'CFPB': 11, 'CFTC': 22, 'TREAS': 80, 'HMT': 131, 'UKLEG': 10, 'OCC': 15, 'SEC': 213, 'CMA': 72, 'FDIC': 31, 'FINCEN': 9, 'NCUA': 13, 'OFAC': 46, 'ICO': 1} |
| test | 759 | {'rule_final': 57, 'consultation': 54, 'intelligence': 66, 'enforcement': 45, 'guidance': 537} | {'A2': 67, 'A11': 184, 'A12': 15, 'A8': 77, 'A1': 1, 'A7': 33, 'A6': 45, 'A3': 239, 'A14': 60, 'A9': 15, 'A13': 3, 'A10': 9, 'A5': 9, 'A4': 2} | {'TREAS': 78, 'HMT': 27, 'CMA': 63, 'NCUA': 13, 'SEC': 423, 'FINCEN': 4, 'OFAC': 35, 'CFTC': 27, 'OCC': 14, 'FDIC': 30, 'FRB': 42, 'CFPB': 2, 'ICO': 1} |

## Token lengths

| split | p50 | p95 | max | over 2048 | context trimmed |
|---|---|---|---|---|---|
| train | 836 | 1026 | 1384 | 0 | 0 |
| val | 847 | 1019 | 1180 | 0 | 0 |
| test | 867 | 1029 | 1824 | 0 | 0 |

Whole-document tokens (for a long-context variant): p50 2147, p95 58503, max 843206 — 4266/5570 fit in 8k, 4777/5570 in 16k.

## Class coverage (train)

| class | label | train | val | test |
|---|---|---|---|---|
| A1 | Primary legislation / statutory instrument | 559 | 17 | 1 |
| A2 | Final rule / policy statement | 569 | 103 | 67 |
| A3 | Proposed rule / consultation / discussion paper  | 824 | 189 | 239 |
| A4 | Supervisory guidance (supervisory statement, sta | 37 | 6 | 2 |
| A5 | Supervisory communication (letter to firms, port | 6 | 6 | 9 |
| A6 | Enforcement action (notice, order, consent order | 80 | 10 | 45 |
| A7 | Sanctions / watchlist delta | 327 | 44 | 33 |
| A8 | Reporting, returns or taxonomy change | 195 | 48 | 77 |
| A9 | Market or operational notice | 28 | 10 | 15 |
| A10 | Thematic review, multi-firm review or market stu | 24 | 10 | 9 |
| A11 | Intelligence: speech, blog, press release, resea | 1223 | 202 | 184 |
| A12 | Codified rulebook, handbook or code, point-in-ti | 59 | 16 | 15 |
| A13 | Court or tribunal determination | 16 | 1 | 3 |
| A14 | Perimeter, register or authorisation change (war | 167 | 35 | 60 |

Classes with fewer than 30 training examples: A5, A9, A10, A13. A model cannot learn a class it has not seen; these need more sources (see docs/uk-datasets.md) or must be excluded from the evaluation claims.

## Train mix vs target

| stratum | records | share | target |
|---|---|---|---|
| rule_final | 1791 | 43.5% | 35% |
| consultation | 814 | 19.8% | 20% |
| guidance | 768 | 18.7% | 15% |
| enforcement | 93 | 2.3% | 15% |
| intelligence | 648 | 15.8% | 15% |

## Diversity and repetition (train inputs)

| class | records | unique 8-gram ratio | records with an internal 10-word repeat |
|---|---|---|---|
| A1 | 559 | 0.669 | 495 |
| A2 | 569 | 0.688 | 306 |
| A3 | 824 | 0.620 | 440 |
| A4 | 37 | 0.868 | 18 |
| A5 | 6 | 0.823 | 3 |
| A6 | 80 | 0.799 | 34 |
| A7 | 327 | 0.334 | 249 |
| A8 | 195 | 0.379 | 122 |
| A9 | 28 | 0.769 | 13 |
| A10 | 24 | 0.794 | 9 |
| A11 | 1223 | 0.717 | 682 |
| A12 | 59 | 0.820 | 25 |
| A13 | 16 | 0.849 | 8 |
| A14 | 167 | 0.326 | 109 |

A ratio near 1.0 means inputs rarely share phrasing; a low ratio flags a templated class (expected for enforcement notices). Internal repeats usually mean leaked navigation or a running header the furniture cleaner missed.

## Dates and label sources

- train: 1948-01-01 → 2026-06-30 (0 undated); labels {'model': 4114}
- val: 2025-12-15 → 2026-07-31 (0 undated); labels {'model': 697}
- test: 2026-08-03 → 2026-09-28 (0 undated); labels {'model': 759}
