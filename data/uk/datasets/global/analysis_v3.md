# Dataset analysis — global v3

Target model `qwen3-4b-instruct` (Qwen/Qwen3-4B-Instruct-2507), budget 2048 tokens; token counter: records' n_tokens.

## Records per split

| split | records | strata | classes | regulators |
|---|---|---|---|---|
| train | 3194 | {'rule_final': 1076, 'intelligence': 186, 'guidance': 1479, 'consultation': 360, 'enforcement': 93} | {'A11': 400, 'A2': 400, 'A1': 400, 'A7': 400, 'A14': 400, 'A3': 400, 'A8': 400, 'A9': 49, 'A12': 60, 'A10': 80, 'A5': 19, 'A6': 80, 'A4': 90, 'A13': 16} | {'HMT': 222, 'SEC': 138, 'TREAS': 397, 'CMA': 381, 'OFAC': 421, 'FRB': 424, 'NCUA': 109, 'CFTC': 169, 'OCC': 170, 'FFIEC': 20, 'FDIC': 170, 'UKLEG': 372, 'CFPB': 155, 'FINCEN': 39, 'ICO': 7} |
| val | 697 | {'rule_final': 199, 'consultation': 90, 'intelligence': 72, 'enforcement': 10, 'guidance': 326} | {'A2': 103, 'A8': 48, 'A1': 17, 'A11': 202, 'A12': 16, 'A4': 6, 'A7': 44, 'A3': 189, 'A14': 35, 'A6': 10, 'A9': 10, 'A13': 1, 'A10': 10, 'A5': 6} | {'FRB': 43, 'CFPB': 11, 'CFTC': 22, 'TREAS': 80, 'HMT': 131, 'UKLEG': 10, 'OCC': 15, 'SEC': 213, 'CMA': 72, 'FDIC': 31, 'FINCEN': 9, 'NCUA': 13, 'OFAC': 46, 'ICO': 1} |
| test | 759 | {'rule_final': 57, 'consultation': 54, 'intelligence': 66, 'enforcement': 45, 'guidance': 537} | {'A2': 67, 'A11': 184, 'A12': 15, 'A8': 77, 'A1': 1, 'A7': 33, 'A6': 45, 'A3': 239, 'A14': 60, 'A9': 15, 'A13': 3, 'A10': 9, 'A5': 9, 'A4': 2} | {'TREAS': 78, 'HMT': 27, 'CMA': 63, 'NCUA': 13, 'SEC': 423, 'FINCEN': 4, 'OFAC': 35, 'CFTC': 27, 'OCC': 14, 'FDIC': 30, 'FRB': 42, 'CFPB': 2, 'ICO': 1} |

## Token lengths

| split | p50 | p95 | max | over 2048 | context trimmed |
|---|---|---|---|---|---|
| train | 1050 | 1236 | 1499 | 0 | 0 |
| val | 1060 | 1232 | 1393 | 0 | 0 |
| test | 1080 | 1242 | 2037 | 0 | 0 |

Whole-document tokens (for a long-context variant): p50 1741, p95 43693, max 843206 — 3783/4650 fit in 8k, 4130/4650 in 16k.

## Class coverage (train)

| class | label | train | val | test |
|---|---|---|---|---|
| A1 | Primary legislation / statutory instrument | 400 | 17 | 1 |
| A2 | Final rule / policy statement | 400 | 103 | 67 |
| A3 | Proposed rule / consultation / discussion paper  | 400 | 189 | 239 |
| A4 | Supervisory guidance (supervisory statement, sta | 90 | 6 | 2 |
| A5 | Supervisory communication (letter to firms, port | 19 | 6 | 9 |
| A6 | Enforcement action (notice, order, consent order | 80 | 10 | 45 |
| A7 | Sanctions / watchlist delta | 400 | 44 | 33 |
| A8 | Reporting, returns or taxonomy change | 400 | 48 | 77 |
| A9 | Market or operational notice | 49 | 10 | 15 |
| A10 | Thematic review, multi-firm review or market stu | 80 | 10 | 9 |
| A11 | Intelligence: speech, blog, press release, resea | 400 | 202 | 184 |
| A12 | Codified rulebook, handbook or code, point-in-ti | 60 | 16 | 15 |
| A13 | Court or tribunal determination | 16 | 1 | 3 |
| A14 | Perimeter, register or authorisation change (war | 400 | 35 | 60 |

Classes with fewer than 30 training examples: A5, A13. A model cannot learn a class it has not seen; these need more sources (see docs/uk-datasets.md) or must be excluded from the evaluation claims.

## Train mix vs target

| stratum | records | share | target |
|---|---|---|---|
| rule_final | 1076 | 33.7% | 35% |
| consultation | 360 | 11.3% | 20% |
| guidance | 1479 | 46.3% | 15% |
| enforcement | 93 | 2.9% | 15% |
| intelligence | 186 | 5.8% | 15% |

## Diversity and repetition (train inputs)

| class | records | unique 8-gram ratio | records with an internal 10-word repeat |
|---|---|---|---|
| A1 | 400 | 0.695 | 350 |
| A2 | 400 | 0.724 | 198 |
| A3 | 400 | 0.704 | 222 |
| A4 | 90 | 0.849 | 37 |
| A5 | 19 | 0.620 | 11 |
| A6 | 80 | 0.799 | 34 |
| A7 | 400 | 0.248 | 296 |
| A8 | 400 | 0.304 | 254 |
| A9 | 49 | 0.760 | 25 |
| A10 | 80 | 0.735 | 41 |
| A11 | 400 | 0.688 | 202 |
| A12 | 60 | 0.823 | 25 |
| A13 | 16 | 0.849 | 8 |
| A14 | 400 | 0.193 | 315 |

A ratio near 1.0 means inputs rarely share phrasing; a low ratio flags a templated class (expected for enforcement notices). Internal repeats usually mean leaked navigation or a running header the furniture cleaner missed.

## Dates and label sources

- train: 1948-01-01 → 2026-06-30 (0 undated); labels {'model': 3194}
- val: 2025-12-15 → 2026-07-31 (0 undated); labels {'model': 697}
- test: 2026-08-03 → 2026-09-28 (0 undated); labels {'model': 759}
