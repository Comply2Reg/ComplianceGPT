# Triage labels report — `triage_v1.jsonl`

2284 labelled documents; model gpt-4o-2024-08-06, prompt triage-v1.

## Alert class distribution

| class | label | documents | share |
|---|---|---|---|
| A1 | Primary legislation / statutory instrument | 602 | 26.4% |
| A2 | Final rule / policy statement | 22 | 1.0% |
| A3 | Proposed rule / consultation / discussion paper / call for input | 228 | 10.0% |
| A4 | Supervisory guidance (supervisory statement, statement of policy, finalised guidance) | 30 | 1.3% |
| A5 | Supervisory communication (Dear CEO / portfolio letter, supervisory priorities) | 0 | 0.0% |
| A6 | Enforcement action (final, decision or warning notice, penalty) | 77 | 3.4% |
| A7 | Sanctions / watchlist delta | 4 | 0.2% |
| A8 | Reporting, returns or taxonomy change | 4 | 0.2% |
| A9 | Market or operational notice | 41 | 1.8% |
| A10 | Thematic review, multi-firm review or market study | 66 | 2.9% |
| A11 | Intelligence: speech, blog, press release, research, working paper | 1117 | 48.9% |
| A12 | Codified rulebook or handbook, point-in-time | 31 | 1.4% |
| A13 | Court or tribunal determination | 24 | 1.1% |
| A14 | Perimeter, register or authorisation change (warning list, waivers) | 38 | 1.7% |

## Stratum (from model class) vs target

| stratum | documents | share | target |
|---|---|---|---|
| rule_final | 659 | 28.9% | 35% |
| consultation | 228 | 10.0% | 20% |
| guidance | 100 | 4.4% | 15% |
| enforcement | 101 | 4.4% | 15% |
| intelligence | 1196 | 52.4% | 15% |

## Priority and confidence

| priority | documents |
|---|---|
| P1 | 708 |
| P2 | 315 |
| P3 | 1261 |

| confidence | documents |
|---|---|
| high | 2200 |
| low | 7 |
| medium | 77 |

## Agreement with the registry class

976/1481 = 65.9% on sources with a fixed class (MIXED sources excluded).

| source | agree | of | share |
|---|---|---|---|
| HMT:policy_papers | 16 | 501 | 3.2% |
| HMT:financial_services_regulations | 145 | 152 | 95.4% |
| HMT:news | 287 | 300 | 95.7% |
| UKLEG:statutory_instruments | 528 | 528 | 100.0% |

Top disagreements (registry → model):

| registry | model | documents |
|---|---|---|
| A2 | A11 | 386 |
| A2 | A1 | 63 |
| A2 | A12 | 16 |
| A2 | A3 | 14 |
| A11 | A7 | 4 |
| A2 | A9 | 3 |
| A3 | A11 | 3 |
| A2 | A8 | 2 |

## How the 803 unclassified (MIXED) documents resolved

| regulator | class | documents |
|---|---|---|
| CMA | A11 | 428 |
| CMA | A6 | 75 |
| CMA | A3 | 65 |
| CMA | A10 | 64 |
| CMA | A9 | 38 |
| CMA | A14 | 37 |
| CMA | A4 | 29 |
| CMA | A13 | 24 |
| CMA | A12 | 15 |
| ICO | A11 | 13 |
| CMA | A1 | 8 |
| CMA | A8 | 2 |
| CMA | A2 | 2 |
| ICO | A3 | 2 |
| ICO | A4 | 1 |

## Agreement with weak (registry + title heuristic) labels

1026/1735 = 59.1%.

## Model vs model (documents labelled by both)

**gpt-4o-2024-08-06** vs **gpt-4o-mini** — 351 documents

| field | agreement |
|---|---|
| alert_class | 71.5% |
| priority | 69.2% |
| obligations_present | 79.8% |
| primary_functions (Jaccard) | 0.596 |

Top class disagreements (gpt-4o-2024-08-06 → gpt-4o-mini):

| gpt-4o-2024-08-06 | gpt-4o-mini | documents |
|---|---|---|
| A11 | A2 | 18 |
| A11 | A3 | 17 |
| A11 | A1 | 10 |
| A10 | A3 | 8 |
| A11 | A4 | 8 |
| A9 | A1 | 7 |
| A11 | A6 | 5 |
| A11 | A5 | 4 |

**gpt-4o-2024-08-06** vs **gpt-5.4-mini** — 351 documents

| field | agreement |
|---|---|
| alert_class | 80.3% |
| priority | 84.9% |
| obligations_present | 93.7% |
| primary_functions (Jaccard) | 0.672 |

Top class disagreements (gpt-4o-2024-08-06 → gpt-5.4-mini):

| gpt-4o-2024-08-06 | gpt-5.4-mini | documents |
|---|---|---|
| A4 | A11 | 9 |
| A3 | A10 | 6 |
| A5 | A11 | 5 |
| A10 | A11 | 4 |
| A3 | A11 | 4 |
| A6 | A11 | 3 |
| A11 | A6 | 3 |
| A2 | A1 | 3 |

**gpt-4o-mini** vs **gpt-5.4** — 20 documents

| field | agreement |
|---|---|
| alert_class | 0.0% |
| priority | 0.0% |
| obligations_present | 40.0% |
| primary_functions (Jaccard) | 0.0 |

Top class disagreements (gpt-4o-mini → gpt-5.4):

| gpt-4o-mini | gpt-5.4 | documents |
|---|---|---|
| A2 | A11 | 14 |
| A3 | A11 | 5 |
| A4 | A11 | 1 |

**gpt-4o-mini** vs **gpt-5.4-mini** — 2284 documents

| field | agreement |
|---|---|
| alert_class | 58.8% |
| priority | 61.9% |
| obligations_present | 74.6% |
| primary_functions (Jaccard) | 0.52 |

Top class disagreements (gpt-4o-mini → gpt-5.4-mini):

| gpt-4o-mini | gpt-5.4-mini | documents |
|---|---|---|
| A2 | A11 | 249 |
| A3 | A11 | 245 |
| A4 | A11 | 96 |
| A3 | A10 | 47 |
| A1 | A11 | 44 |
| A1 | A9 | 31 |
| A6 | A11 | 26 |
| A2 | A1 | 25 |

**gpt-5.4** vs **gpt-5.4-mini** — 20 documents

| field | agreement |
|---|---|
| alert_class | 90.0% |
| priority | 95.0% |
| obligations_present | 100.0% |
| primary_functions (Jaccard) | 0.8 |

Top class disagreements (gpt-5.4 → gpt-5.4-mini):

| gpt-5.4 | gpt-5.4-mini | documents |
|---|---|---|
| A11 | A8 | 1 |
| A11 | A1 | 1 |

951 documents where two models chose different alert classes -> `data/uk/labels/disputed_v1.txt` (feed to `labeller --model <strong> --ids-file`).


## Flags

| flag | documents |
|---|---|
| function_off_matrix | 1132 |
| no_primary_function | 1084 |
| class_disagrees_with_registry | 505 |
| low_confidence | 7 |

## By regulator and tier

| regulator | tier | documents |
|---|---|---|
| CMA | GREEN | 787 |
| HMT | GREEN | 953 |
| ICO | GREEN | 16 |
| UKLEG | GREEN | 528 |

## Cost

```
{
  "version": "v1",
  "model": "gpt-4o-2024-08-06",
  "prompt_version": "triage-v1",
  "runs": 7,
  "documents": 2284,
  "calls_this_run": 0,
  "cached_this_run": 2284,
  "failures": 0,
  "prompt_tokens": 6707726,
  "completion_tokens": 540511,
  "models": [
    "gpt-5.4",
    "gpt-5.4-mini"
  ],
  "missing": 0,
  "shard": null,
  "tiers": [
    "GREEN"
  ],
  "counsel_signoff": null,
  "updated_at": "2026-09-25T05:32:31.880429+00:00",
  "estimated_usd": 7.81431,
  "by_model": {
    "gpt-5.4": {
      "documents": 20,
      "prompt_tokens": 56773,
      "completion_tokens": 7094,
      "estimated_usd": 0.425719
    },
    "gpt-5.4-mini": {
      "documents": 2264,
      "prompt_tokens": 6650953,
      "completion_tokens": 533417,
      "estimated_usd": 7.388591
    }
  }
}
```
