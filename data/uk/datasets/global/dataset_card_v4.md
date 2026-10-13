# UK alert-triage dataset v4

Generated 2026-10-13T06:03:01.366502+00:00 from `[PosixPath('data/uk/labels/triage_uk_v2.jsonl'), PosixPath('data/uk/labels/triage_us_v1.jsonl')]` (tiers GREEN).

| split | records |
|---|---:|
| train | 3194 |
| val | 697 |
| test | 759 |
| all | 3891 |

## Train balance: caps proportional to available^0.5, target 3194

| class | available | cap | kept | shortfall |
|---|---:|---:|---:|---:|
| A1 | 560 | 337 | 337 | 0 |
| A10 | 80 | 80 | 80 | 0 |
| A11 | 1974 | 632 | 632 | 0 |
| A12 | 60 | 60 | 60 | 0 |
| A13 | 16 | 16 | 16 | 0 |
| A14 | 493 | 316 | 316 | 0 |
| A2 | 572 | 340 | 340 | 0 |
| A3 | 882 | 422 | 422 | 0 |
| A4 | 90 | 90 | 90 | 0 |
| A5 | 19 | 19 | 19 | 0 |
| A6 | 80 | 80 | 80 | 0 |
| A7 | 743 | 388 | 388 | 0 |
| A8 | 660 | 365 | 365 | 0 |
| A9 | 49 | 49 | 49 | 0 |

Resulting stratum mix (not targeted in this mode):

| stratum | available | kept |
|---|---:|---:|
| rule_final | 1791 | 988 |
| consultation | 814 | 381 |
| guidance | 2932 | 1490 |
| enforcement | 93 | 93 |
| intelligence | 648 | 242 |

Test = latest 2 months per stratum; val = next 10% by date; undated rows train. Gold candidates: 2482 rows in `/Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT/data/uk/gold/gold_candidates_v4.csv`.

Record envelope: metadata, classification (alert class A1–A14), instruction, input_text (title + first head-chars of canonical text), tool_use (null), thought_trace (model rationale), output (TriageRecord without rationale), plus `messages` (system/user/assistant) and `n_tokens` measured with Qwen/Qwen3-4B-Instruct-2507 against a 2048-token budget (0 contexts trimmed). AMBER-tier records are internal use only and never leave `datasets/internal/`.
