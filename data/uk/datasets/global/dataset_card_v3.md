# UK alert-triage dataset v3

Generated 2026-09-27T09:18:08.253443+00:00 from `[PosixPath('data/uk/labels/triage_uk_v2.jsonl'), PosixPath('data/uk/labels/triage_us_v1.jsonl')]` (tiers GREEN).

| split | records |
|---|---:|
| train | 3194 |
| val | 697 |
| test | 759 |
| all | 3891 |

## Train balance: each class capped at 400

| class | available | cap | kept | shortfall |
|---|---:|---:|---:|---:|
| A1 | 560 | 400 | 400 | 0 |
| A10 | 80 | 400 | 80 | 320 |
| A11 | 1974 | 400 | 400 | 0 |
| A12 | 60 | 400 | 60 | 340 |
| A13 | 16 | 400 | 16 | 384 |
| A14 | 493 | 400 | 400 | 0 |
| A2 | 572 | 400 | 400 | 0 |
| A3 | 882 | 400 | 400 | 0 |
| A4 | 90 | 400 | 90 | 310 |
| A5 | 19 | 400 | 19 | 381 |
| A6 | 80 | 400 | 80 | 320 |
| A7 | 743 | 400 | 400 | 0 |
| A8 | 660 | 400 | 400 | 0 |
| A9 | 49 | 400 | 49 | 351 |

Resulting stratum mix (not targeted in this mode):

| stratum | available | kept |
|---|---:|---:|
| rule_final | 1791 | 1076 |
| consultation | 814 | 360 |
| guidance | 2932 | 1479 |
| enforcement | 93 | 93 |
| intelligence | 648 | 186 |

Test = latest 2 months per stratum; val = next 10% by date; undated rows train. Gold candidates: 2409 rows in `/Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT/data/uk/gold/gold_candidates_v3.csv`.

Record envelope: metadata, classification (alert class A1–A14), instruction, input_text (title + first head-chars of canonical text), tool_use (null), thought_trace (model rationale), output (TriageRecord without rationale), plus `messages` (system/user/assistant) and `n_tokens` measured with Qwen/Qwen3-4B-Instruct-2507 against a 2048-token budget (0 contexts trimmed). AMBER-tier records are internal use only and never leave `datasets/internal/`.
