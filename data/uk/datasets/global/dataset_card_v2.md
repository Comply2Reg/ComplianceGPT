# UK alert-triage dataset v2

Generated 2026-09-26T19:59:29.748602+00:00 from `[PosixPath('data/uk/labels/triage_uk_v2.jsonl'), PosixPath('data/uk/labels/triage_us_v1.jsonl')]` (tiers GREEN).

| split | records |
|---|---:|
| train | 4114 |
| val | 697 |
| test | 759 |
| all | 4811 |

## Train balance vs target

| stratum | available | cap | kept | shortfall | target |
|---|---:|---:|---:|---:|---:|
| rule_final | 1791 | 1791 | 1791 | 0 | 35% |
| consultation | 814 | 1023 | 814 | 209 | 20% |
| guidance | 2932 | 768 | 768 | 0 | 15% |
| enforcement | 93 | 768 | 93 | 675 | 15% |
| intelligence | 648 | 768 | 648 | 120 | 15% |

Test = latest 2 months per stratum; val = next 10% by date; undated rows train. Gold candidates: 2426 rows in `/Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT/data/uk/gold/gold_candidates_v2.csv`.

Record envelope: metadata, classification (alert class A1–A14), instruction, input_text (title + first head-chars of canonical text), tool_use (null), thought_trace (model rationale), output (TriageRecord without rationale), plus `messages` (system/user/assistant) and `n_tokens` measured with Qwen/Qwen3-4B-Instruct-2507 against a 2048-token budget (0 contexts trimmed). AMBER-tier records are internal use only and never leave `datasets/internal/`.
