# UK alert-triage dataset v1

Generated 2026-09-25T05:32:55.759825+00:00 from `data/uk/labels/triage_v1.jsonl` (tiers GREEN).

| split | records |
|---|---:|
| train | 1683 |
| val | 218 |
| test | 104 |
| all | 1901 |

## Train balance vs target

| stratum | available | cap | kept | shortfall | target |
|---|---:|---:|---:|---:|---:|
| rule_final | 940 | 940 | 940 | 0 | 35% |
| consultation | 191 | 537 | 191 | 346 | 20% |
| guidance | 68 | 403 | 68 | 335 | 15% |
| enforcement | 81 | 403 | 81 | 322 | 15% |
| intelligence | 682 | 403 | 403 | 0 | 15% |

Test = latest 2 months per stratum; val = next 10% by date; undated rows train. Gold candidates: 995 rows in `/Users/karthigeyanrj/Documents/17-Comply2Reg/15-SLM/ComplianceGPT/data/uk/gold/gold_candidates_v1.csv`.

Record envelope: metadata, classification (alert class A1–A14), instruction, input_text (title + first head-chars of canonical text), tool_use (null), thought_trace (model rationale), output (TriageRecord without rationale), plus `messages` (system/user/assistant) and `n_tokens` measured with Qwen/Qwen3-4B-Instruct-2507 against a 2048-token budget (0 contexts trimmed). AMBER-tier records are internal use only and never leave `datasets/internal/`.
