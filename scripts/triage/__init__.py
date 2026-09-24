"""Alert-triage labelling and dataset building for the UK corpus.

Input is the Stage-1 corpus c2r-inventory-kit exports (`canonical/`,
`chunks.jsonl`, `manifest.jsonl`); output is training records in the same
7-key envelope the obligation pipeline on `main` trains from, so the Gemma
notebook runs unchanged. See docs/uk-datasets.md.
"""
