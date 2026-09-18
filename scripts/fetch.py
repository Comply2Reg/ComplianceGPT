"""Quarantined. This entry point used to overwrite data/chunks.jsonl.

  python scripts/experiments/fetch.py   # download XML only
  python scripts/clml.py --xml PATH     # parse P1/P2/P3 with source-derived IDs
"""

raise SystemExit(
    "scripts/fetch.py is quarantined — it used to overwrite data/chunks.jsonl.\n"
    "Download XML:  python scripts/experiments/fetch.py\n"
    "Parse CLML:    python scripts/clml.py --xml data/raw/fsma2000.xml"
)
