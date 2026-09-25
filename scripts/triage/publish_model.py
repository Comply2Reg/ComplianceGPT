#!/usr/bin/env python3
"""Validate a fused model directory and publish it to the Hugging Face Hub.

Publishing is the one step in this pipeline that leaves the machine, so it is
gated: --dry-run makes no network call and fails on anything the Hub would
accept but a reader should not — a card whose stated metrics disagree with the
evaluation artefacts, a missing weight file, an absent licence.

    # check everything, touch nothing
    python -m scripts.triage.publish_model \\
        --model-dir models/publish/regulatory-alert-triage-qwen3-4b-v1 --dry-run

    # create the private repo and upload
    python -m scripts.triage.publish_model \\
        --model-dir models/publish/regulatory-alert-triage-qwen3-4b-v1 \\
        --repo-id Comply2Reg/regulatory-alert-triage-qwen3-4b-v1

Needs HF_TOKEN in .env (or the environment) with write access to the org.
Repositories are created PRIVATE. Going public is a separate, deliberate act:
it needs --public together with --i-have-checked-licensing, because the
training corpus carries attribution obligations that a typo should not publish
its way past.

Note: mlx_lm's own `fuse --upload-repo` is not used anywhere in this pipeline.
It overwrites the card's body with generic boilerplate, discarding the model
card entirely.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402

log = logging.getLogger("triage.publish_model")

# Without these the repo is not a loadable model, whatever else is present.
REQUIRED_FILES = (
    "README.md",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
WEIGHT_SUFFIXES = (".safetensors", ".npz", ".bin")
REQUIRED_FRONTMATTER = ("license", "base_model", "library_name", "pipeline_tag")
# Hub limits a single file to 50 GB; a card that large is a bug, not a card.
MAX_CARD_BYTES = 200_000


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GB"


def split_frontmatter(text: str) -> Tuple[Dict, str]:
    """YAML frontmatter and body of a model card, without importing yaml."""
    import yaml  # a hub dependency; imported late so --help works without it

    if not text.startswith("---"):
        raise ValueError("model card has no YAML frontmatter")
    end = text.find("\n---", 3)
    if end < 0:
        raise ValueError("model card frontmatter is not terminated")
    return yaml.safe_load(text[3:end]), text[end + 4 :]


def card_metrics(meta: Dict) -> Dict[str, float]:
    """{metric name: value} from a model-index block."""
    out: Dict[str, float] = {}
    for entry in meta.get("model-index") or []:
        for result in entry.get("results") or []:
            for metric in result.get("metrics") or []:
                out[metric["name"]] = float(metric["value"])
    return out


def validate(model_dir: Path) -> List[str]:
    """Every problem with the directory, rather than only the first."""
    problems: List[str] = []
    if not model_dir.is_dir():
        return [f"{model_dir} is not a directory"]

    for name in REQUIRED_FILES:
        if not (model_dir / name).exists():
            problems.append(f"missing {name}")

    weights = [
        p
        for p in model_dir.iterdir()
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES
    ]
    if not weights:
        problems.append("no weight file (.safetensors/.npz/.bin)")

    card_path = model_dir / "README.md"
    if card_path.exists():
        text = card_path.read_text(encoding="utf-8")
        if len(text.encode("utf-8")) > MAX_CARD_BYTES:
            problems.append("model card is implausibly large")
        try:
            meta, body_text = split_frontmatter(text)
        except (ValueError, ImportError) as exc:
            problems.append(f"model card frontmatter: {exc}")
            return problems
        for field in REQUIRED_FRONTMATTER:
            if not meta.get(field):
                problems.append(f"model card frontmatter is missing '{field}'")
        if len(body_text.strip()) < 2000:
            problems.append(
                "model card body is too short to describe a model honestly"
            )
        for heading in ("Limitations", "Training data", "Evaluation"):
            if f"## {heading}" not in body_text:
                problems.append(f"model card has no '{heading}' section")

        # The card's numbers must be the evaluation's numbers.
        summary_path = model_dir / "eval_summary.json"
        if not summary_path.exists():
            problems.append("missing eval_summary.json to check the card against")
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            declared = card_metrics(meta)
            if not declared:
                problems.append("model card declares no metrics in model-index")
            measured = summary.get("metrics", {})
            aliases = {
                "alert_class accuracy": "alert_class_accuracy",
                "alert_class macro-F1": "alert_class_macro_f1",
                "priority accuracy": "priority_accuracy",
                "obligations_present accuracy": "obligations_present_accuracy",
                "primary_functions Jaccard": "primary_function_jaccard",
                "JSON validity": "json_valid_rate",
            }
            for name, value in declared.items():
                key = aliases.get(name)
                if key is None:
                    problems.append(f"card declares unknown metric '{name}'")
                    continue
                actual = measured.get(key)
                if actual is None:
                    problems.append(f"no measured value for '{name}'")
                elif abs(float(actual) - value) > 1e-6:
                    problems.append(
                        f"card says {name}={value} but the evaluation says {actual}"
                    )
            fusion = summary.get("fusion") or {}
            if fusion and not fusion.get("passed", True):
                problems.append(
                    "the fused model regressed against its adapter beyond "
                    f"tolerance: {fusion.get('regressions')}"
                )
    return problems


def manifest(model_dir: Path) -> Tuple[List[Tuple[str, int]], int]:
    files = sorted(
        (p.relative_to(model_dir).as_posix(), p.stat().st_size)
        for p in model_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )
    return files, sum(size for _, size in files)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--repo-id", default="Comply2Reg/regulatory-alert-triage-qwen3-4b-v1")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the manifest; make no network call",
    )
    ap.add_argument(
        "--public",
        action="store_true",
        help="create a PUBLIC repository (requires --i-have-checked-licensing)",
    )
    ap.add_argument(
        "--i-have-checked-licensing",
        action="store_true",
        help="confirm the training-data licences permit public release",
    )
    ap.add_argument(
        "--commit-message", default="Publish UK alert triage model v1 (fused, 4-bit)"
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg.load_env()

    problems = validate(args.model_dir)
    files, total = manifest(args.model_dir)

    print(f"\n{args.model_dir} -> {args.repo_id}")
    print(f"{'PRIVATE' if not args.public else 'PUBLIC'} repository\n")
    for name, size in files:
        print(f"  {human(size):>10}  {name}")
    print(f"  {'':>10}  {len(files)} files, {human(total)} total\n")

    if problems:
        for p in problems:
            log.error("%s", p)
        log.error("%d problem(s) — nothing published", len(problems))
        return 1
    log.info("validation passed")

    if args.dry_run:
        log.info("dry run: no network call made")
        return 0

    if args.public and not args.i_have_checked_licensing:
        log.error(
            "--public also needs --i-have-checked-licensing: the training corpus "
            "carries attribution obligations, so a public release is a decision, "
            "not a flag"
        )
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        log.error(
            "no HF_TOKEN. Create a write-scoped token at "
            "https://huggingface.co/settings/tokens and put it in .env as "
            "HF_TOKEN=... (the file is gitignored)"
        )
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=not args.public,
        exist_ok=True,
    )
    log.info("repository ready: https://huggingface.co/%s", args.repo_id)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.model_dir),
        commit_message=args.commit_message,
    )
    info = api.model_info(args.repo_id, files_metadata=False)
    log.info(
        "published %s (private=%s, %d files) -> https://huggingface.co/%s",
        args.repo_id,
        info.private,
        len(info.siblings or []),
        args.repo_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
