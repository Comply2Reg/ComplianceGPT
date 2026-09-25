"""The model card and the publish gate.

No network and no 2 GB of weights: the fixtures are a directory shaped like a
fused model, so the validator is exercised on the failures that matter — a card
whose numbers disagree with the evaluation, a missing weight file, a card with
no limitations section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.triage import model_card, publish_model  # noqa: E402

METRICS = {
    "alert_class_accuracy": 0.6442,
    "alert_class_macro_f1": 0.2445,
    "priority_accuracy": 0.7596,
    "obligations_present_accuracy": 0.9135,
    "primary_function_jaccard": 0.5744,
    "json_valid_rate": 1.0,
}
CARD_SECTIONS = """
## Training data

Documents from UK public bodies under the Open Government Licence.
""" + ("Filler prose that makes the body long enough to be a real card. " * 40) + """

## Evaluation

104 held-out documents.

## Limitations

It has not learned most of the taxonomy.
"""


def _card(metrics: dict, *, license_: str = "apache-2.0", body: str = CARD_SECTIONS) -> str:
    rows = "\n".join(
        f"      - type: accuracy\n        name: {name}\n        value: {value}"
        for name, value in metrics.items()
    )
    return f"""---
license: {license_}
base_model: mlx-community/Qwen3-4B-Instruct-2507-4bit
library_name: mlx
pipeline_tag: text-generation
model-index:
- name: m
  results:
  - task:
      type: text-classification
      name: t
    dataset:
      type: private
      name: d
      split: test
    metrics:
{rows}
---
# Model
{body}
"""


NAMES = {
    "alert_class accuracy": "alert_class_accuracy",
    "alert_class macro-F1": "alert_class_macro_f1",
    "priority accuracy": "priority_accuracy",
    "obligations_present accuracy": "obligations_present_accuracy",
    "primary_functions Jaccard": "primary_function_jaccard",
    "JSON validity": "json_valid_rate",
}


def _model_dir(tmp_path: Path, *, card_metrics: dict | None = None) -> Path:
    d = tmp_path / "uk-alert-triage-qwen3-4b-v1"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"\x00" * 16)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    declared = card_metrics if card_metrics is not None else {
        display: METRICS[key] for display, key in NAMES.items()
    }
    (d / "README.md").write_text(_card(declared), encoding="utf-8")
    (d / "eval_summary.json").write_text(
        json.dumps({"metrics": METRICS, "fusion": {"passed": True}}),
        encoding="utf-8",
    )
    return d


# ── the publish gate ────────────────────────────────────────────────────────


def test_a_well_formed_model_directory_validates(tmp_path: Path) -> None:
    assert publish_model.validate(_model_dir(tmp_path)) == []


def test_a_card_claiming_a_score_the_evaluation_does_not_support_is_rejected(
    tmp_path: Path,
) -> None:
    inflated = {display: METRICS[key] for display, key in NAMES.items()}
    inflated["alert_class accuracy"] = 0.95  # the number a reader would act on
    problems = publish_model.validate(_model_dir(tmp_path, card_metrics=inflated))
    assert any("0.95" in p and "0.6442" in p for p in problems), problems


def test_missing_weights_and_missing_card_sections_are_both_reported(
    tmp_path: Path,
) -> None:
    d = _model_dir(tmp_path)
    (d / "model.safetensors").unlink()
    (d / "README.md").write_text(
        _card(
            {display: METRICS[key] for display, key in NAMES.items()},
            body="\n## Training data\n\n" + "short. " * 400,
        ),
        encoding="utf-8",
    )
    problems = publish_model.validate(d)
    assert any("no weight file" in p for p in problems)
    assert any("Limitations" in p for p in problems)
    assert any("Evaluation" in p for p in problems)


def test_a_failed_fusion_check_blocks_publication(tmp_path: Path) -> None:
    d = _model_dir(tmp_path)
    (d / "eval_summary.json").write_text(
        json.dumps(
            {
                "metrics": METRICS,
                "fusion": {"passed": False, "regressions": {"json_valid_rate": -0.4}},
            }
        ),
        encoding="utf-8",
    )
    assert any("regressed" in p for p in publish_model.validate(d))


@pytest.mark.parametrize("field", ["license", "base_model", "pipeline_tag"])
def test_frontmatter_must_carry_the_fields_the_hub_renders(
    tmp_path: Path, field: str
) -> None:
    d = _model_dir(tmp_path)
    text = (d / "README.md").read_text()
    (d / "README.md").write_text(
        "\n".join(l for l in text.split("\n") if not l.startswith(f"{field}:")),
        encoding="utf-8",
    )
    assert any(field in p for p in publish_model.validate(d))


def test_going_public_needs_the_licensing_confirmation(tmp_path: Path) -> None:
    d = _model_dir(tmp_path)
    rc = publish_model.main(
        ["--model-dir", str(d), "--repo-id", "org/m", "--public"]
    )
    assert rc == 1  # refused before any network call


def test_dry_run_makes_no_network_call_and_reports_the_manifest(
    tmp_path: Path, capsys
) -> None:
    d = _model_dir(tmp_path)
    assert publish_model.main(["--model-dir", str(d), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "PRIVATE repository" in out
    assert "model.safetensors" in out


# ── the card generator ──────────────────────────────────────────────────────


def test_fusion_check_tolerates_quantization_noise_but_not_a_regression() -> None:
    adapter = dict(METRICS, alert_class_accuracy=0.6346)
    # Fusing into 4-bit re-quantizes, so small movement in both directions is
    # expected and must not fail the build.
    noisy = dict(METRICS)
    assert model_card.fusion_check(noisy, adapter)["passed"]

    broken = dict(METRICS, alert_class_accuracy=0.40)
    check = model_card.fusion_check(broken, adapter)
    assert not check["passed"]
    assert "alert_class_accuracy" in check["regressions"]

    unparseable = dict(METRICS, json_valid_rate=0.5)
    assert not model_card.fusion_check(unparseable, adapter)["passed"]


def test_majority_baseline_is_the_score_of_always_guessing_the_commonest_class() -> (
    None
):
    cls, share = model_card.majority_baseline({"A11": 56, "A10": 17, "A3": 7})
    assert cls == "A11"
    assert share == pytest.approx(56 / 80)


def test_prediction_agreement_counts_documents_not_metrics(tmp_path: Path) -> None:
    def write(path: Path, preds):
        path.write_text(
            "\n".join(
                json.dumps(
                    {"source_id": f"d{i}", "gold_alert_class": "A3",
                     "pred_alert_class": p}
                )
                for i, p in enumerate(preds)
            ),
            encoding="utf-8",
        )

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, ["A3", "A3", "A11", "A10"])
    write(b, ["A3", "A3", "A11", "A11"])  # one document changed
    result = model_card.prediction_agreement(b, a)
    assert result["documents"] == 4
    assert result["same_prediction"] == 3
    assert result["agreement"] == pytest.approx(0.75)
    assert result["changed"] == [
        {"source_id": "d3", "gold": "A3", "adapter": "A10", "fused": "A11"}
    ]


def test_val_loss_curve_is_read_from_the_training_log(tmp_path: Path) -> None:
    log = tmp_path / "train.out"
    log.write_text(
        "Iter 1: Val loss 3.075, Val took 38.6s\n"
        "Iter 10: Train loss 1.644, Peak mem 4.361 GB\n"
        "Iter 300: Val loss 0.961, Val took 39.2s\n"
        "Trainable parameters: 0.182% (7.340M/4022.468M)\n",
        encoding="utf-8",
    )
    assert model_card.val_loss_curve(log) == [(1, 3.075), (300, 0.961)]
    assert model_card.peak_memory_gb(log) == pytest.approx(4.361)
    assert model_card.trainable_params(log) == "7.340M of 4022.468M (0.182%)"


def test_reflow_wraps_prose_and_leaves_code_tables_and_bullets_alone() -> None:
    source = (
        "**Bold lead-in.** " + "word " * 40 + "\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "- a bullet that is quite long " + "word " * 20 + "\n\n"
        "```python\nx = 1  # a deliberately long comment " + "y " * 30 + "\n```\n"
    )
    out = model_card.reflow(source, width=80)
    assert "| a | b |" in out and "|---|---|" in out  # table rows untouched
    assert out.count("- a bullet") == 1
    assert any(l.startswith("- a bullet") and len(l) > 80 for l in out.split("\n"))
    code = out.split("```")[1]
    assert "x = 1  # a deliberately long comment" in code
    prose = [
        l
        for l in out.split("\n")
        if l.startswith(("**Bold", "word"))
    ]
    assert prose and all(len(l) <= 80 for l in prose)
