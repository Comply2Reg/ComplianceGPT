"""Rendering, budget trimming and answer parsing — no model download needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import config as cfg  # noqa: E402
from scripts.triage.render import (  # noqa: E402
    TRAIN_SYSTEM_PROMPT,
    assistant_payload,
    clean_prose,
    count_tokens,
    fit_to_budget,
    parse_assistant_json,
    render,
    to_messages,
)


class _Tok:
    """A stand-in tokenizer: one token per whitespace-separated word."""

    chat_template = "yes"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        body = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _record(body_words: int = 50) -> dict:
    return {
        "metadata": {
            "source_id": "hmt:policy_papers#1",
            "regulation_name": "Doc",
            "jurisdiction": "GB",
            "doc_type": "policy_papers",
            "regulator": "HMT",
            "release_date": "2026-01-02",
            "doc_hash": "ab",
        },
        "classification": "A2",
        "instruction": "Triage this.",
        "input_text": "Regulator: HMT\nPublication type: policy_papers\nTitle: Doc\n"
        "Published: 2026-01-02\n\n" + " ".join(f"w{i}" for i in range(body_words)),
        "tool_use": None,
        "thought_trace": "because",
        "output": {
            "alert_class": "A2",
            "alert_class_confidence": "high",
            "priority": "P1",
            "primary_functions": ["Compliance"],
            "secondary_functions": [],
            "lines_of_defence": [2],
            "summary": "s",
            "key_dates": [],
            "applicability": [],
            "obligations_present": True,
        },
    }


def test_registry_has_focus_and_fallback_with_markers() -> None:
    assert cfg.FOCUS_MODEL == "qwen3-4b-instruct"
    for key, spec in cfg.MODELS.items():
        assert spec["hf_id"] and spec["instruction_part"] and spec["response_part"]
        assert spec["max_seq_length"] <= spec["max_context"]
    assert cfg.MODELS["qwen3-4b-instruct"]["response_part"] == "<|im_start|>assistant\n"


def test_messages_have_three_roles_and_compact_ordered_answer() -> None:
    msgs = to_messages(_record())
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == TRAIN_SYSTEM_PROMPT
    assert msgs[1]["content"].startswith(
        "Instruction: Triage this.\nContext: Regulator: HMT"
    )
    assert (
        '"regulator": "HMT"' in msgs[1]["content"]
        and "doc_hash" not in msgs[1]["content"]
    )
    payload = json.loads(msgs[2]["content"])
    assert list(payload)[0] == "alert_class" and list(payload)[-1] == "thought_trace"
    assert "status" not in payload and payload == assistant_payload(_record())


def test_render_uses_the_tokenizer_template_or_chatml_fallback() -> None:
    msgs = to_messages(_record())
    with_tok = render(msgs, _Tok(), "qwen3-4b-instruct")
    without = render(msgs, None, "qwen3-4b-instruct")
    assert with_tok == without
    assert with_tok.endswith("<|im_end|>\n")
    prompt = render(msgs[:2], _Tok(), "qwen3-4b-instruct", add_generation_prompt=True)
    assert prompt.endswith("<|im_start|>assistant\n")


def test_fit_to_budget_trims_context_only() -> None:
    rec = _record(body_words=800)
    before_answer = rec["output"].copy()
    rec, n, trimmed = fit_to_budget(
        rec, _Tok(), "qwen3-4b-instruct", max_seq_length=300
    )
    assert trimmed and n <= 300
    assert rec["input_text"].startswith("Regulator: HMT\nPublication type")
    assert rec["output"] == before_answer and rec["thought_trace"] == "because"
    rec2, n2, trimmed2 = fit_to_budget(
        _record(body_words=20), _Tok(), "qwen3-4b-instruct", max_seq_length=300
    )
    assert not trimmed2 and n2 <= 300


def test_count_tokens_proxy_when_no_tokenizer() -> None:
    assert count_tokens("a" * 400, None) == 100


def test_parse_assistant_json_tolerates_wrapping_text() -> None:
    assert parse_assistant_json('{"alert_class": "A6", "x": 1}')["alert_class"] == "A6"
    assert (
        parse_assistant_json('Sure: {"alert_class":"A3"} done')["alert_class"] == "A3"
    )
    assert parse_assistant_json("no json here") is None


def test_clean_prose_strips_urls_emails_and_blank_runs() -> None:
    text = "See https://www.fca.org.uk/x and mail press@cma.gov.uk\n\n\n\nNext   line"
    cleaned = clean_prose(text)
    assert "http" not in cleaned and "@" not in cleaned
    assert "\n\n\n" not in cleaned and "Next line" in cleaned
