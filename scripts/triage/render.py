"""Model-neutral messages, model-specific rendering and the token budget.

Records store `messages` (system / user / assistant). The chat template is
never hand-built: `render()` calls the target tokenizer's
`apply_chat_template`, so Qwen's ChatML and Gemma 4's `<|turn>` markers both
come from the model's own files. The budget is measured with the target
tokenizer and the *context* is trimmed to fit — never the answer.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from scripts import config as cfg
from scripts.triage.schema import OUTPUT_FIELDS

log = logging.getLogger("triage.render")

TRAIN_SYSTEM_PROMPT = (
    "You triage financial-services regulatory publications for an "
    "internationally active bank. Return a JSON triage record with exactly "
    "these keys: alert_class (A1-A14), alert_class_confidence, priority "
    "(P1-P3), primary_functions, secondary_functions, lines_of_defence, "
    "summary, key_dates, applicability, obligations_present, jurisdiction, "
    "frameworks, thought_trace."
)
PROMPT_META_KEYS = ("regulator", "doc_type", "regulation_name", "release_date")
CONTEXT_HEADER_LINES = 4  # regulator / type / title / date lines in input_text
MIN_CONTEXT_CHARS = 400
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")


def clean_prose(text: str) -> str:
    """The book's TextCleaner steps that are safe for prose: no URLs or
    e-mail addresses, no runs of blank lines or spaces. Canonical text is
    never touched — this is for the training input only."""
    text = _URL_RE.sub("", text)
    text = _EMAIL_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prompt_metadata(rec: Dict) -> Dict:
    """Only what the model needs to see; provenance stays in `metadata`."""
    meta = rec.get("metadata") or {}
    return {k: meta.get(k) for k in PROMPT_META_KEYS if meta.get(k) is not None}


def assistant_payload(rec: Dict) -> Dict:
    """Compact JSON in schema key order, thought_trace last (short)."""
    out = rec.get("output") or {}
    payload = {k: out[k] for k in OUTPUT_FIELDS if k in out}
    payload["thought_trace"] = rec.get("thought_trace") or ""
    return payload


def to_messages(rec: Dict) -> List[Dict[str, str]]:
    user = (
        f"Instruction: {rec['instruction']}\n"
        f"Context: {rec['input_text']}\n"
        f"Metadata: {json.dumps(prompt_metadata(rec), ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {
            "role": "assistant",
            "content": json.dumps(
                assistant_payload(rec), ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]


def load_tokenizer(model_key: str):
    """The target model's tokenizer, or None with a logged reason."""
    spec = cfg.MODELS[model_key]
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(spec["hf_id"])
    except Exception as exc:  # noqa: BLE001 - gated / offline / not installed
        log.warning(
            "tokenizer for %s unavailable (%s): %s",
            model_key,
            spec["hf_id"],
            str(exc).splitlines()[0][:120],
        )
        return None


def _fallback_chatml(
    messages: List[Dict[str, str]], add_generation_prompt: bool
) -> str:
    text = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def render(
    messages: List[Dict[str, str]],
    tokenizer,
    model_key: str,
    add_generation_prompt: bool = False,
) -> str:
    spec = cfg.MODELS[model_key]
    if tokenizer is None or not getattr(tokenizer, "chat_template", None):
        return _fallback_chatml(messages, add_generation_prompt)
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if spec.get("thinking_switch"):
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text) // 4
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def split_context(input_text: str) -> Tuple[str, str]:
    lines = input_text.split("\n")
    header = "\n".join(lines[:CONTEXT_HEADER_LINES])
    body = "\n".join(lines[CONTEXT_HEADER_LINES:]).strip()
    return header, body


def fit_to_budget(
    rec: Dict, tokenizer, model_key: str, max_seq_length: int
) -> Tuple[Dict, int, bool]:
    """Trim the document context until the rendered record fits.

    Returns (record, n_tokens, trimmed). The header lines and the answer are
    never cut; when the context is already at MIN_CONTEXT_CHARS the record is
    returned over budget so the validator can reject it.
    """
    trimmed = False
    header, body = split_context(rec["input_text"])
    while True:
        rec["input_text"] = f"{header}\n\n{body}" if body else header
        n = count_tokens(render(to_messages(rec), tokenizer, model_key), tokenizer)
        if n <= max_seq_length or len(body) <= MIN_CONTEXT_CHARS:
            return rec, n, trimmed
        cut = int(len(body) * 0.9)
        space = body.rfind(" ", 0, cut)
        body = body[: space if space > MIN_CONTEXT_CHARS else cut].rstrip()
        trimmed = True


def parse_assistant_json(text: str) -> Optional[Dict]:
    """Recover the JSON object from a model turn (any key order, stray text)."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
