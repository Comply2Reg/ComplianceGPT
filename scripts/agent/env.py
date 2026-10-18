"""One agent episode: the prompt, the tool calls the model makes, and what they return.

The model speaks Qwen3's native tool format (`<tool_call>{"name", "arguments"}
</tool_call>`), which `mlx_lm.server` turns into OpenAI `tool_calls`. Messages
are stored in OpenAI shape so the same trajectory renders through the tokenizer's
own chat template in training and in serving.

Rules, enforced here and nowhere else:
- at most TOOL_BUDGET calls per episode; later calls are refused with a tool
  message telling the model to answer, and count as a budget violation;
- arguments are checked against tools.json; a bad call returns an error as the
  tool result and the episode continues (the verifier fails it);
- a tool result is cut at RESULT_CHARS, and the cut is marked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import jsonschema

from scripts.agent.index import Corpus

TOOLS: List[Dict] = json.loads((Path(__file__).with_name("tools.json")).read_text(encoding="utf-8"))
TOOL_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
TOOL_BUDGET = 3
MAX_TURNS = TOOL_BUDGET + 2
RESULT_CHARS = 6000

SYSTEM_PROMPT = (
    "You are a regulatory compliance analyst with read-only tools over a library of regulations. "
    "Use the tools to find the text you need and never answer from memory. You may make at most "
    f"{TOOL_BUDGET} tool calls in total, so search precisely. If the question already gives you the "
    "passage, answer from it without calling tools. Quote text exactly as written in the tool results "
    "or the passage, and cite the provision_id of every provision you quote. When you have the answer, "
    "reply with the requested JSON object only. If the library does not contain what was asked for, "
    'reply {"status": "not_found"}.'
)

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
OPEN_CALL_RE = re.compile(r"<tool_call>(?!.*</tool_call>)", re.DOTALL)


@dataclass
class Call:
    name: str
    arguments: Dict
    ok: bool
    error: Optional[str] = None
    returned_ids: List[str] = field(default_factory=list)
    result_text: str = ""


@dataclass
class Episode:
    task: Dict
    messages: List[Dict]
    calls: List[Call] = field(default_factory=list)
    final: Optional[str] = None
    turns: int = 0
    over_budget: int = 0
    malformed: int = 0

    @property
    def done(self) -> bool:
        return self.final is not None or self.turns >= MAX_TURNS

    @property
    def returned_ids(self) -> Set[str]:
        return {pid for c in self.calls for pid in c.returned_ids}

    @property
    def seen_text(self) -> str:
        """Everything the tools showed the model, as it saw it."""
        return "\n".join(c.result_text for c in self.calls if c.ok)


def start(task: Dict) -> Episode:
    return Episode(task=task, messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": task["prompt"]}])


def parse_calls(text: str) -> Tuple[str, List[Optional[Dict]]]:
    """(content before the first call, parsed calls). An unparseable call is None."""
    blocks = CALL_RE.findall(text)
    if not blocks and OPEN_CALL_RE.search(text):
        blocks = [text.split("<tool_call>", 1)[1]]  # truncated: still a (malformed) call
    content = text.split("<tool_call>", 1)[0].strip()
    calls: List[Optional[Dict]] = []
    for b in blocks:
        try:
            obj = json.loads(b)
        except json.JSONDecodeError:
            calls.append(None)
            continue
        ok = isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments", {}), dict)
        calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})} if ok else None)
    return content, calls


def _cut(text: str) -> str:
    return text if len(text) <= RESULT_CHARS else text[:RESULT_CHARS] + "\n[… result cut at 6000 characters]"


def run_tool(corpus: Corpus, name: str, args: Dict) -> Call:
    schema = TOOL_SCHEMAS.get(name)
    if schema is None:
        return Call(name, args, False, f"unknown tool {name!r}; tools are {sorted(TOOL_SCHEMAS)}")
    try:
        jsonschema.validate(args, schema)
    except jsonschema.ValidationError as exc:
        return Call(name, args, False, f"invalid arguments: {exc.message}")
    if name == "search_regulation":
        if args.get("document_id") and corpus.document(args["document_id"]) is None:
            return Call(name, args, False, f"no document with id {args['document_id']!r}; document_ids come from "
                                           "search results, so search without document_id first")
        rows = corpus.search(args["query"], regulator=args.get("regulator"), document_id=args.get("document_id"),
                             limit=args.get("limit", 5))
        result = {"results": rows} if rows else {"results": [], "note": "no matching provisions"}
        ids = [r["provision_id"] for r in rows]
    elif name == "get_provision":
        row = corpus.provision(args["provision_id"])
        if row is None:
            return Call(name, args, False, f"no provision with id {args['provision_id']!r}")
        result, ids = row, [row["provision_id"]]
    else:
        if corpus.document(args["document_id"]) is None:
            return Call(name, args, False, f"no document with id {args['document_id']!r}")
        rows, total = corpus.sections(args["document_id"])
        result = {"document_id": args["document_id"], "provisions": rows, "total": total}
        if total > len(rows):
            result["note"] = f"first {len(rows)} of {total}; use search_regulation with document_id for the rest"
        ids = [r["provision_id"] for r in rows]
    text = _cut(json.dumps(result, ensure_ascii=False))
    return Call(name, args, True, None, ids, text)


def step(ep: Episode, completion: str, corpus: Corpus) -> None:
    """Apply one model turn: either run its tool calls or record its final answer."""
    ep.turns += 1
    content, calls = parse_calls(completion)
    if not calls:
        ep.messages.append({"role": "assistant", "content": completion.strip()})
        ep.final = completion
        return
    tool_calls, results = [], []
    for c in calls:
        if c is None:
            ep.malformed += 1
            results.append("error: the tool call was not valid JSON with a name and arguments")
            continue
        tool_calls.append({"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}})
        if len(ep.calls) >= TOOL_BUDGET:
            ep.over_budget += 1
            results.append(f"error: tool budget of {TOOL_BUDGET} calls is used up. Answer now with the JSON object.")
            continue
        call = run_tool(corpus, c["name"], c["arguments"])
        ep.calls.append(call)
        results.append(call.result_text if call.ok else f"error: {call.error}")
    ep.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
    for r in results:
        ep.messages.append({"role": "tool", "content": r})


def transcript(ep: Episode) -> Dict:
    """JSON-able record of an episode, for logs and training data."""
    return {
        "task_id": ep.task["task_id"],
        "family": ep.task["family"],
        "messages": ep.messages,
        "calls": [{"name": c.name, "arguments": c.arguments, "ok": c.ok, "error": c.error,
                   "returned_ids": c.returned_ids} for c in ep.calls],
        "final": ep.final,
        "turns": ep.turns,
        "over_budget": ep.over_budget,
        "malformed": ep.malformed,
    }
