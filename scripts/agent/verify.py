"""Episode verifier: binary, deterministic, code only — the RegNano rule.

An episode passes when the answer is right *and* it was reached properly:

- every tool call was valid JSON, named a real tool, had valid arguments and
  hit something that exists (a guessed provision_id fails);
- no call was attempted past the budget;
- the final message is the JSON object asked for, with no tool call;
- every quote appears in text a tool actually returned in this episode, and
  every cited provision_id was returned by a tool (no invented citations);
- the answer itself passes the RegNano verifier for its family.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from scripts.agent.env import Episode
from scripts.agent.index import Corpus
from scripts.regnano.verifiers import PASS, Verdict, _fail, answer_json, locate, verify as regnano_verify


def _regnano(task: Dict, final: str) -> Verdict:
    inner = {"family": task["base_family"], "context": task["context"], "gold": task["gold"]}
    return regnano_verify(inner, final)


def _cited(ep: Episode, corpus: Corpus, pid) -> Optional[Dict]:
    if not isinstance(pid, str) or pid not in ep.returned_ids:
        return None
    return corpus.provision(pid)


def _grounded(span: str, ep: Episode) -> bool:
    return locate(span, ep.seen_text) is not None


def check_definition(task: Dict, ep: Episode, ans: Dict, corpus: Corpus) -> Verdict:
    v = _regnano(task, ep.final)
    if not v.passed:
        return v
    if not _grounded(ans["definition"], ep):
        return _fail("quote not in any tool result")
    prov = _cited(ep, corpus, ans.get("provision_id"))
    if prov is None:
        return _fail("provision_id missing or never returned by a tool")
    if locate(ans["definition"], prov["text"]) is None:
        return _fail("quote not in the cited provision")
    return PASS


def check_date(task: Dict, ep: Episode, ans: Dict, corpus: Corpus) -> Verdict:
    v = _regnano(task, ep.final)
    if not v.passed:
        return v
    prov = _cited(ep, corpus, ans.get("provision_id"))
    if prov is None:
        return _fail("provision_id missing or never returned by a tool")
    if prov["document_id"] != task["source_id"]:
        # The Federal Register sometimes carries one notice twice under two ids;
        # the same title on the same day is the same document.
        gold = corpus.document(task["source_id"])
        if (prov["title"], prov["release_date"]) != (gold["title"], gold["release_date"]):
            return _fail("cited provision is from another document")
    return PASS


def check_obligations(task: Dict, ep: Episode, ans: Dict, corpus: Corpus) -> Verdict:
    v = _regnano(task, ep.final)
    if not v.passed:
        return v
    spans = [s for s in ans.get("spans", []) if isinstance(s, str)]
    if not all(_grounded(s, ep) for s in spans):
        return _fail("span not in any tool result")
    pids = ans.get("provision_ids")
    if not isinstance(pids, list) or not pids:
        return _fail("no provision_ids")
    provs = [_cited(ep, corpus, p) for p in pids]
    if any(p is None for p in provs):
        return _fail("provision_id never returned by a tool")
    cited_text = "\n".join(p["text"] for p in provs)
    if not all(locate(s, cited_text) is not None for s in spans):
        return _fail("span not in the cited provisions")
    return PASS


def check_no_tool(task: Dict, ep: Episode, ans: Dict, corpus: Corpus) -> Verdict:
    if ep.calls or ep.over_budget or ep.malformed:
        return _fail("called a tool when the passage was given")
    return _regnano(task, ep.final)


def check_not_found(task: Dict, ep: Episode, ans: Dict, corpus: Corpus) -> Verdict:
    if not ep.calls:
        return _fail("said not_found without looking")
    if ans.get("status") != "not_found" or ans.get("definition"):
        return _fail("answered where the library has no answer")
    return PASS


CHECKS: Dict[str, Callable[[Dict, Episode, Dict, Corpus], Verdict]] = {
    "lookup_definition": check_definition,
    "lookup_effective_date": check_date,
    "cited_obligations": check_obligations,
    "no_tool_needed": check_no_tool,
    "not_found": check_not_found,
}


def verify_episode(ep: Episode, corpus: Corpus) -> Verdict:
    task = ep.task
    check = CHECKS.get(task.get("family"))
    if check is None:
        return _fail(f"unknown family {task.get('family')!r}")
    if ep.malformed:
        return _fail("malformed tool call")
    if ep.over_budget:
        return _fail("tool budget exceeded")
    bad = [c for c in ep.calls if not c.ok]
    if bad:
        return _fail(f"invalid tool call: {bad[0].error[:60]}")
    if ep.final is None:
        return _fail("no final answer")
    ans = answer_json(ep.final)
    if ans is None:
        return _fail("final answer is not JSON")
    if task["family"] != "not_found" and ans.get("status") == "not_found":
        return _fail("said not_found for an answerable question")
    try:
        return check(task, ep, ans, corpus)
    except (KeyError, TypeError) as exc:
        return _fail(f"invalid answer or task: {exc}")


def summary(rows: List[Dict]) -> Dict:
    """Per-family pass rate and tool behaviour from rollout rows."""
    out: Dict[str, Dict] = {}
    for r in rows:
        f = out.setdefault(r["family"], {"episodes": 0, "passed": 0, "calls": 0, "invalid_calls": 0,
                                         "over_budget": 0, "seconds": 0.0, "reasons": {}})
        f["episodes"] += 1
        f["passed"] += int(r["passed"])
        f["calls"] += len(r["calls"])
        f["invalid_calls"] += sum(not c["ok"] for c in r["calls"]) + r["malformed"]
        f["over_budget"] += int(r["over_budget"] > 0)
        f["seconds"] += r.get("seconds", 0.0)
        if not r["passed"]:
            f["reasons"][r["reason"]] = f["reasons"].get(r["reason"], 0) + 1
    for f in out.values():
        n = f["episodes"]
        f["pass_rate"] = round(f["passed"] / n, 3)
        f["calls_per_episode"] = round(f["calls"] / n, 2)
        f["seconds_per_episode"] = round(f["seconds"] / n, 1)
        del f["calls"], f["seconds"]
    return out
