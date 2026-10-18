"""Agent playground: index, tool harness and episode verifier, on a two-document fake corpus."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent import env  # noqa: E402
from scripts.agent.index import Corpus, build, fts_query, provisions  # noqa: E402
from scripts.agent.tasks import held_out  # noqa: E402
from scripts.agent.verify import verify_episode  # noqa: E402

DEFINITION = "“Covered institution” means an insured depository institution with total assets of $10 billion or more."
RULE_TEXT = (
    "DEPARTMENT OF THE TREASURY\nReporting Requirements for Covered Institutions\nDATES:\n"
    "This rule is effective January 1, 2027.\n§ 1.2 Definitions.\n" + DEFINITION + "\n"
    "§ 1.3 Reporting.\nA covered institution must submit the report within 30 days of the end of each quarter.\n"
)


@dataclass(frozen=True)
class FakeDoc:
    source_id: str
    regulator: str
    title: str
    text: str
    document_type: str = "final_rules"
    release_date: str = "2026-11-01"
    tier: str = "GREEN"
    licence: str = "us_public_domain"
    url: str = ""
    doc_hash: str = "h"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Corpus:
    db = tmp_path_factory.mktemp("agent") / "corpus.sqlite"
    build(db, [
        FakeDoc("us:1", "OCC", "Reporting Requirements for Covered Institutions", RULE_TEXT),
        FakeDoc("us:2", "SEC", "Unrelated Fee Schedule", "Fees are payable annually.\nNo definitions here.\n"),
    ])
    return Corpus(db)


def _task(family: str, **kw) -> dict:
    base = {"task_id": "t", "family": family, "prompt": "q", "source_id": "us:1", "context": RULE_TEXT}
    base.update(kw)
    return base


def _call(name: str, **arguments) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def test_provisions_never_split_a_line_and_track_sections() -> None:
    provs = provisions("d", RULE_TEXT, limit=120)
    assert any(DEFINITION in p.text for p in provs)
    assert [p.provision_id for p in provs] == [f"d#{i}" for i in range(len(provs))]
    assert any(p.section.startswith("§ 1.3") for p in provs)


def test_fts_query_drops_stopwords_and_quotes_terms() -> None:
    assert fts_query('What is the "effective date"?') == '"effective" OR "date"'
    assert fts_query("the of and") is None


def test_search_get_and_sections(corpus: Corpus) -> None:
    hits = corpus.search("covered institution definition", regulator="occ")
    assert hits and hits[0]["document_id"] == "us:1" and hits[0]["snippet"]
    assert corpus.search("covered institution", regulator="SEC") == []
    assert DEFINITION in corpus.provision(hits[0]["provision_id"])["text"]
    rows, total = corpus.sections("us:1")
    assert total == len(rows) >= 1


def test_bad_calls_return_errors_not_exceptions(corpus: Corpus) -> None:
    assert not env.run_tool(corpus, "delete_everything", {}).ok
    assert "invalid arguments" in env.run_tool(corpus, "search_regulation", {"query": "x", "limit": 9}).error
    assert not env.run_tool(corpus, "get_provision", {"provision_id": "us:1#999"}).ok


def _definition_episode(corpus: Corpus, answer: dict) -> env.Episode:
    ep = env.start(_task("lookup_definition", base_family="defined_term",
                         gold={"term": "Covered institution", "definition": DEFINITION}))
    env.step(ep, _call("search_regulation", query="covered institution means", regulator="OCC"), corpus)
    pid = ep.calls[0].returned_ids[0]
    env.step(ep, _call("get_provision", provision_id=pid), corpus)
    env.step(ep, json.dumps({k: (pid if v == "PID" else v) for k, v in answer.items()}), corpus)
    return ep


def test_grounded_cited_definition_passes(corpus: Corpus) -> None:
    ep = _definition_episode(corpus, {"definition": DEFINITION, "provision_id": "PID"})
    assert verify_episode(ep, corpus).passed
    assert ep.messages[2]["tool_calls"][0]["function"]["name"] == "search_regulation"
    assert ep.messages[3]["role"] == "tool"


def test_invented_citation_fails(corpus: Corpus) -> None:
    ep = _definition_episode(corpus, {"definition": DEFINITION, "provision_id": "us:1#77"})
    assert verify_episode(ep, corpus).reason == "provision_id missing or never returned by a tool"


def test_answer_from_memory_fails(corpus: Corpus) -> None:
    ep = env.start(_task("lookup_definition", base_family="defined_term",
                         gold={"term": "Covered institution", "definition": DEFINITION}))
    env.step(ep, json.dumps({"definition": DEFINITION, "provision_id": "us:1#0"}), corpus)
    assert not verify_episode(ep, corpus).passed


def test_budget_is_enforced(corpus: Corpus) -> None:
    ep = env.start(_task("not_found", gold={"status": "not_found"}))
    four = "\n".join(_call("search_regulation", query=f"term {i}") for i in range(4))
    env.step(ep, four, corpus)
    assert len(ep.calls) == env.TOOL_BUDGET and ep.over_budget == 1
    assert "budget" in ep.messages[-1]["content"]
    env.step(ep, '{"status": "not_found"}', corpus)
    assert verify_episode(ep, corpus).reason == "tool budget exceeded"


def test_malformed_call_fails(corpus: Corpus) -> None:
    ep = env.start(_task("not_found", gold={"status": "not_found"}))
    env.step(ep, "<tool_call>\n{not json}\n</tool_call>", corpus)
    env.step(ep, '{"status": "not_found"}', corpus)
    assert verify_episode(ep, corpus).reason == "malformed tool call"


def test_not_found_needs_a_look_and_no_answer(corpus: Corpus) -> None:
    ep = env.start(_task("not_found", gold={"status": "not_found"}))
    env.step(ep, '{"status": "not_found"}', corpus)
    assert verify_episode(ep, corpus).reason == "said not_found without looking"
    ep = env.start(_task("not_found", gold={"status": "not_found"}))
    env.step(ep, _call("search_regulation", query="covered institution", document_id="us:2"), corpus)
    env.step(ep, '{"status": "not_found"}', corpus)
    assert verify_episode(ep, corpus).passed


def test_no_tool_needed_penalises_any_call(corpus: Corpus) -> None:
    gold = {"term": "Covered institution", "definition": DEFINITION}
    ep = env.start(_task("no_tool_needed", base_family="defined_term", gold=gold))
    env.step(ep, json.dumps({"definition": DEFINITION}), corpus)
    assert verify_episode(ep, corpus).passed
    ep = env.start(_task("no_tool_needed", base_family="defined_term", gold=gold))
    env.step(ep, _call("search_regulation", query="covered"), corpus)
    env.step(ep, json.dumps({"definition": DEFINITION}), corpus)
    assert not verify_episode(ep, corpus).passed


def test_effective_date_must_cite_the_right_document(corpus: Corpus) -> None:
    def run(doc_filter: str) -> env.Episode:
        ep = env.start(_task("lookup_effective_date", base_family="effective_date", gold={"date": "2027-01-01"}))
        env.step(ep, _call("search_regulation", query="effective fees", document_id=doc_filter), corpus)
        pid = ep.calls[0].returned_ids[0]
        env.step(ep, json.dumps({"date": "2027-01-01", "provision_id": pid}), corpus)
        return ep

    assert verify_episode(run("us:1"), corpus).passed
    assert verify_episode(run("us:2"), corpus).reason == "cited provision is from another document"


def test_held_out_split_is_by_document_and_stable() -> None:
    assert held_out("abc") == held_out("abc")
    share = sum(held_out(f"doc{i}") for i in range(2000)) / 2000
    assert 0.15 < share < 0.25


def test_quote_spanning_a_line_break_is_grounded(corpus: Corpus) -> None:
    """Tool results reach the model as JSON, where line breaks are escaped; quotes are checked unescaped."""
    span = "§ 1.3 Reporting.\nA covered institution must submit the report"
    ep = env.start(_task("lookup_definition"))
    env.step(ep, _call("search_regulation", query="covered institution report", document_id="us:1"), corpus)
    env.step(ep, _call("get_provision", provision_id=ep.calls[0].returned_ids[0]), corpus)
    assert "\\n" in ep.calls[1].result_text
    from scripts.regnano.verifiers import locate
    assert locate(span, ep.seen_text) is not None


def test_sft_per_turn_cuts_at_each_assistant_turn() -> None:
    from scripts.agent.sft import per_turn

    msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant", "tool_calls": [1]}, {"role": "tool"},
            {"role": "assistant", "content": "{}"}]
    cuts = per_turn(msgs)
    assert [len(c) for c in cuts] == [3, 5]
    assert all(c[-1]["role"] == "assistant" for c in cuts)


def test_compare_pairs_tasks() -> None:
    from scripts.agent.compare import compare

    def row(tid, ok):
        return {"family": "f", "task_id": tid, "passed": ok, "calls": [], "malformed": 0, "over_budget": 0,
                "reason": "ok"}

    res = compare([row("a", False), row("b", True)], [row("a", True), row("b", True), row("c", True)])
    assert res["families"]["f"] == {"tasks": 2, "base": 0.5, "new": 1.0, "delta": 0.5, "ci95": [0.0, 1.0],
                                    "up": 1, "down": 0}
