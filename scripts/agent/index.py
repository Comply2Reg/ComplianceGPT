#!/usr/bin/env python3
"""The agent's library: GREEN documents cut into provisions, in a SQLite FTS5 index.

    python -m scripts.agent.index build [--db data/agent/corpus.sqlite]
    python -m scripts.agent.index search "effective date" --regulator SEC

Only GREEN text enters, through scripts.regnano.sources.iter_docs, the same gate
as RegNano: no AMBER/RED override exists. Search is BM25 over title, section
heading and text; no embeddings, no API calls.

A provision is a run of whole lines up to PROVISION_CHARS. Lines are never split
unless one alone is longer than that, because mined definitions run to the end
of their line and a split would make the task unanswerable. Its id is
`{source_id}#{n}` (n from 0, in document order) and its section is the nearest
heading line at or before its start.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DB = ROOT / "data" / "agent" / "corpus.sqlite"
PROVISION_CHARS = 1500
SNIPPET_TOKENS = 40
MAX_SECTIONS = 60

HEADING_RE = re.compile(
    r"^(?:§+\s*\d[\w.\-]*.{0,80}"
    r"|(?:PART|Part|SUBPART|Subpart|Article|ARTICLE|Regulation|Schedule|SCHEDULE|Chapter|CHAPTER)\s+[\dIVXA-Z]+\b.{0,80}"
    r"|[A-Z][A-Z ,;&/\-]{2,48}:"
    r"|\d{1,3}\.\s+[A-Z][^.]{2,80})$"
)
WORD_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were will with what "
    "when which who how does do did document rule".split()
)


@dataclass(frozen=True)
class Provision:
    provision_id: str
    document_id: str
    seq: int
    section: str
    text: str


def _split_long(line: str, limit: int) -> List[str]:
    """Cut an over-long line at sentence ends, else hard at the limit."""
    out, rest = [], line
    while len(rest) > limit:
        cut = rest.rfind(". ", 0, limit)
        cut = cut + 1 if cut > limit // 3 else limit
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


def provisions(document_id: str, text: str, limit: int = PROVISION_CHARS) -> List[Provision]:
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            lines.extend(_split_long(line, limit) if len(line) > limit else [line])
    out: List[Provision] = []
    buf: List[str] = []
    size = 0
    section = ""
    start_section = ""
    for line in lines:
        is_heading = bool(HEADING_RE.match(line))
        if buf and size + len(line) + 1 > limit:
            out.append(Provision(f"{document_id}#{len(out)}", document_id, len(out), start_section, "\n".join(buf)))
            buf, size = [], 0
        if is_heading:
            section = line[:100]
        if not buf:
            start_section = section
        buf.append(line)
        size += len(line) + 1
    if buf:
        out.append(Provision(f"{document_id}#{len(out)}", document_id, len(out), start_section, "\n".join(buf)))
    return out


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY, regulator TEXT, document_type TEXT, title TEXT,
    release_date TEXT, licence TEXT, url TEXT, doc_hash TEXT, n_provisions INTEGER);
CREATE TABLE IF NOT EXISTS provisions (
    provision_id TEXT UNIQUE, document_id TEXT, seq INTEGER, title TEXT, section TEXT, text TEXT);
CREATE INDEX IF NOT EXISTS provisions_doc ON provisions(document_id, seq);
CREATE VIRTUAL TABLE IF NOT EXISTS provisions_fts USING fts5(
    title, section, text, content='provisions', content_rowid='rowid', tokenize='porter unicode61');
"""


def build(db_path: Path, docs: Iterable) -> Dict[str, int]:
    """Write a fresh index. Refuses to overwrite: delete the file deliberately first."""
    if db_path.exists():
        raise SystemExit(f"{db_path} exists; remove it first to rebuild")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    n_docs = n_prov = 0
    started = time.monotonic()
    for doc in docs:
        provs = provisions(doc.source_id, doc.text)
        con.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
            (doc.source_id, doc.regulator, doc.document_type, doc.title, doc.release_date, doc.licence,
             doc.url, doc.doc_hash, len(provs)),
        )
        for p in provs:
            cur = con.execute(
                "INSERT INTO provisions(provision_id, document_id, seq, title, section, text) VALUES (?,?,?,?,?,?)",
                (p.provision_id, p.document_id, p.seq, doc.title, p.section, p.text),
            )
            con.execute(
                "INSERT INTO provisions_fts(rowid, title, section, text) VALUES (?,?,?,?)",
                (cur.lastrowid, doc.title, p.section, p.text),
            )
        n_docs += 1
        n_prov += len(provs)
        if n_docs % 500 == 0:
            con.commit()
            print(f"{n_docs} documents, {n_prov} provisions, {time.monotonic() - started:.0f}s", file=sys.stderr)
    con.commit()
    con.execute("INSERT INTO provisions_fts(provisions_fts) VALUES ('optimize')")
    con.execute("ANALYZE")
    con.commit()
    con.close()
    return {"documents": n_docs, "provisions": n_prov}


def fts_query(text: str) -> Optional[str]:
    """User words → an FTS5 OR-query of quoted terms; BM25 ranks by how many match."""
    words = [w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS]
    seen: List[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return " OR ".join(f'"{w}"' for w in seen[:16]) or None


class Corpus:
    """Read-only access to the index. Every method returns plain JSON-able data."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found; run: python -m scripts.agent.index build")
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.con.row_factory = sqlite3.Row

    def document(self, document_id: str) -> Optional[Dict]:
        row = self.con.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return dict(row) if row else None

    def documents(self) -> Iterator[Dict]:
        for row in self.con.execute("SELECT * FROM documents ORDER BY document_id"):
            yield dict(row)

    def search(self, query: str, regulator: Optional[str] = None, document_id: Optional[str] = None,
               limit: int = 5) -> List[Dict]:
        q = fts_query(query)
        if not q:
            return []
        sql = (
            "SELECT p.provision_id, p.document_id, d.title, d.regulator, d.release_date, p.section, "
            f"snippet(provisions_fts, 2, '', '', ' … ', {SNIPPET_TOKENS}) AS snippet "
            "FROM provisions_fts JOIN provisions p ON p.rowid = provisions_fts.rowid "
            "JOIN documents d ON d.document_id = p.document_id "
            "WHERE provisions_fts MATCH ?"
        )
        params: List = [q]
        if regulator:
            sql += " AND d.regulator = ?"
            params.append(regulator.strip().upper())
        if document_id:
            sql += " AND p.document_id = ?"
            params.append(document_id)
        sql += " ORDER BY bm25(provisions_fts, 4.0, 2.0, 1.0) LIMIT ?"
        params.append(max(1, min(int(limit), 5)))
        return [dict(r) for r in self.con.execute(sql, params)]

    def provision(self, provision_id: str) -> Optional[Dict]:
        row = self.con.execute(
            "SELECT p.provision_id, p.document_id, d.title, d.regulator, d.release_date, p.section, p.text "
            "FROM provisions p JOIN documents d ON d.document_id = p.document_id WHERE p.provision_id = ?",
            (provision_id,),
        ).fetchone()
        return dict(row) if row else None

    def sections(self, document_id: str) -> Tuple[List[Dict], int]:
        """(first MAX_SECTIONS provisions, total count). Consecutive same-heading rows are kept: each is a provision."""
        total = self.con.execute("SELECT COUNT(*) FROM provisions WHERE document_id = ?", (document_id,)).fetchone()[0]
        rows = self.con.execute(
            "SELECT provision_id, section FROM provisions WHERE document_id = ? ORDER BY seq LIMIT ?",
            (document_id, MAX_SECTIONS),
        )
        return [dict(r) for r in rows], total

    def document_text(self, document_id: str) -> str:
        rows = self.con.execute("SELECT text FROM provisions WHERE document_id = ? ORDER BY seq", (document_id,))
        return "\n".join(r[0] for r in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--db", type=Path, default=DEFAULT_DB)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--regulator")
    s.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        from scripts.regnano.sources import iter_docs

        print(json.dumps(build(args.db, iter_docs())))
    else:
        for r in Corpus(args.db).search(args.query, regulator=args.regulator):
            print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
