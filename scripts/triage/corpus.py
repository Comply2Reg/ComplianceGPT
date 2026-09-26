"""Read the Stage-1 corpus c2r-inventory-kit exports.

One place that knows the folder layout (`manifest.jsonl`, `chunks.jsonl`,
`canonical/{id}.txt`) and the tier rule, so every CLI below shares it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

TIERS = ("GREEN", "AMBER", "RED")


@dataclass(frozen=True)
class Corpus:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def coverage_path(self) -> Path:
        return self.root / "coverage.json"

    def canonical(self, doc_id: int) -> str:
        return (self.root / "canonical" / f"{doc_id}.txt").read_text(encoding="utf-8")

    def check(self) -> None:
        for p in (self.manifest_path, self.chunks_path, self.root / "canonical"):
            if not p.exists():
                raise FileNotFoundError(
                    f"not a corpus folder: {self.root} (missing {p.name})"
                )

    def manifest(self) -> List[Dict]:
        """Exported documents only — `duplicate_of` pointer rows are dropped."""
        rows = []
        with self.manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if not row.get("duplicate_of"):
                        rows.append(row)
        return rows

    def chunks_by_doc(
        self, doc_ids: Optional[Iterable[int]] = None
    ) -> Dict[int, List[Dict]]:
        wanted = set(doc_ids) if doc_ids is not None else None
        out: Dict[int, List[Dict]] = defaultdict(list)
        with self.chunks_path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                c = json.loads(line)
                if wanted is None or c["doc_id"] in wanted:
                    out[c["doc_id"]].append(c)
        return out

    def iter_chunks(self) -> Iterator[Dict]:
        with self.chunks_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def filter_rows(
    rows: Iterable[Dict],
    tiers: Iterable[str],
    regulators: Optional[Iterable[str]] = None,
    strata: Optional[Iterable[str]] = None,
) -> List[Dict]:
    allowed = {t.upper() for t in tiers}
    regs = {r.upper() for r in regulators} if regulators else None
    strs = set(strata) if strata else None
    out = []
    for r in rows:
        if str(r.get("tier", "")).upper() not in allowed:
            continue
        if regs and str(r.get("regulator", "")).upper() not in regs:
            continue
        if strs and r.get("stratum") not in strs:
            continue
        out.append(r)
    return out


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            for raw, esc in (("", "\\u0085"), (" ", "\\u2028"), (" ", "\\u2029")):
                line = line.replace(raw, esc)
            fh.write(line + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh.read().split("\n") if line.strip()]


@dataclass(frozen=True)
class MultiCorpus:
    """Several jurisdiction corpora addressed as one.

    Document ids are per-database and collide across jurisdictions — the UK
    and US exports share 5,989 of them — so anything that merges corpora must
    key on the content hash, which is globally unique by construction. Keying
    on the id would silently pair a document with another country's text.
    """

    corpora: Tuple[Corpus, ...]

    @classmethod
    def of(cls, roots: Iterable[Path]) -> "MultiCorpus":
        return cls(tuple(Corpus(Path(r)) for r in roots))

    def check(self) -> None:
        for c in self.corpora:
            c.check()
        seen: Dict[str, Path] = {}
        for c in self.corpora:
            for row in c.manifest():
                h = row.get("sha256") or row.get("doc_hash")
                if not h:
                    continue
                if h in seen and seen[h] != c.root:
                    raise ValueError(
                        f"content hash {h[:12]} appears in both {seen[h].name} "
                        f"and {c.root.name}; corpora must not overlap"
                    )
                seen[h] = c.root

    def _index(self) -> Dict[str, Tuple[Corpus, int]]:
        out: Dict[str, Tuple[Corpus, int]] = {}
        for c in self.corpora:
            for row in c.manifest():
                h = row.get("sha256") or row.get("doc_hash")
                if h:
                    out[h] = (c, int(row["id"]))
        return out

    def manifest(self) -> List[Dict]:
        """Every corpus's manifest, each row tagged with the corpus it came
        from so a later lookup does not have to guess."""
        rows: List[Dict] = []
        for c in self.corpora:
            for row in c.manifest():
                rows.append({**row, "_corpus": str(c.root)})
        return rows

    def canonical_by_hash(self, doc_hash: str) -> str:
        c, doc_id = self._index()[doc_hash]
        return c.canonical(doc_id)

    def chunks_by_hash(self) -> Dict[str, List[Dict]]:
        """Chunks keyed by the document's content hash, not its id."""
        out: Dict[str, List[Dict]] = defaultdict(list)
        for c in self.corpora:
            by_id = c.chunks_by_doc()
            for row in c.manifest():
                h = row.get("sha256") or row.get("doc_hash")
                if h:
                    out[h].extend(by_id.get(int(row["id"]), []))
        return out
