from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_no_hierarchical_imports() -> None:
    hits = []
    for path in (ROOT / "scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from hierarchical" in text or "import hierarchical" in text:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []
