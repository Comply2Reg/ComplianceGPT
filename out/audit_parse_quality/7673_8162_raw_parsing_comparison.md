# Raw Docling Stage A comparison — 4424 / 7673 / 8162

Evidence-gathering only. No parser changes. 4424 artifacts were read-only.

## Snapshot

| Doc | Pages | Texts | Tables | PASS | MINOR | MAJOR | UNKNOWN | Recommendation |
|-----|------:|------:|-------:|-----:|------:|------:|--------:|----------------|
| 4424 | 20 | 166 | 2 | 3 | 17 | 0 | 0 | APPROVE RAW PARSING |
| 7673 | 33 | 301 | 4 | 3 | 30 | 0 | 0 | APPROVE RAW PARSING |
| 8162 | 31 | 241 | 3 | 3 | 28 | 0 | 0 | APPROVE RAW PARSING |

## Label mix (texts)

| Label | 4424 | 7673 | 8162 |
|-------|------:|------:|------:|
| `list_item` | 18 | 50 | 10 |
| `page_footer` | 19 | 33 | 32 |
| `page_header` | 20 | 33 | 31 |
| `section_header` | 33 | 55 | 44 |
| `text` | 76 | 130 | 124 |

## Answers

### 1. Does the same Docling extraction approach work across all three?
Yes. Same `DocumentConverter` + table-structure settings produced usable raw JSON for all three newsletters with **zero MAJOR** pages in automated audit.

### 2. Recurring structural problems?
- Recurring **normalization** (not structural) issues: double-spaces / occasional word-glue; page chrome as `page_header`/`page_footer`; Events/Consultations dates as separate text blocks rather than paired table rows.
- Suspicious reading-order vs bbox page counts: 4424=0, 7673=0, 8162=1.
- Structural-flag page counts: 4424=0, 7673=2, 8162=3.

### 3. Document-specific layout problems?
- **4424** (20 pages, tables=2): roles seen — Next steps / Background subsection(s) (7), contains lists (4), TOC / index (2), contains table object(s) (2), contact page (2), promotional / social-media (2).
- **7673** (33 pages, tables=4): roles seen — contains lists (9), Next steps / Background subsection(s) (9), Consultations (7), contains table object(s) (3), TOC / index (2), contact page (2).
- **8162** (31 pages, tables=3): roles seen — Next steps / Background subsection(s) (11), Consultations (4), contains lists (3), contains table object(s) (3), contact page (3), TOC / index (2).

### 4. Important content types consistently lost?
No automated evidence of consistent content loss. Minor token gaps vs pypdf appear on some list/promo pages and are expected layout variance, not wholesale omission.

### 5. Tables / events / consultations representation?
- **4424**: tables=2 ({'document_index': 2}); pages with separate date blocks=1.
- **7673**: tables=4 ({'document_index': 2, 'table': 2}); pages with separate date blocks=0.
- **8162**: tables=3 ({'document_index': 2, 'table': 1}); pages with separate date blocks=1.
TOC-style `document_index` tables appear when present. Events/Consultations often remain free-text / separate date blocks — a **downstream pairing** task, not a raw-parse reject.

### 6. Reading order reliable?
Generally yes (mostly top-down). Pages with multi-column/promo/calendar layouts show more bbox inversions and should be handled carefully in classification, but automated audit did not find systemic misordering requiring parser rewrite.

### 7. Headings and lists consistently detected?
- **4424**: heading pages=19, list-preserved pages=4, `section_header` count=33, `list_item` count=18.
- **7673**: heading pages=31, list-preserved pages=8, `section_header` count=55, `list_item` count=50.
- **8162**: heading pages=30, list-preserved pages=2, `section_header` count=44, `list_item` count=10.

### 8. Any MAJOR issues requiring parser changes?
**No.** Automated audit found no MAJOR pages on 4424, 7673, or 8162. Do not change the parser based on these findings.

### 9. Approve Stage A and move to classification/canonicalization?
**Yes — APPROVE Stage A raw parsing for all three documents.** Proceed to classification/canonicalization design; keep furniture removal, Events/Consultations date pairing, and whitespace normalization in later stages.

**Overall:** APPROVE Stage A for all three

## Distinction reminder

| Class | Examples | Action |
|-------|----------|--------|
| Structural extraction | empty page, glued dates, severe misorder, corruption | would block Stage A / consider parser |
| Normalization later | double spaces, date/title pairing, furniture labels, promo duplicates | handle in classification/canonicalization |
