# Platform contract — regulation inventory spine

**Canonical copy:** `00-apps/c2r_cga_api/docs/platform-contract.md`  
Other repos keep a verbatim copy so clones stay self-contained.

## Coupling

Four independent GitHub repos share **runtime data only** — not code:

1. **MySQL table** `regulation_documents`
2. **S3 bucket** typically `regulens-uploaded-docs` (or `-dev`), env names vary by repo

| Repo | Role |
|------|------|
| `c2r-inventory-kit` | **Writes** rows + uploads PDFs |
| `c2r_cga_api` (ReguLens) | **Reads** / indexes into Milvus + Neo4j; may upsert app columns |
| `ComplianceGPT` | **Reads** for parse/extract → JSONL / chunks |
| `c2r_cga_ui` | Displays via API only |

## Uniqueness

`regulation_documents` unique key: `(regulator, url, document_type)`  
(CGA comments sometimes note `url` length limits on MySQL.)

Core fields used across writers/readers: `regulator`, `countries` / jurisdiction, `document_type`, `title`, `url`, `pdf_url`, `file_path` (local path or `s3://…`), `status` (`PENDING` / `DOWNLOADED` / `FAILED` / …).

## Body text (added 2026-09-20)

The crawler now captures each alert's body text, not just its metadata. Exactly
**one S3 object per document**:

| Source page | S3 object | `description` | `content_source` |
|---|---|---|---|
| has an attachment | the PDF/DOCX | page text, first 4,000 chars | `pdf` |
| has none | a **`.txt`** of the extracted text | the same text | `page` |
| GOV.UK Content API | `.txt` of the API body | the same text | `api` |

Two objects for one document would register as two rows and raise two alerts,
because ReguLens extracts a PDF's own text at index time.

`.txt` is deliberate — the CGA ingestion gateway parses `.pdf`, `.html`, `.htm`
and `.txt`, and rejects `.json`, `.xml` and extensionless keys with
`UNSUPPORTED_FILE_TYPE`.

New **crawler-owned** columns: `content_sha256`, `content_chars`,
`content_source`, and `text_path` — a path to the crawler's local full-text
cache, relative to its own data directory. `text_path` is **crawler-local and
nullable**; consumers must ignore it (the text they need is in S3 and, for the
first 4,000 characters, in `description`).

**This required one CGA change.** `list_s3_pdf_keys()` discarded every key that
was not `.pdf`, so a text object was never discovered, indexed or alerted on. It
is now `list_s3_ingestable_keys()` (`apps/regu_lens/s3_corpus.py`) and accepts
every format the ingestion gateway can parse. The old name remains as a
deprecated alias.

## S3 key convention (crawler)

Relative key under the bucket (country from config):

`{Country}/{REGULATOR}/{doc_type}/[{subtype}/]{sanitized_title}.{ext}`

Stored as `s3://<bucket>/{key}` in `file_path`. `{ext}` is `.pdf` when the page
carried an attachment and `.txt` when the page itself is the document.

The `{doc_type}` segment is now **always** present. It used to be dropped for
any regulator whose `config.yaml` lacked a matching `document_types` entry,
which mattered: the CGA poller derives `document_type` from the key and
`instrument_tier` from that, and the tier decides whether an alert is raised.

Env aliases seen in the wild:

- Crawler: `S3_BUCKET_NAME`, `AWS_*`
- CGA / SLM: `AWS_S3_BUCKET_REGU_LENS`, `S3_BUCKET`

Point local `.env` files at the **same** bucket + DB; do not symlink one `.env` across repos.

## Schema drift (do not “fix” casually)

**CGA-owned columns** (app side): e.g. `organization_id`, `indexing_status`, `instrument_tier`, lifecycle/governance fields.

**Crawler-owned extras** (UK / inventory): e.g. `content_category`, `priority`, `document_type_label`, `verticals`, `external_id`, `is_alert`, and — once the source registry v2 cutover lands — `alert_class` (A1–A14) and `tier` (GREEN/AMBER/RED licence tier).

Writers must not overwrite foreign columns blindly; readers should tolerate missing optional columns.

## “Alert” means two different things — this is deliberate, do not reconcile in code

| | an alert is | so an FCA Policy Statement is | and a speech is |
|---|---|---|---|
| crawler `is_alert`, from `ALERT_CATEGORIES` (`core/taxonomy.py`) | `SUPERVISION`, `ENFORCEMENT`, `INTELLIGENCE` | **not** an alert | an alert |
| CGA `ALERTABLE_TIERS` (`apps/regu_lens/instrument_tier.py`) | `binding_rule`, `supervisory_guidance`, `consultation` | an alert | **not** an alert |

The two sets have **no overlap**. A row the crawler flags `is_alert=True` never becomes a RegPulse alert, and a RegPulse alert is never flagged by the crawler.

Neither side should “fix” the other's definition unilaterally: widening the crawler's set rewrites `is_alert` on every existing row, and widening `ALERTABLE_TIERS` changes what RegPulse surfaces to every tenant. The agreed target set and the migration are in `c2r-inventory-kit/docs/alert-taxonomy.md`.

`instrument_tier` stays **CGA-owned and CGA-derived**. The crawler's registry carries an advisory `instrument_tier_hint`; where the two disagree, that is a finding to investigate, not an error to auto-correct.

## Local club path (this machine)

```
~/Documents/17-Comply2Reg/
  00-apps/c2r_cga_api
  00-apps/c2r_cga_ui
  09-inventory-kit/c2r-inventory-kit
  15-SLM/ComplianceGPT
```
