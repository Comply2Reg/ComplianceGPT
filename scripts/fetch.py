import re
import json
import hashlib
import pathlib
import time

import requests
from lxml import etree


LEG = "http://www.legislation.gov.uk/namespaces/legislation"

HEADERS = {
    "User-Agent": "Comply2Reg-research/0.1"
}


# Start with four documents.
DOCS = [
    (
        "fsma2000",
        "https://www.legislation.gov.uk/ukpga/2000/8/data.xml"
    ),
    (
        "mlr2017",
        "https://www.legislation.gov.uk/uksi/2017/692/data.xml"
    ),
    (
        "psr2017",
        "https://www.legislation.gov.uk/uksi/2017/752/data.xml"
    ),
    (
        "emr2011",
        "https://www.legislation.gov.uk/uksi/2011/99/data.xml"
    ),
]


def chunk_document(doc_id, xml_bytes):
    root = etree.fromstring(xml_bytes)

    parts = []
    chunks = []
    cursor = 0

    for prov in root.iter(f"{{{LEG}}}P1"):

        text = " ".join(
            t.strip()
            for t in prov.itertext()
            if t.strip()
        )

        text = re.sub(r"\s+", " ", text).strip()

        # Skip headings and very short pieces.
        if len(text) < 40:
            continue

        num = prov.find(
            f".//{{{LEG}}}Pnumber"
        )

        ref = (
            num.text.strip()
            if num is not None and num.text
            else str(len(chunks) + 1)
        )

        start = cursor
        end = cursor + len(text)

        chunks.append({
            "chunk_id": f"{doc_id}#{ref}",
            "doc_id": doc_id,
            "provision": ref,
            "text": text,
            "start": start,
            "end": end,
        })

        parts.append(text)

        # +1 for the newline between chunks.
        cursor = end + 1

    return "\n".join(parts), chunks


if __name__ == "__main__":

    all_chunks = []

    for doc_id, url in DOCS:

        print("fetching", doc_id, flush=True)

        r = requests.get(
            url,
            headers=HEADERS,
            timeout=90
        )

        r.raise_for_status()

        pathlib.Path(
            f"data/raw/{doc_id}.xml"
        ).write_bytes(r.content)

        canonical, chunks = chunk_document(
            doc_id,
            r.content
        )

        pathlib.Path(
            f"data/canonical/{doc_id}.txt"
        ).write_text(
            canonical,
            encoding="utf-8"
        )

        # THE ROUND-TRIP TEST.
        for c in chunks:
            assert (
                canonical[c["start"]:c["end"]]
                == c["text"]
            ), c["chunk_id"]

        h = hashlib.sha256(
            canonical.encode()
        ).hexdigest()[:16]

        for c in chunks:
            c["doc_hash"] = h
            c["source_url"] = url

        all_chunks += chunks

        print(
            f"  {len(chunks)} chunks, offsets verified"
        )

        time.sleep(1)

    with open(
        "data/chunks.jsonl",
        "w",
        encoding="utf-8"
    ) as f:

        for c in all_chunks:
            f.write(
                json.dumps(
                    c,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"\nDONE — {len(all_chunks)} chunks -> "
        f"data/chunks.jsonl"
    )