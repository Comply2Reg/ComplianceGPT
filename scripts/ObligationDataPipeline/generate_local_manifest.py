"""Generate a manifest.json file from the local PDFs in data/input_pdfs for testing without DB/S3 ingestion.

Usage:
  python generate_local_manifest.py
"""

import json
from pathlib import Path

def main():
    input_dir = Path("data/input_pdfs")
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} not found.")
        return

    documents = []
    for pdf_path in input_dir.glob("*.pdf"):
        # Create a simplified metadata entry for the local PDF
        doc_entry = {
            "ingested": True,
            "local_path": str(pdf_path),
            "title": pdf_path.stem.replace("_", " "),
            "regulator": "TEST_REGULATOR",
            "document_type": "Act",
            "jurisdiction": "IN"
        }
        documents.append(doc_entry)

    if not documents:
        print(f"No PDFs found in {input_dir}.")
        return

    manifest = {"documents": documents}
    
    out_dir = Path("data/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = out_dir / "local_test_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated {manifest_path} with {len(documents)} documents.")

if __name__ == "__main__":
    main()
