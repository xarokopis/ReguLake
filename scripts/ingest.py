from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def add_windows_dll_directories() -> None:
    if os.name != "nt":
        return

    candidate_dirs = [
        ROOT_DIR / ".venv" / "Lib" / "site-packages" / "sklearn" / ".libs",
        ROOT_DIR / ".venv" / "Lib" / "site-packages" / "scipy" / ".libs",
        ROOT_DIR / ".venv" / "Lib" / "site-packages" / "numpy.libs",
    ]

    for dll_dir in candidate_dirs:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))


add_windows_dll_directories()

import shutil
from gtgh_team3_compliance_assistant.config import (
    PDF_DIR, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, METADATA_FILE,
)
from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder
from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

shutil.rmtree(CHROMA_PATH, ignore_errors=True)
print("Cleared old ChromaDB\n")

METADATA_FILE.write_text("[]")
print("Cleared documents.json\n")

embedding_model = LocalEmbedder(model_name=EMBEDDING_MODEL_NAME)
vector_store = ChromaVectorStore(persist_path=str(CHROMA_PATH), collection_name=COLLECTION_NAME)

pdf_files = list(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDFs\n")

for pdf in pdf_files:
    print("=" * 50)
    RAGPipeline(
        pdf_path=str(pdf),
        embedding_model=embedding_model,
        vector_store=vector_store,
    ).ingest()

print("\nDone. Check data/chunks/ for JSON files.")
