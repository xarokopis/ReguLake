from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gtgh_team3_compliance_assistant.config import PDF_DIR
from gtgh_team3_compliance_assistant.embedding.CloudEmbedder import CloudEmbedder
from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

AZURE_CHROMA_PATH = ROOT_DIR / "chroma_db_azure"
AZURE_COLLECTION_NAME = "compliance_docs_azure"

shutil.rmtree(AZURE_CHROMA_PATH, ignore_errors=True)
print("Cleared old Azure ChromaDB\n")

embedding_model = CloudEmbedder()
vector_store = ChromaVectorStore(persist_path=str(AZURE_CHROMA_PATH), collection_name=AZURE_COLLECTION_NAME)

pdf_files = list(PDF_DIR.glob("*.pdf"))
print(f"Found {len(pdf_files)} PDFs\n")

for pdf in pdf_files:
    print("=" * 50)
    RAGPipeline(
        pdf_path=str(pdf),
        embedding_model=embedding_model,
        vector_store=vector_store,
    ).ingest()

print("\nDone.")
