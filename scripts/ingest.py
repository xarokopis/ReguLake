import shutil

from gtgh_team3_compliance_assistant.config import (
    PDF_DIR, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME,
)
from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder
from gtgh_team3_compliance_assistant.storing.Storage import ChromaVectorStore
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

shutil.rmtree(CHROMA_PATH, ignore_errors=True)
print("Cleared old ChromaDB\n")

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
