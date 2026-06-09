from fastapi import APIRouter, HTTPException

from gtgh_team3_compliance_assistant.config import (
    PDF_DIR, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME,
)
from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder
from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

router = APIRouter(prefix="/ingestion")


@router.post("/process")
def process_all_pdfs():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise HTTPException(status_code=404, detail=f"No PDFs found in {PDF_DIR}")

    embedding_model = LocalEmbedder(model_name=EMBEDDING_MODEL_NAME)
    vector_store = ChromaVectorStore(
        persist_path=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
    )

    results = []
    for pdf_path in pdf_files:
        pipeline = RAGPipeline(
            pdf_path=str(pdf_path),
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        pipeline.ingest()
        results.append(pdf_path.name)

    return {"processed": results, "count": len(results)}
