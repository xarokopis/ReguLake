from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder
from gtgh_team3_compliance_assistant.model_communication.llm import ChatLLM
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline
from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore
from gtgh_team3_compliance_assistant.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

router = APIRouter()

embedding_model = LocalEmbedder(model_name=EMBEDDING_MODEL_NAME)
vector_store = ChromaVectorStore(
    persist_path=str(CHROMA_PATH),
    collection_name=COLLECTION_NAME,
)
llm = ChatLLM()
rag = RAGPipeline(
    pdf_path=None,
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm=llm,
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list


@router.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = rag.ask(request.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            retrieved_chunks=result["retrieved_chunks"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
