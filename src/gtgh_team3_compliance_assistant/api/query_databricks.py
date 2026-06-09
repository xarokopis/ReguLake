from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder
from gtgh_team3_compliance_assistant.model_communication.llm import ChatLLM
from gtgh_team3_compliance_assistant.storing.DatabricksDelta import (
    DatabricksDeltaVectorStore,
)
load_dotenv()
router = APIRouter(prefix="/databricks", tags=["databricks"])

DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG")
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA")
DATABRICKS_EMBEDDED_TABLE = os.getenv("DATABRICKS_EMBEDDED_TABLE")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")


class DatabricksQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class DatabricksQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]


def _get_chunk_text(chunk: dict[str, Any]) -> str:
    return (
        chunk.get("chunk_text")
        or chunk.get("text")
        or chunk.get("content")
        or ""
    )


@router.post("/query", response_model=DatabricksQueryResponse)
def query_databricks(request: DatabricksQueryRequest) -> DatabricksQueryResponse:
    try:
        embedder = LocalEmbedder(model_name=EMBEDDING_MODEL_NAME)

        vector_store = DatabricksDeltaVectorStore(
            catalog=DATABRICKS_CATALOG,
            db_schema=DATABRICKS_SCHEMA,
            table_name=DATABRICKS_EMBEDDED_TABLE,
        )

        query_embedding = embedder.embed_query(request.question)

        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
        )

        if not search_results:
            return DatabricksQueryResponse(
                question=request.question,
                answer="I could not find relevant regulatory context in the Databricks table.",
                sources=[],
                retrieved_chunks=[],
            )

        context_blocks: list[str] = []
        sources: list[dict[str, Any]] = []
        retrieved_chunks: list[dict[str, Any]] = []

        for idx, chunk in enumerate(search_results, start=1):
            chunk_text = _get_chunk_text(chunk)

            source_file = chunk.get("source_file") or chunk.get("file_name")
            page = chunk.get("page") or chunk.get("page_number")
            article = chunk.get("article_number")
            distance = chunk.get("distance")
            score = chunk.get("score")

            context_blocks.append(
                f"[Source {idx}]\n"
                f"File: {source_file}\n"
                f"Page: {page}\n"
                f"Article: {article}\n"
                f"Title: {chunk.get('title')}\n"
                f"Regulation: {chunk.get('regulation_family')}\n"
                f"Text:\n{chunk_text}"
            )

            source = {
                "source_index": idx,
                "chunk_uid": chunk.get("chunk_uid"),
                "source_file": source_file,
                "page": page,
                "article_number": article,
                "score": score,
                "distance": distance,
                "title": chunk.get("title"),
                "regulation_family": chunk.get("regulation_family"),
                "source_path": chunk.get("source_path"),
                "text": chunk_text[:1200],
            }

            sources.append(source)

            retrieved_chunks.append(
                {
                    **source,
                    "text": chunk_text,
                    "metadata": {
                        "document_id": chunk.get("document_id"),
                        "file_name": chunk.get("file_name"),
                        "chunk_id": chunk.get("chunk_id"),
                        "page_number": chunk.get("page_number"),
                        "char_length": chunk.get("char_length"),
                        "source_name": chunk.get("source_name"),
                        "source_path": chunk.get("source_path"),
                    },
                }
            )

        context = "\n\n".join(context_blocks)

        llm = ChatLLM()
        answer = llm.generate(
            question=request.question,
            context=context,
        )

        return DatabricksQueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc