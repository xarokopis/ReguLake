import json

from uuid import uuid4

from datetime import datetime, timezone

from azure.core.paging import ItemPaged

from gtgh_team3_compliance_assistant.config import RUN_MODE
from gtgh_team3_compliance_assistant.models.chunks import AddChunksInput
from gtgh_team3_compliance_assistant.storing.storageFactory import StorageFactory
from gtgh_team3_compliance_assistant.embedding.EmbedderFactory import EmbedderFactory
from gtgh_team3_compliance_assistant.models.chunks import ChunkInput
from gtgh_team3_compliance_assistant.logger.Logger import log
from gtgh_team3_compliance_assistant.models.search import SearchInput

def run_storage(source: str, recreate_index: bool, limit_chunks: bool):
    log.info("Storage run started", source=source, recreate_index=recreate_index, limit_chunks=limit_chunks)

    with open(source, 'r') as file:
        raw_data = json.load(file)
    raw_chunks = raw_data["chunks"]
    log.info("Source file loaded", source=source, total_chunks=len(raw_chunks))

    chunk_models = []
    failed_chunks = 0
    for idx, chunk in enumerate(raw_chunks):
        chunk_uid = str(uuid4())
        try:
            model = ChunkInput(
                chunk_uid=chunk_uid,
                chunk_id=idx,
                type=chunk.get("type"),
                article_number=chunk.get("article_number"),
                annex_number=chunk.get("annex_number"),
                title=chunk.get("title"),
                part_index=chunk.get("part_index", 0),
                part_count=chunk.get("part_count", 1),
                text=chunk["text"],
                source_file=source.split("/")[-1],
                page=chunk.get("page"),
                char_length=len(chunk["text"]),
                law_passed_date=raw_data["law_passed_date"],
                ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                regulation_title=raw_data["regulation_title"],
                document_version=raw_data["document_version"],
                issuing_authority=raw_data["issuing_authority"],
            )
            chunk_models.append(model)
        
            if(limit_chunks):
                log.warning("limit_chunks enabled — stopping after first chunk")
                break

        except Exception as e:
            failed_chunks += 1
            log.error("Chunk model creation failed", exc=e, chunk_id=idx, chunk=chunk)
            raise e

    log.info("Chunks built", total=len(chunk_models), failed=failed_chunks)
    embedding_factory = EmbedderFactory(picked_model=RUN_MODE)
    log.info("Embedding started", chunk_count=len(chunk_models), model=RUN_MODE)

    with log.timer("embedding"):  # 7
        embeddings = embedding_factory.embed_documents(
            [chunk.text for chunk in chunk_models]
        )

    log.info("Embeddings ready", count=len(embeddings), dimensions=len(embeddings[0]) if embeddings else 0)

    if len(embeddings) != len(chunk_models):
        log.warning("Embedding/chunk count mismatch", chunks=len(chunk_models), embeddings=len(embeddings))

    storage_factory = StorageFactory(storage_type=RUN_MODE, index_collection_name="team03")
    if recreate_index:
        log.info("Recreating index", storage_type=RUN_MODE)
        storage_factory.create()

    add_input = AddChunksInput(chunks=chunk_models, embeddings=embeddings)
    storage_factory.add_chunks(add_input)
    log.info("Chunks uploaded to storage", count=len(chunk_models), storage_type=RUN_MODE)

def run_search(question: str):
    log.info("Query received", question=question, mode=RUN_MODE)

    storage_factory = StorageFactory(storage_type=RUN_MODE, index_collection_name='team03')
    embedding_factory = EmbedderFactory(picked_model=RUN_MODE)
    
    with log.timer("query embedding"):
        embedded_prompt = embedding_factory.embed_query(question)
    
    with log.timer("vector search"):  # 3
        result: ItemPaged = storage_factory.search(SearchInput(query_embedding=embedded_prompt))

    results = list(result)
    log.info("Search complete", hits=len(results))

    if not results:
        log.warning("No results returned", question=question)
