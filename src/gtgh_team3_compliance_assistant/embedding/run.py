import json

from uuid import uuid4

from datetime import datetime, timezone

from gtgh_team3_compliance_assistant.config import RUN_MODE
from gtgh_team3_compliance_assistant.embedding.EmbedderFactory import EmbedderFactory
from gtgh_team3_compliance_assistant.models.chunks import ChunkInput
from gtgh_team3_compliance_assistant.logger.Logger import log

def run_embed(source: str, limit_chunks: bool, save_embeds: bool = False, destination: str= ""):
    log.info("Embedding started", source=source, limit_chunks=limit_chunks, destination=destination)

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
    
    if save_embeds:
        log.info("Saving embeddings to local file", target_path=destination)
        
        payload = []
        for chunk, embedding in zip(chunk_models, embeddings):
            chunk_dict = chunk.model_dump() 
            chunk_dict["embedding"] = embedding
            payload.append(chunk_dict)

        with open(destination, "w") as f:
            json.dump(payload, f, indent=4)

        log.info("Embeddings successfully saved", target_path=destination)