from pydantic import BaseModel, field_validator


class ChunkInput(BaseModel):
    chunk_uid: str = None
    chunk_id: int = None
    type: str = None
    article_number: str | None = None
    annex_number: str | None = None
    title: str | None = None
    text: str = None
    source_file: str = None
    page: int | None = None
    part_index: int = 0
    part_count: int = 1
    char_length: int = None
    law_passed_date: str | None = None
    ingested_at: str | None = None
    regulation_title: str | None = None
    document_version: str | None = None
    issuing_authority: str | None = None


class AddChunksInput(BaseModel):
    chunks: list[ChunkInput]
    embeddings: list[list[float]]

    @field_validator("embeddings")
    @classmethod
    def embeddings_match_chunks(cls, v, info):
        chunks = info.data.get("chunks", [])
        if len(v) != len(chunks):
            raise ValueError(
                f"Embeddings length ({len(v)}) must match chunks length ({len(chunks)})"
            )
        return v
