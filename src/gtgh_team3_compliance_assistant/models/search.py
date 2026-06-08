from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query_embedding: list[float]
    top_k: int = Field(default=5, gt=0)


class SearchResult(BaseModel):
    chunk_uid: str
    content: str
    metadata: dict
    distance: float
