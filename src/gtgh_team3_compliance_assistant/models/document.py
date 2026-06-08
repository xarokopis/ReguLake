from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    id: str
    title: str
    source: str
    url: str
    local_path: str
    last_modified: str | None = None
    ingested_at: str
    file_hash: str
