from pydantic import BaseModel


class DownloaderFromEUVariables(BaseModel):
    celex_id: str
    language: str | None = None
