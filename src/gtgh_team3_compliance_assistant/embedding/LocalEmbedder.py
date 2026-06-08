from typing import Any

from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


class LocalEmbedder(BaseModel):
    model_name: str
    model: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any):
        self.model = SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()
