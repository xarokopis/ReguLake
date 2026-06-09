from typing import Any, Literal

from pydantic import BaseModel
from gtgh_team3_compliance_assistant.embedding.CloudEmbedder import CloudEmbedder
from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder


class EmbedderFactory(BaseModel):
    picked_model: Literal['local', 'cloud'] = 'local' # 0 = local, 1 = cloud
    model: LocalEmbedder | CloudEmbedder = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any):
        self.set_model(self.picked_model)
    
    def set_model(self, picked_model: int = 1):
        if picked_model == 'local':
            self.model = LocalEmbedder(self.model_name)
        elif picked_model == 'cloud':
            self.model = CloudEmbedder()
        else:
            raise ValueError("Unknown Model")

    def get_model(self) -> LocalEmbedder | CloudEmbedder:
        return self.model;

    def embed_query(self, text: str) -> list[float]:
        return self.model.embed_query(text);

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_documents(texts)