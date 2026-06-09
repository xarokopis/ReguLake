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

if __name__ == "__main__":
    embedder_factory = EmbedderFactory(picked_model='cloud')
    embedder_model = embedder_factory.get_model()
    print(embedder_factory)