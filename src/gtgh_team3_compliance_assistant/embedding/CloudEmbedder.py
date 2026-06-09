from typing import Any

from pydantic import BaseModel, PrivateAttr
from sentence_transformers import SentenceTransformer

from openai import AzureOpenAI
import os
import dotenv
dotenv.load_dotenv()

class CloudEmbedder(BaseModel):
    _model: AzureOpenAI = PrivateAttr()
    _config: dict = PrivateAttr()
    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any):
        self._config = {
            "endpoint": os.getenv("CLOUD_EMBEDDING_ENDPOINT"),
            "model_name": os.getenv("CLOUD_EMBEDDING_MODEL_NAME"),
            "deployment": os.getenv("CLOUD_EMBEDDING_DEPLOYMENT"),
            "api_key": os.getenv("CLOUD_EMBEDDING_API_KEY"),
            "api_version": os.getenv("CLOUD_EMBEDDING_API_VERSION"),
        }
        self._model = AzureOpenAI(
            azure_endpoint=self._config["endpoint"],
            api_key=self._config["api_key"],
            api_version=self._config["api_version"],
        )

    def embed_query(self, text: str) -> list[float]:
        response = self._model.embeddings.create(
            model=self._config["deployment"],
            input=text
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._model.embeddings.create(
            model=self._config["deployment"],
            input=texts
        )
        return response.data[0].embedding
