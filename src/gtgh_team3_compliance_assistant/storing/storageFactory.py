
from typing import Literal

from gtgh_team3_compliance_assistant.models.chunks import AddChunksInput
from gtgh_team3_compliance_assistant.models.search import SearchInput
from pydantic import BaseModel, PrivateAttr
from gtgh_team3_compliance_assistant.storing.cloudStorage import CloudStorage
from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore


class StorageFactory(BaseModel):
    storage_type = Literal["local", "cloud"] = "cloud"
    persist_path: str = None
    index_collection_name: str = None
    _storage_model: ChromaVectorStore | CloudStorage = PrivateAttr()

    def model_post_init(self, __context):
        if self.storage_type == "local":
            if(not self.persist_path or not self.index_collection_name):
                raise ValueError("Persist Path and Collection Name are required for local development")
            self._storage_model = ChromaVectorStore(persist_path=str(self.persist_path), collection_name=self.index_collection_name)
        
        elif self.storage_type == "cloud":
            self._storage_model = CloudStorage(index_name=self.index_collection_name)

    def create(self):
        if self.storage_type == "local":
            return
        
        if self.storage_type != "cloud":
            raise ValueError("I don't know how you got here but the storage type has to be either 'local' or 'cloud'")
        
        self._storage_model.createIndex()
    
    def search(self, input_data: SearchInput) -> list:
        self._storage_model.search(input_data)
    
    def add_chunks(self, input_data: AddChunksInput) -> None:
        self._storage_model.add_chunks(input_data)