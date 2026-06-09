"""
creating an index
"""
from gtgh_team3_compliance_assistant.models.chunks import AddChunksInput
from gtgh_team3_compliance_assistant.models.search import SearchInput
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.core.credentials import AzureKeyCredential

import os
import dotenv
from pydantic import BaseModel, PrivateAttr
from typing import List
dotenv.load_dotenv()

TEAM_NAME = "team03"
endpoint=os.getenv("AZURE_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY")

DEFAULT_FIELDS = [
    SimpleField(
        name="chunk_id",
        type=SearchFieldDataType.String,
        key=True
    ),
    SearchableField(
        name="chunk_text",
        type=SearchFieldDataType.String
    ),
    SearchField(
        name="embedding",

        type=SearchFieldDataType.Collection(
            SearchFieldDataType.Single
        ),

        searchable=True,

        vector_search_dimensions=1536,

        vector_search_profile_name=
            "vector-profile"
    )
]

class CloudStorage(BaseModel):
    index_name: str = TEAM_NAME
    _client: SearchClient = PrivateAttr()
    _vector_search: VectorSearch = PrivateAttr()
    _fields: List[SimpleField, SearchableField, SearchField] = PrivateAttr()

    def model_post_init(self, __context):
        admin_key = os.getenv("AZURE_SEARCH_KEY")
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=TEAM_NAME,
            credential=AzureKeyCredential(admin_key)
        )

        self._vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config"
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )
    
    def createIndex(self, fields: list = DEFAULT_FIELDS):
        index = SearchIndex(
            name=TEAM_NAME,
            fields=fields,
            vector_search=self._vector_search
        )

        self._client.create_or_update_index(index)
    
    def add_chunks(self, input: AddChunksInput) -> None:
        documents = self._process_chunks(input)
        if type(documents) != list:
            documents = [documents]

        result = self._client.upload_documents(
            documents=documents
        )

        print(result) # TODO: Remove this line

    def search(self, input: SearchInput) -> list:
        results = self._client.search(
            search_text=None,
            vector_queries=[
                VectorizedQuery(
                    vector=input.query_embedding,
                    k_nearest_neighbors=input.top_k,
                    fields="embedding"
                )
            ],

            # TODO: Change Select Statement
            select=[
                "chunk_id",
                "chunk_text"
            ]
        )
        print("Warning: SELECT statement hasn't been changed")

        return results
    
    def _process_chunks(input_data: AddChunksInput) -> list[dict]:
        return [
            {**chunk.model_dump(), "embedding": emb}
            for chunk, emb in zip(input_data.chunks, input_data.embeddings)
        ]



if __name__ == "__main__":

    cloud_storage = CloudStorage()
    cloud_storage.createIndex(DEFAULT_FIELDS)
    print("Index created.")
