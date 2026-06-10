from gtgh_team3_compliance_assistant.models.chunks import AddChunksInput
from gtgh_team3_compliance_assistant.models.search import SearchInput, SearchResult

from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
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
dotenv.load_dotenv()

from gtgh_team3_compliance_assistant.logger.Logger import log

TEAM_NAME = "team03"
endpoint=os.getenv("AZURE_SEARCH_ENDPOINT") or os.getenv("AI_SEARCH_ENDPOINT")
admin_key = os.getenv("AZURE_SEARCH_KEY") or os.getenv("AI_SEARCH_API_KEY")

DEFAULT_FIELDS = [
    SimpleField(name="chunk_uid",         type=SearchFieldDataType.String,  key=True),
    SimpleField(name="chunk_id",          type=SearchFieldDataType.Int32,   filterable=True),
    SimpleField(name="source_file",       type=SearchFieldDataType.String,  filterable=True, facetable=True),
    SimpleField(name="regulation_title",  type=SearchFieldDataType.String,  filterable=True, facetable=True),
    SimpleField(name="document_version",  type=SearchFieldDataType.String,  filterable=True),
    SimpleField(name="issuing_authority", type=SearchFieldDataType.String,  filterable=True),
    SimpleField(name="law_passed_date",   type=SearchFieldDataType.String,  filterable=True, sortable=True),
    SimpleField(name="ingested_at",       type=SearchFieldDataType.String,  filterable=True, sortable=True),
    SimpleField(name="type",              type=SearchFieldDataType.String,  filterable=True, facetable=True),
    SimpleField(name="article_number",    type=SearchFieldDataType.String,  filterable=True),
    SimpleField(name="annex_number",      type=SearchFieldDataType.String,  filterable=True),
    SimpleField(name="page",       type=SearchFieldDataType.Int32,   filterable=True, sortable=True),
    SimpleField(name="part_index",        type=SearchFieldDataType.Int32,   filterable=True),
    SimpleField(name="part_count",        type=SearchFieldDataType.Int32,   filterable=True),
    SimpleField(name="char_length",       type=SearchFieldDataType.Int32,   filterable=True),
    SearchableField(name="text",    type=SearchFieldDataType.String),
    SearchableField(name="title",         type=SearchFieldDataType.String),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name="vector-profile"
    )
]

class CloudStorage(BaseModel):
    index_name: str = TEAM_NAME
    _client: SearchClient = PrivateAttr()
    _vector_search: VectorSearch = PrivateAttr()
    _fields: list = PrivateAttr()

    def model_post_init(self, __context):
        admin_key = os.getenv("AZURE_SEARCH_KEY") or os.getenv("AI_SEARCH_API_KEY")

        if not admin_key:
            raise Exception("Azure Admin key does not exist")
        self._client = SearchClient(
            endpoint=endpoint,
            index_name=TEAM_NAME,
            credential=AzureKeyCredential(admin_key)
        )

        # Check if index exists
        index_client = SearchIndexClient(
            endpoint=endpoint, 
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

        try:
            index_client.get_index(name=TEAM_NAME)
            log.info(f"Index '{TEAM_NAME}' already exists.")
        except ResourceNotFoundError:
            log.info(f"Index '{TEAM_NAME}' not found. Creating it now...")
            index = SearchIndex(name=TEAM_NAME, fields=DEFAULT_FIELDS, vector_search=self._vector_search)
            index_client.create_index(index)
    
    def recreateIndex(self, fields: list = DEFAULT_FIELDS):
        index_client = SearchIndexClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(admin_key)
        )

        try:
            log.info(f"Deleting index '{TEAM_NAME}' if it exists...")
            index_client.delete_index(TEAM_NAME)
        except ResourceNotFoundError:
            log.info(f"Index {TEAM_NAME} didn't exist yet, skipping deletion.")
        
        new_index_definition = SearchIndex(name=TEAM_NAME, fields=fields, vector_search=self._vector_search)
        log.info(f"Creating fresh index '{TEAM_NAME}'...")
        index_client.create_index(new_index_definition)
        log.info(f"Index '{TEAM_NAME}' recreated successfully!")
    
    def add_chunks(self, input: AddChunksInput) -> None:
        documents = self._process_chunks(input)
        if type(documents) != list:
            documents = [documents]

        self._client.upload_documents(
            documents=documents
        )

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
            select=[
                "chunk_uid",
                "chunk_id",
                "source_file",
                "regulation_title",
                "document_version",
                "issuing_authority",
                "law_passed_date",
                "ingested_at",
                "type",
                "article_number",
                "annex_number",
                "page",
                "part_index",
                "part_count",
                "char_length",
                "text",
                "title"
            ]
        )
        search_results = []
        for row in results:
            search_results.append(SearchResult(
                chunk_uid=row["chunk_uid"],
                content=row["text"],
                metadata={
                    "source_file": row.get("source_file", ""),
                    "regulation_title": row.get("regulation_title", ""),
                    "document_version": row.get("document_version", ""),
                    "article_number": row.get("article_number", ""),
                    "annex_number": row.get("annex_number", ""),
                    "page": row.get("page"),
                    "part_index": row.get("part_index", 0),
                    "part_count": row.get("part_count", 1),
                    "type": row.get("type", ""),
                    "title": row.get("title", ""),
                },
                distance=1.0 - row.get("@search.score", 0.0),
            ))
        return search_results
    
    def _process_chunks(self, input_data: AddChunksInput) -> list[dict]:
        return [
            {**chunk.model_dump(), "embedding": emb}
            for chunk, emb in zip(input_data.chunks, input_data.embeddings)
        ]
