import chromadb
from chromadb.config import Settings
from pydantic import BaseModel

from gtgh_team3_compliance_assistant.models.chunks import AddChunksInput
from gtgh_team3_compliance_assistant.models.search import SearchInput, SearchResult


class ChromaVectorStore(BaseModel):
    persist_path: str
    collection_name: str

    client: chromadb.ClientAPI = None
    collection: chromadb.Collection = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context):
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, input: AddChunksInput) -> None:
        ids = [chunk.chunk_uid for chunk in input.chunks]
        documents = [chunk.text for chunk in input.chunks]

        metadatas = [
            {
                "source_file": chunk.source_file,
                "page_number": chunk.page if chunk.page is not None else -1,
                "chunk_index": chunk.chunk_id,
                "type": chunk.type,
                "article_number": chunk.article_number or "",
                "annex_number": chunk.annex_number or "",
                "title": chunk.title or "",
                "part_index": chunk.part_index,
                "part_count": chunk.part_count,
                "law_passed_date": chunk.law_passed_date or "",
                "ingested_at": chunk.ingested_at or "",
                "regulation_title": chunk.regulation_title or "",
                "document_version": chunk.document_version or "",
                "issuing_authority": chunk.issuing_authority or "",
            }
            for chunk in input.chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=input.embeddings,
            metadatas=metadatas,
        )

    def search(self, input: SearchInput) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[input.query_embedding],
            n_results=input.top_k,
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            SearchResult(
                chunk_uid=ids[i],
                content=documents[i],
                metadata=metadatas[i],
                distance=distances[i],
            )
            for i in range(len(ids))
        ]
