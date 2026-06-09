import os
import pytest
import numpy as np
from pydantic import ValidationError
from unittest.mock import MagicMock, patch
from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore
from gtgh_team3_compliance_assistant.models.Chunks import ChunkInput, AddChunksInput
from gtgh_team3_compliance_assistant.models.Search import SearchInput, SearchResult

# Fixures
@pytest.fixture
def mock_collection():
    collection = MagicMock()

    collection.query.return_value = {
        "ids": [[1, 2]],
        "documents": [["content 1", "content 2"]],
        "metadatas": [
            [
                {"source_file": "doc.pdf", "page_number": 1, "chunk_index": 0},
                {"source_file": "doc.pdf", "page_number": 1, "chunk_index": 1}
            ]
        ],
        "distances": [[0.1, 0.2]]
    }

    return collection

@pytest.fixture
def store(mock_collection):
    with patch("gtgh_team3_compliance_assistant.storing.Storage.chromadb.PersistentClient") as mock_client:
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        yield ChromaVectorStore(
            persist_path="/tmp/test_chroma",
            collection_name="test_collection",
        )

@pytest.fixture
def sample_chunks():
    return [
        ChunkInput(
            chunk_id=1,
            type="article",
            article="Article 1",
            article_number="1",
            recital_number=1,
            title="Title 1",
            text="hello world",
            source_file="doc.pdf",
            page=1,
            char_length=11,
        ),
        ChunkInput(
            chunk_id=2,
            type="recital",
            article="Article 2",
            article_number="2",
            recital_number=2,
            title="Title 2",
            text="foo bar",
            source_file="doc.pdf",
            page=2,
            char_length=7,
        ),
    ]

@pytest.fixture
def sample_embeddings():
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


# Initialization Tests
def test_store_initializes(store):
    assert store.persist_path == "/tmp/test_chroma"
    assert store.collection_name == "test_collection"
    assert store.client is not None
    assert store.collection is not None

def test_store_creates_collection_with_cosine(store):
    store.client.get_or_create_collection.assert_called_once_with(
        name="test_collection",
        metadata={"hnsw:space": "cosine"}
    )

# AddChunksInput validation tests
def test_add_chunks_input_mismatched_embeddings_raises(sample_chunks):
    with pytest.raises(ValueError, match="Embeddings length"):
        AddChunksInput(chunks=sample_chunks, embeddings=[[0.1, 0.2, 0.3]])

def test_add_chunks_input_valid(sample_chunks, sample_embeddings):
    input = AddChunksInput(chunks=sample_chunks, embeddings=sample_embeddings)
    assert len(input.chunks) == len(input.embeddings)

# add_chunks tests
def test_add_chunks_calls_upsert(store, sample_chunks, sample_embeddings):
    input = AddChunksInput(chunks=sample_chunks, embeddings=sample_embeddings)
    store.add_chunks(input)
    store.collection.upsert.assert_called_once()

def test_add_chunks_passes_correct_ids(store, sample_chunks, sample_embeddings):
    input = AddChunksInput(chunks=sample_chunks, embeddings=sample_embeddings)
    store.add_chunks(input)
    call_kwargs = store.collection.upsert.call_args.kwargs
    assert call_kwargs["ids"] == [1, 2]

def test_add_chunks_passes_correct_documents(store, sample_chunks, sample_embeddings):
    input = AddChunksInput(chunks=sample_chunks, embeddings=sample_embeddings)
    store.add_chunks(input)
    call_kwargs = store.collection.upsert.call_args.kwargs
    assert call_kwargs["documents"] == ["hello world", "foo bar"]

def test_add_chunks_passes_correct_metadatas(store, sample_chunks, sample_embeddings):
    input = AddChunksInput(chunks=sample_chunks, embeddings=sample_embeddings)
    store.add_chunks(input)
    call_kwargs = store.collection.upsert.call_args.kwargs
    assert call_kwargs["metadatas"][0]["source_file"] == "doc.pdf"
    assert call_kwargs["metadatas"][0]["page_number"] == 1

def test_add_chunks_mismatched_embeddings_raises(sample_chunks):
    with pytest.raises(ValidationError, match="Embeddings length"):
        AddChunksInput(chunks=sample_chunks, embeddings=[[0.1, 0.2, 0.3]])  # only 1 embedding for 2 chunks

# --- search tests ---

def test_search_returns_list_of_search_results(store):
    input = SearchInput(query_embedding=[0.1, 0.2, 0.3], top_k=2)
    results = store.search(input)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_returns_correct_number_of_results(store):
    input = SearchInput(query_embedding=[0.1, 0.2, 0.3], top_k=2)
    results = store.search(input)
    assert len(results) == 2


def test_search_result_fields(store):
    input = SearchInput(query_embedding=[0.1, 0.2, 0.3], top_k=2)
    results = store.search(input)
    assert results[0].chunk_id == 1
    assert results[0].content == "content 1"
    assert results[0].distance == 0.1
    assert results[1].chunk_id == 2


def test_search_calls_collection_query(store):
    input = SearchInput(query_embedding=[0.1, 0.2, 0.3], top_k=2)
    store.search(input)
    store.collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2, 0.3]],
        n_results=2,
    )


def test_search_invalid_top_k_raises(store):
    with pytest.raises(ValueError):
        SearchInput(query_embedding=[0.1, 0.2, 0.3], top_k=0)  # gt=0 violated

# --- Integration test ---
SKIP_INTEGRATION_TESTS = True
@pytest.mark.skipif(SKIP_INTEGRATION_TESTS, reason="Set SKIP_INTEGRATION_TESTS=1 to run integration tests")
def test_real_chroma_store():
    store = ChromaVectorStore(
        persist_path="/tmp/real_chroma_test",
        collection_name="integration_test",
    )
    chunks = [
        ChunkInput(
            chunk_id="real_chunk_1",
            content="real content",
            source_file="real.pdf",
            page_number=1,
            chunk_index=0,
        )
    ]
    store.add_chunks(AddChunksInput(chunks=chunks, embeddings=[[0.1, 0.2, 0.3]]))
    results = store.search(SearchInput(query_embedding=[0.1, 0.2, 0.3], top_k=1))
    assert len(results) == 1
    assert results[0].chunk_id == "real_chunk_1"
