
from __future__ import annotations
import os
import sys
from pathlib import Path

import argparse

from gtgh_team3_compliance_assistant.config import RUN_MODE

def runAPI(args):
    port = args.port;
    host = args.host;
    dev_env = args.dev;
    print(f"Attempting to run API on host {host}, port {port}, as dev {'in development mode' if(dev_env) else 'in production mode'}")

    # All API imports
    import uvicorn
    uvicorn.run("gtgh_team3_compliance_assistant.api.run:app", host=host, port=port, reload=dev_env)

def runIngestion(args):
    source = args.source # source path or None
    recreate_index = args.recreate_index
    # Ingestion Imports
    ROOT_DIR = Path(__file__).resolve().parents[1]
    SRC_DIR = ROOT_DIR / "src"

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


    def add_windows_dll_directories() -> None:
        if os.name != "nt":
            return

        candidate_dirs = [
            ROOT_DIR / ".venv" / "Lib" / "site-packages" / "sklearn" / ".libs",
            ROOT_DIR / ".venv" / "Lib" / "site-packages" / "scipy" / ".libs",
            ROOT_DIR / ".venv" / "Lib" / "site-packages" / "numpy.libs",
        ]

        for dll_dir in candidate_dirs:
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))


    add_windows_dll_directories()

    import shutil
    from gtgh_team3_compliance_assistant.config import PDF_DIR, CHROMA_PATH, METADATA_FILE
    from gtgh_team3_compliance_assistant.embedding.EmbedderFactory import EmbedderFactory
    from gtgh_team3_compliance_assistant.storing.storageFactory import StorageFactory
    from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    print("Cleared old ChromaDB\n")

    METADATA_FILE.write_text("[]")
    print("Cleared documents.json\n")

    embedding_model = EmbedderFactory(picked_model=RUN_MODE)
    vector_store = StorageFactory(storage_type=RUN_MODE, index_collection_name=os.getenv("CLOUD_EMBEDDING_MODEL_NAME"))
    if recreate_index:
        vector_store.create()

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs\n")

    for pdf in pdf_files:
        print("=" * 50)
        RAGPipeline(
            pdf_path=str(pdf),
            embedding_model=embedding_model,
            vector_store=vector_store,
        ).ingest()

    print("\nDone. Check data/chunks/ for JSON files.")

def runStorage(args):
    source = args.source
    recreate_index = args.recreate_index
    limit_chunks = args.limit_chunks

    # Required Imports
    from gtgh_team3_compliance_assistant.models.chunks import AddChunksInput
    from gtgh_team3_compliance_assistant.storing.storageFactory import StorageFactory
    from gtgh_team3_compliance_assistant.embedding.EmbedderFactory import EmbedderFactory
    from gtgh_team3_compliance_assistant.models.chunks import ChunkInput
    import json
    from uuid import uuid4
    from pprint import pprint
    from datetime import datetime
    from datetime import timezone
    
    with open(source, 'r') as file:
        raw_data = json.load(file)
    raw_chunks = raw_data["chunks"]

    print("Creating chunks")
    chunk_models = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_uid = str(uuid4())
        try:
            model = ChunkInput(
                chunk_uid=chunk_uid,
                chunk_id=idx,
                type=chunk.get("type"),
                article_number=chunk.get("article_number"),
                annex_number=chunk.get("annex_number"),
                title=chunk.get("title"),
                part_index=chunk.get("part_index", 0),
                part_count=chunk.get("part_count", 1),
                text=chunk["text"],
                source_file=source.split("/")[-1],
                page=chunk.get("page"),
                char_length=len(chunk["text"]),
                law_passed_date=raw_data["law_passed_date"],
                ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                regulation_title=raw_data["regulation_title"],
                document_version=raw_data["document_version"],
                issuing_authority=raw_data["issuing_authority"],
            )
            chunk_models.append(model)
        
            if(limit_chunks):
                break

        except Exception as e:
            print(f"Chunk failed: {idx}")
            print(chunk)
            raise e

    print("Created chunks")
    embedding_factory = EmbedderFactory(picked_model=RUN_MODE)
    print("Creating Embeds")
    embeddings = embedding_factory.embed_documents(
        [chunk.text for chunk in chunk_models]
    )
    print("len(embeddings) ==================== ", len(embeddings))
    print("len(embeddings[0]) ==================== ", len(embeddings[0]))
    print("len(chunks) ==================== ", len(raw_chunks))
    storage_factory = StorageFactory(storage_type=RUN_MODE, index_collection_name="team03")
    if recreate_index:
        storage_factory.create()

    add_input = AddChunksInput(chunks=chunk_models, embeddings=embeddings)
    storage_factory.add_chunks(add_input)

def runSearch(args):
    question = args.question
    from gtgh_team3_compliance_assistant.storing.storageFactory import StorageFactory
    from gtgh_team3_compliance_assistant.models.search import SearchInput, SearchResult
    from gtgh_team3_compliance_assistant.embedding.EmbedderFactory import EmbedderFactory
    from azure.core.paging import ItemPaged
    from pprint import pprint

    storage_factory = StorageFactory(storage_type=RUN_MODE, index_collection_name='team03')
    embedding_factory = EmbedderFactory(picked_model=RUN_MODE)
    embedded_prompt = embedding_factory.embed_query(question)

    result: ItemPaged = storage_factory.search(SearchInput(query_embedding=embedded_prompt))
    for res in result:
        pprint(res)

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to run API or Ingestion paths."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available subcommands"
    )
    # --- API Subcommand Setup ---
    parser_api = subparsers.add_parser("api", help="Run the API server")
    parser_api.add_argument(
        "-p", "--port", type=int, default=8000, help="Port to listen on"
    )
    parser_api.add_argument(
        "-H", "--host", type=str, default="localhost", help="Host ip address"
    )
    parser_api.add_argument(
        "-d", "--dev", "--development", action="store_true", help="Run API Server in development environment"
    )
    parser_api.set_defaults(func=runAPI)

    # --- Ingestion Subcommand Setup ---
    parser_ingestion = subparsers.add_parser(
        "ingest", help="Run the ingestion pipeline"
    )
    parser_ingestion.add_argument(
        "-s", "--source", required=False, help="Path to data source file"
    )
    parser_ingestion.add_argument(
        "-r", "--recreate-index", action="store_true", help="Force delete and recreate the Azure index before uploading"
    )
    parser_ingestion.set_defaults(func=runIngestion)

    # --- Store Subcommand Setup ---
    parser_ingestion = subparsers.add_parser(
        "store", help="Store JSON document data in Azure"
    )
    parser_ingestion.add_argument(
        "-s", "--source", required=True, help="Path to data source file"
    )
    parser_ingestion.add_argument(
        "-l", "--limit-chunks", action="store_true", help="Only ingest the first chunk for testing purposes"
    )
    parser_ingestion.add_argument(
        "-r", "--recreate-index", action="store_true", help="Force delete and recreate the Azure index before uploading"
    )
    parser_ingestion.set_defaults(func=runStorage)

    # --- Search Subcommand Setup ---
    parser_ingestion = subparsers.add_parser(
        "search", help="Search a query in Azure Storage"
    )
    parser_ingestion.add_argument(
        "-q", "--question", required=True, help="Question to search about"
    )
    parser_ingestion.set_defaults(func=runSearch)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()