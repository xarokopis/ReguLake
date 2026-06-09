
from __future__ import annotations
import os
import sys
from pathlib import Path

import argparse

def runAPI(args):
    port = args.port;
    host = args.host;
    dev_env = args.dev;
    print(f"Attempting to run API on host {host}, port {port}, as dev {dev_env} ")

    # All API imports
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from gtgh_team3_compliance_assistant.api.health import router as health_router
    from gtgh_team3_compliance_assistant.api.ingestion import router as ingestion_router
    from gtgh_team3_compliance_assistant.api.query import router as query_router

    app = FastAPI(title="Compliance Assistant")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(ingestion_router)
    app.include_router(query_router)

    @app.get("/")
    def root():
        return {"message": "running"}
    
    uvicorn.run(app, host=host, port=port, reload=dev_env)

def runIngestion(args):
    source = args.source # source path or None
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
    from gtgh_team3_compliance_assistant.config import (
        PDF_DIR, CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, METADATA_FILE,
    )
    from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder
    from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore
    from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    print("Cleared old ChromaDB\n")

    METADATA_FILE.write_text("[]")
    print("Cleared documents.json\n")

    embedding_model = LocalEmbedder(model_name=EMBEDDING_MODEL_NAME)
    vector_store = ChromaVectorStore(persist_path=str(CHROMA_PATH), collection_name=COLLECTION_NAME)

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
    parser_ingestion.set_defaults(func=runIngestion)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()