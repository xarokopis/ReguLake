import os
from pathlib import Path
import sys

import shutil

from gtgh_team3_compliance_assistant.config import PDF_DIR, CHROMA_PATH, METADATA_FILE, RUN_MODE
from gtgh_team3_compliance_assistant.embedding.EmbedderFactory import EmbedderFactory
from gtgh_team3_compliance_assistant.storing.storageFactory import StorageFactory
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline
from gtgh_team3_compliance_assistant.logger.Logger import log

def run_ingestion(args):
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
                log.debug("Adding DLL directory", path=str(dll_dir))
                os.add_dll_directory(str(dll_dir))


    add_windows_dll_directories()

    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    log.info("Cleared ChromaDB", path=str(CHROMA_PATH))

    METADATA_FILE.write_text("[]")
    log.info("Reset metadata file", path=str(METADATA_FILE))

    embedding_model = EmbedderFactory(picked_model=RUN_MODE)
    log.info("Embedder ready", model="cloud")

    vector_store = StorageFactory(storage_type=RUN_MODE, index_collection_name=os.getenv("CLOUD_EMBEDDING_MODEL_NAME"))
    log.info("Vector store ready", storage_type="cloud", index=os.getenv("CLOUD_EMBEDDING_MODEL_NAME"))
    
    if recreate_index:
        log.info("Recreating index")
        vector_store.create()

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    log.info(f"PDFs discovered", count=len(pdf_files), directory=str(PDF_DIR))

    if not pdf_files:
        log.warning("No PDFs found, ingestion will do nothing", directory=str(PDF_DIR))

    for pdf in pdf_files:
        log.info("Ingesting PDF", file=pdf.name)
        print("=" * 50)
        try:
            RAGPipeline(
                pdf_path=str(pdf),
                embedding_model=embedding_model,
                vector_store=vector_store,
            ).ingest()
            log.info("PDF ingested successfully", file=pdf.name)
        except Exception as e:
            log.error("PDF ingestion failed", exc=e, file=pdf.name)

    log.info("Ingestion complete", total_pdfs=len(pdf_files))