import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# run locally or on cloud
RUN_MODE = 'cloud' # 'cloud' or 'local'

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
EXTRACTED_DIR = DATA_DIR / "extracted"
CHUNK_DIR = DATA_DIR / "chunks"
CHROMA_PATH = BASE_DIR / "chroma_db"

METADATA_DIR = DATA_DIR / "metadata"
METADATA_FILE = METADATA_DIR / "documents.json"

for _dir in (PDF_DIR, EXTRACTED_DIR, CHUNK_DIR, METADATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "compliance_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Databricks local-connect API settings
DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG", "accenture2026dbcks")
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "team3")
DATABRICKS_VOLUME = os.getenv("DATABRICKS_VOLUME", "volume")
DATABRICKS_EMBEDDED_TABLE = os.getenv("DATABRICKS_EMBEDDED_TABLE", "gold_embedded_chunks")
