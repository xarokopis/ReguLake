uv sync 
uv run -m scripts.ingest
uv run uvicorn gtgh_team3_compliance_assistant.main:app --reload
