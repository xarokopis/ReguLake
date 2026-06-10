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