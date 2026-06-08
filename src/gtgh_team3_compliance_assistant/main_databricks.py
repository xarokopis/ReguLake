from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from gtgh_team3_compliance_assistant.api.health import router as health_router
from gtgh_team3_compliance_assistant.api.query_databricks import router as databricks_query_router

app = FastAPI(title="Compliance Assistant - Databricks Local Connect")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(databricks_query_router)


@app.get("/")
def root():
    return {"message": "running with Databricks-backed retrieval"}