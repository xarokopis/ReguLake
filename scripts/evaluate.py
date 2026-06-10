from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
import dotenv
dotenv.load_dotenv()

from gtgh_team3_compliance_assistant.config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
)

CLOUD_EMBEDDING_ENDPOINT = os.getenv("CLOUD_EMBEDDING_ENDPOINT")
CLOUD_EMBEDDING_API_KEY = os.getenv("CLOUD_EMBEDDING_API_KEY")
CLOUD_EMBEDDING_API_VERSION = os.getenv("CLOUD_EMBEDDING_API_VERSION")
CLOUD_EMBEDDING_DEPLOYMENT = os.getenv("CLOUD_EMBEDDING_DEPLOYMENT")
from gtgh_team3_compliance_assistant.embedding.CloudEmbedder import CloudEmbedder
from gtgh_team3_compliance_assistant.storing.localStorage import ChromaVectorStore
from gtgh_team3_compliance_assistant.model_communication.llm import ChatLLM
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AZURE_CHROMA_PATH = ROOT_DIR / "chroma_db_azure"
AZURE_COLLECTION_NAME = "compliance_docs_azure"
GOLDEN_DATASET_PATH = ROOT_DIR / "data" / "evaluation" / "golden_dataset.json"
OUTPUT_PATH = ROOT_DIR / "data" / "evaluation" / "results" / "eval_results.csv"


def build_rag() -> RAGPipeline:
    embedding_model = CloudEmbedder()
    vector_store = ChromaVectorStore(
        persist_path=str(AZURE_CHROMA_PATH),
        collection_name=AZURE_COLLECTION_NAME,
    )
    llm = ChatLLM()
    return RAGPipeline(pdf_path=None, embedding_model=embedding_model, vector_store=vector_store, llm=llm)


def build_ragas_judges():
    ragas_llm = LangchainLLMWrapper(
        AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        AzureOpenAIEmbeddings(
            azure_endpoint=CLOUD_EMBEDDING_ENDPOINT,
            api_key=CLOUD_EMBEDDING_API_KEY,
            api_version=CLOUD_EMBEDDING_API_VERSION,
            azure_deployment=CLOUD_EMBEDDING_DEPLOYMENT,
        )
    )
    return ragas_llm, ragas_embeddings


async def score_one(
    question: str,
    contexts: list[str],
    answer: str,
    ragas_llm: LangchainLLMWrapper,
    ragas_embeddings: LangchainEmbeddingsWrapper,
) -> dict:
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=contexts,
        response=answer,
    )
    faithfulness = await Faithfulness(llm=ragas_llm).single_turn_ascore(sample)
    response_relevancy = await ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings).single_turn_ascore(sample)
    context_precision = await LLMContextPrecisionWithoutReference(llm=ragas_llm).single_turn_ascore(sample)
    return {
        "faithfulness": faithfulness,
        "response_relevancy": response_relevancy,
        "context_precision": context_precision,
    }


async def main() -> None:
    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded %d questions from golden dataset", len(dataset))

    rag = build_rag()
    ragas_llm, ragas_embeddings = build_ragas_judges()

    results = []
    for i, item in enumerate(dataset, 1):
        logger.info("Query %d/%d: %s", i, len(dataset), item["question"])
        try:
            response = rag.ask(item["question"])
            contexts = [r["content"] for r in response["retrieved_chunks"]]
            scores = await score_one(item["question"], contexts, response["answer"], ragas_llm, ragas_embeddings)
            results.append({
                "act": item["act"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": response["answer"],
                "faithfulness": scores["faithfulness"],
                "response_relevancy": scores["response_relevancy"],
                "context_precision": scores["context_precision"],
            })
        except Exception as e:
            logger.error("Row %d failed: %s", i, e, exc_info=True)

    if not results:
        logger.error("No results to write.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results saved to: %s", OUTPUT_PATH)


if __name__ == "__main__":
    asyncio.run(main())
