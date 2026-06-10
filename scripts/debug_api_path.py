import sys, os
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()

from gtgh_team3_compliance_assistant.embedding.EmbedderFactory import EmbedderFactory
from gtgh_team3_compliance_assistant.storing.storageFactory import StorageFactory
from gtgh_team3_compliance_assistant.pipeline.rag_pipeline import RAGPipeline

embedding_model = EmbedderFactory(picked_model='cloud')
vector_store = StorageFactory(storage_type='cloud', index_collection_name="team03")
rag = RAGPipeline(pdf_path=None, embedding_model=embedding_model, vector_store=vector_store, llm=None)

question = "What are the rights of data subjects under GDPR?"
print(f"Q: {question}\n")

results = rag.retrieve(question, top_k=5)
print(f"=== RETRIEVED {len(results)} CHUNKS ===")
for i, r in enumerate(results):
    print(f"[{i+1}] {r.metadata.get('regulation_title')} | Art {r.metadata.get('article_number')} | score={1.0 - r.distance:.3f}")
    print(f"     {r.content[:120]}")

print("\n=== CONTEXT PASSED TO LLM ===")
context = rag.build_context(results)
print(context[:800] if context else "(EMPTY)")
