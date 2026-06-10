import sys, os
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()

from gtgh_team3_compliance_assistant.embedding.CloudEmbedder import CloudEmbedder
from gtgh_team3_compliance_assistant.storing.cloudStorage import CloudStorage
from gtgh_team3_compliance_assistant.models.search import SearchInput

print("Embedding deployment:", os.getenv("CLOUD_EMBEDDING_DEPLOYMENT"))

embedder = CloudEmbedder()
storage = CloudStorage()

questions = [
    "What does GDPR say about data retention?",
    "Rights of data subjects under EU law",
    "What is MiFID II?",
]

for q in questions:
    print(f"\nQ: {q}")
    emb = embedder.embed_query(q)
    results = storage.search(SearchInput(query_embedding=emb, top_k=3))
    for r in results:
        print(f"  [{r.metadata.get('regulation_title')} | Art {r.metadata.get('article_number')}] score={1.0 - r.distance:.3f} — {r.content[:80]}")
