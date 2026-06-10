import sys, os
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

endpoint = os.getenv('AI_SEARCH_ENDPOINT')
key = os.getenv('AI_SEARCH_API_KEY')

# Show index fields
index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
index = index_client.get_index('team03')
print("=== INDEX FIELDS ===")
for f in index.fields:
    print(f" - {f.name} ({f.type})")

# Show a sample document
print("\n=== SAMPLE DOCUMENT ===")
search_client = SearchClient(endpoint=endpoint, index_name='team03', credential=AzureKeyCredential(key))
for doc in search_client.search(search_text="*", top=1):
    for k, v in doc.items():
        if k != 'embedding':
            print(f"  {k}: {str(v)[:100]}")
