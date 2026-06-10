import sys, os
sys.path.insert(0, 'src')
from dotenv import load_dotenv
load_dotenv()
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

client = SearchIndexClient(
    endpoint=os.getenv('AI_SEARCH_ENDPOINT'),
    credential=AzureKeyCredential(os.getenv('AI_SEARCH_API_KEY'))
)
stats = client.get_index_statistics('team03')
print(dict(stats))
