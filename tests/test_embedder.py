import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from gtgh_team3_compliance_assistant.embedding.LocalEmbedder import LocalEmbedder
from gtgh_team3_compliance_assistant.config import EMBEDDING_MODEL_NAME


@pytest.fixture
def embedder():
  with patch("gtgh_team3_compliance_assistant.embedding.LocalEmbedder.SentenceTransformer") as mock_st:
    mock_instance = MagicMock()
    mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_st.return_value = mock_instance
    yield LocalEmbedder(model_name='mock-model')


# -------------------
# ------ Tests ------
# -------------------

def test_embedder_initialization(embedder):
  assert embedder.model_name == "mock-model"
  assert embedder.model is not None

def test_embed_returns_array(embedder):
  result = embedder.embed("hello world")
  assert isinstance(result, np.ndarray)

SKIP_INTEGRATION_TESTS = True
@pytest.mark.skipif(SKIP_INTEGRATION_TESTS, reason="Set SKIP_INTEGRATION_TESTS=1 to run integration tests")
def test_embed_real_model():
  real_embedded = LocalEmbedder(model_name=EMBEDDING_MODEL_NAME)
  result = real_embedded.embed("hello world")
  result_empty_string = real_embedded.embed("")
  result_batch = real_embedded.embed_batch([
    "hello world",
    "hello world 2",
    "hello world 3",
    "hello world 4",
    "hello world 5",
    "hello world 6"
    ])
  assert isinstance(result, np.ndarray)
  assert len(result) > 0
  assert isinstance(result_empty_string, np.ndarray)
  assert isinstance(result_batch, np.ndarray)
  assert len(result_batch) == 6
  assert len(result_batch[0]) > 0
