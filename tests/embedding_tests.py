"""
Embedding tests for LLM interfaces.
Tests embedding functionality and error handling for OpenAI and Gemini.
Note: Claude does not support embeddings.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv()

from llm_utils.wrapper import LLMWrapper
from llm_utils.openai_interface import OpenAIInterface
from llm_utils.gemini_interface import GeminiInterface


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# Test fixtures
@pytest.fixture
def llm():
    return LLMWrapper()


@pytest.fixture
def test_texts():
    """Simple test texts for embedding"""
    return ["cat", "kitty", "dog", "potato"]


@pytest.fixture
def test_ids(test_texts):
    """IDs for test texts"""
    return [str(i) for i in range(len(test_texts))]


@pytest.fixture
def batch_texts():
    """Larger batch of texts for concurrent embedding tests"""
    return ["cat", "kitty", "dog", "potato"] * 25  # 100 texts


@pytest.fixture
def batch_ids(batch_texts):
    """IDs for batch texts"""
    return [str(i) for i in range(len(batch_texts))]


# ============== OpenAI Embedding Tests ==============

class TestOpenAIEmbeddings:
    """Tests for OpenAI embeddings"""

    def test_openai_basic_embedding(self, llm, test_texts, test_ids):
        """Test basic OpenAI embedding generation"""
        _, embeddings = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=1536,
            db=None,
            name="test_openai_emb",
            verbose=False,
            model="text-embedding-3-small"
        )

        assert embeddings is not None
        assert len(embeddings) == len(test_texts)
        assert len(embeddings[0]) == 1536  # Verify dimension

    def test_openai_embedding_similarity(self, llm, test_texts, test_ids):
        """Test that semantically similar words have higher similarity (from demo.ipynb)"""
        _, embeddings = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=1536,
            db=None,
            name="test_openai_emb",
            verbose=False,
            model="text-embedding-3-small"
        )

        cat_emb = np.array(embeddings[0])
        kitty_emb = np.array(embeddings[1])
        dog_emb = np.array(embeddings[2])
        potato_emb = np.array(embeddings[3])

        sim_cat_kitty = cosine_similarity(cat_emb, kitty_emb)
        sim_dog_potato = cosine_similarity(dog_emb, potato_emb)

        # cat-kitty should be more similar than dog-potato
        assert sim_cat_kitty > sim_dog_potato, \
            f"Expected cat-kitty ({sim_cat_kitty:.4f}) > dog-potato ({sim_dog_potato:.4f})"

    def test_openai_batch_embedding(self, llm, batch_texts, batch_ids):
        """Test batch embedding processing"""
        _, embeddings = llm.embed(
            ids=batch_ids,
            texts=batch_texts,
            size=1536,
            db=None,
            name="test_openai_batch",
            verbose=False,
            model="text-embedding-3-small"
        )

        assert embeddings is not None
        assert len(embeddings) == len(batch_texts)

    def test_openai_different_dimensions(self, llm, test_texts, test_ids):
        """Test OpenAI embeddings with different output dimensions"""
        _, embeddings_small = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=256,
            db=None,
            name="test_openai_small",
            verbose=False,
            model="text-embedding-3-small"
        )

        assert embeddings_small is not None
        assert len(embeddings_small[0]) == 256


class TestOpenAIEmbeddingErrorHandling:
    """Tests for OpenAI embedding error handling"""

    def test_openai_embedding_auth_error(self):
        """Test that authentication errors return None for embeddings"""
        with patch.dict('os.environ', {'OPENAI_KEY': 'invalid-key'}):
            interface = OpenAIInterface()
            result = interface._embedding_call(
                texts=["test"],
                embedding_model="text-embedding-3-small",
                size=1536,
                max_retries=1  # Reduce retries for faster test
            )

            assert result is None

    def test_openai_embedding_bad_model(self):
        """Test that invalid model returns None"""
        interface = OpenAIInterface()
        result = interface._embedding_call(
            texts=["test"],
            embedding_model="nonexistent-embedding-model",
            size=1536,
            max_retries=1
        )

        assert result is None


# ============== Gemini Embedding Tests ==============

class TestGeminiEmbeddings:
    """Tests for Gemini embeddings"""

    def test_gemini_basic_embedding(self, llm, test_texts, test_ids):
        """Test basic Gemini embedding generation"""
        _, embeddings = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=768,
            db=None,
            name="test_gemini_emb",
            verbose=False,
            model="gemini-embedding-001"
        )

        assert embeddings is not None
        assert len(embeddings) == len(test_texts)
        assert len(embeddings[0]) == 768  # Verify dimension

    def test_gemini_embedding_similarity(self, llm, test_texts, test_ids):
        """Test that semantically similar words have higher similarity (from demo.ipynb)"""
        _, embeddings = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=768,
            db=None,
            name="test_gemini_emb",
            verbose=False,
            model="gemini-embedding-001"
        )

        cat_emb = np.array(embeddings[0])
        kitty_emb = np.array(embeddings[1])
        dog_emb = np.array(embeddings[2])
        potato_emb = np.array(embeddings[3])

        sim_cat_kitty = cosine_similarity(cat_emb, kitty_emb)
        sim_dog_potato = cosine_similarity(dog_emb, potato_emb)

        # cat-kitty should be more similar than dog-potato
        assert sim_cat_kitty > sim_dog_potato, \
            f"Expected cat-kitty ({sim_cat_kitty:.4f}) > dog-potato ({sim_dog_potato:.4f})"

    def test_gemini_batch_embedding(self, llm, batch_texts, batch_ids):
        """Test batch embedding processing"""
        _, embeddings = llm.embed(
            ids=batch_ids,
            texts=batch_texts,
            size=768,
            db=None,
            name="test_gemini_batch",
            verbose=False,
            model="gemini-embedding-001"
        )

        assert embeddings is not None
        assert len(embeddings) == len(batch_texts)


class TestGeminiEmbeddingErrorHandling:
    """Tests for Gemini embedding error handling"""

    def test_gemini_embedding_bad_model(self):
        """Test that invalid model returns None"""
        interface = GeminiInterface()
        result = interface._embedding_call(
            texts=["test"],
            embedding_model="nonexistent-embedding-model",
            size=768,
            max_retries=1
        )

        assert result is None


# ============== Claude Embedding Tests ==============

class TestClaudeEmbeddings:
    """Tests for Claude embeddings (should not be supported)"""

    def test_claude_embedding_not_supported(self):
        """Test that Claude embedding returns None (not supported)"""
        from llm_utils.claude_interface import ClaudeInterface

        interface = ClaudeInterface()
        result = interface._embedding_call(
            texts=["test"],
            embedding_model="any-model",
            size=1536
        )

        assert result is None, "Claude should not support embeddings"


# ============== Wrapper Routing Tests ==============

class TestWrapperRouting:
    """Tests for LLMWrapper embedding routing"""

    def test_wrapper_routes_to_openai(self, llm, test_texts, test_ids):
        """Test that wrapper correctly routes to OpenAI for OpenAI models"""
        _, embeddings = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=1536,
            db=None,
            name="test_routing",
            verbose=False,
            model="text-embedding-3-small"
        )

        assert embeddings is not None

    def test_wrapper_routes_to_gemini(self, llm, test_texts, test_ids):
        """Test that wrapper correctly routes to Gemini for Gemini models"""
        _, embeddings = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=768,
            db=None,
            name="test_routing",
            verbose=False,
            model="gemini-embedding-001"
        )

        assert embeddings is not None


# ============== Embedding Quality Tests ==============

class TestEmbeddingQuality:
    """Tests for embedding quality and consistency"""

    def test_embedding_determinism(self, llm, test_texts, test_ids):
        """Test that same input produces same embeddings"""
        _, embeddings1 = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=1536,
            db=None,
            name="test_determinism1",
            verbose=False,
            model="text-embedding-3-small"
        )

        _, embeddings2 = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=1536,
            db=None,
            name="test_determinism2",
            verbose=False,
            model="text-embedding-3-small"
        )

        # Embeddings should be very similar (allowing for small floating point differences)
        similarity = cosine_similarity(
            np.array(embeddings1[0]),
            np.array(embeddings2[0])
        )
        assert similarity > 0.99, f"Expected high similarity, got {similarity}"

    def test_embedding_uniqueness(self, llm, test_texts, test_ids):
        """Test that different texts produce different embeddings"""
        _, embeddings = llm.embed(
            ids=test_ids,
            texts=test_texts,
            size=1536,
            db=None,
            name="test_uniqueness",
            verbose=False,
            model="text-embedding-3-small"
        )

        # cat and potato should have lower similarity
        cat_emb = np.array(embeddings[0])
        potato_emb = np.array(embeddings[3])

        similarity = cosine_similarity(cat_emb, potato_emb)
        assert similarity < 0.9, f"Different words should have lower similarity, got {similarity}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
