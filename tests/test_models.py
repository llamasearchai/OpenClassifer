"""
Comprehensive tests for all model components in OpenClassifier.
Tests DSPy, LangChain, embeddings, and agent implementations.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
from typing import List, Dict, Any

from open_classifier.models.dspy_classifier import ClassifierModule
from open_classifier.models.langchain_classifier import LangChainClassifier
from open_classifier.models.embeddings import EmbeddingModel, CachedEmbeddingModel
from open_classifier.models.agent import ClassificationAgent, MultiStepClassificationAgent


class TestDSPyClassifier:
    """Test suite for DSPy classifier implementation."""

    @pytest.fixture
    def sample_labels(self):
        """Sample classification labels."""
        return ["positive", "negative", "neutral"]

    @patch('open_classifier.models.dspy_classifier.dspy.OpenAI')
    @patch('open_classifier.models.dspy_classifier.dspy.configure')
    def test_classifier_initialization(self, mock_configure, mock_openai, sample_labels):
        """Test DSPy classifier initialization."""
        # Arrange
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        # Act
        classifier = ClassifierModule(labels=sample_labels)
        
        # Assert
        assert classifier.labels == sample_labels
        assert hasattr(classifier, 'classify')
        mock_configure.assert_called_once()

    @patch('open_classifier.models.dspy_classifier.dspy.OpenAI')
    @patch('open_classifier.models.dspy_classifier.dspy.configure')
    def test_classifier_forward_method(self, mock_configure, mock_openai, sample_labels):
        """Test DSPy classifier forward method."""
        # Arrange
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        mock_prediction = MagicMock()
        mock_prediction.classification = "positive"
        mock_prediction.confidence = "0.85"
        mock_prediction.explanation = "Strong positive sentiment"
        
        classifier = ClassifierModule(labels=sample_labels)
        classifier.classify = MagicMock(return_value=mock_prediction)
        
        # Act
        result = classifier.forward("This is an amazing product!")
        
        # Assert
        assert result["class"] == "positive"
        assert result["confidence"] == 0.85
        assert result["explanation"] == "Strong positive sentiment"
        assert "metadata" in result
        assert result["metadata"]["model"] == "dspy"

    @patch('open_classifier.models.dspy_classifier.dspy.OpenAI')
    @patch('open_classifier.models.dspy_classifier.dspy.configure')
    def test_classifier_invalid_confidence(self, mock_configure, mock_openai, sample_labels):
        """Test DSPy classifier with invalid confidence score."""
        # Arrange
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        mock_prediction = MagicMock()
        mock_prediction.classification = "positive"
        mock_prediction.confidence = "invalid"
        mock_prediction.explanation = "Test explanation"
        
        classifier = ClassifierModule(labels=sample_labels)
        classifier.classify = MagicMock(return_value=mock_prediction)
        
        # Act
        result = classifier.forward("Test text")
        
        # Assert
        assert result["confidence"] == 0.5  # Default fallback

    @patch('open_classifier.models.dspy_classifier.dspy.OpenAI')
    @patch('open_classifier.models.dspy_classifier.dspy.configure')
    def test_classifier_exception_handling(self, mock_configure, mock_openai, sample_labels):
        """Test DSPy classifier exception handling."""
        # Arrange
        mock_llm = MagicMock()
        mock_openai.return_value = mock_llm
        
        classifier = ClassifierModule(labels=sample_labels)
        classifier.classify = MagicMock(side_effect=Exception("API Error"))
        
        # Act & Assert
        with pytest.raises(Exception):
            classifier.forward("Test text")


class TestLangChainClassifier:
    """Test suite for LangChain classifier implementation."""

    @pytest.fixture
    def sample_labels(self):
        """Sample classification labels."""
        return ["positive", "negative", "neutral"]

    @patch('open_classifier.models.langchain_classifier.ChatOpenAI')
    def test_langchain_classifier_initialization(self, mock_chat_openai, sample_labels):
        """Test LangChain classifier initialization."""
        # Arrange
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Act
        classifier = LangChainClassifier(labels=sample_labels)
        
        # Assert
        assert classifier.labels == sample_labels
        assert hasattr(classifier, 'chain')
        mock_chat_openai.assert_called_once()

    @patch('open_classifier.models.langchain_classifier.ChatOpenAI')
    def test_langchain_classifier_classify(self, mock_chat_openai, sample_labels):
        """Test LangChain classifier classification."""
        # Arrange
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = '{"classification": "positive", "confidence": 0.9, "explanation": "Very positive sentiment"}'
        
        classifier = LangChainClassifier(labels=sample_labels)
        classifier.chain = MagicMock()
        classifier.chain.invoke = MagicMock(return_value=mock_response)
        
        # Act
        result = classifier.classify("This is fantastic!")
        
        # Assert
        assert result["class"] == "positive"
        assert result["confidence"] == 0.9
        assert result["explanation"] == "Very positive sentiment"
        assert result["metadata"]["model"] == "langchain"

    @patch('open_classifier.models.langchain_classifier.ChatOpenAI')
    def test_langchain_classifier_invalid_json(self, mock_chat_openai, sample_labels):
        """Test LangChain classifier with invalid JSON response."""
        # Arrange
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = "Invalid JSON response"
        
        classifier = LangChainClassifier(labels=sample_labels)
        classifier.chain = MagicMock()
        classifier.chain.invoke = MagicMock(return_value=mock_response)
        
        # Act
        result = classifier.classify("Test text")
        
        # Assert
        assert result["class"] == sample_labels[0]  # Fallback to first label
        assert result["confidence"] == 0.5
        assert "error" in result["explanation"].lower()

    @patch('open_classifier.models.langchain_classifier.ChatOpenAI')
    async def test_langchain_classifier_async_classify(self, mock_chat_openai, sample_labels):
        """Test LangChain classifier async classification."""
        # Arrange
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_response = MagicMock()
        mock_response.content = '{"classification": "negative", "confidence": 0.8, "explanation": "Negative sentiment"}'
        
        classifier = LangChainClassifier(labels=sample_labels)
        classifier.chain = MagicMock()
        classifier.chain.ainvoke = AsyncMock(return_value=mock_response)
        
        # Act
        result = await classifier.aclassify("This is terrible!")
        
        # Assert
        assert result["class"] == "negative"
        assert result["confidence"] == 0.8
        assert result["explanation"] == "Negative sentiment"


class TestEmbeddingModel:
    """Test suite for embedding model implementations."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for embedding."""
        return ["Hello world", "How are you?", "Machine learning is great"]

    @patch('open_classifier.models.embeddings.SentenceTransformer')
    def test_embedding_model_initialization(self, mock_transformer):
        """Test embedding model initialization."""
        # Arrange
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Act
        embedding_model = EmbeddingModel(model_name="test-model")
        
        # Assert
        assert embedding_model.model_name == "test-model"
        mock_transformer.assert_called_once_with("test-model")

    @patch('open_classifier.models.embeddings.SentenceTransformer')
    def test_embedding_model_encode(self, mock_transformer, sample_texts):
        """Test embedding model encoding."""
        # Arrange
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(3, 384)  # 3 texts, 384 dimensions
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        embedding_model = EmbeddingModel(model_name="test-model")
        
        # Act
        embeddings = embedding_model.encode(sample_texts)
        
        # Assert
        assert embeddings.shape == (3, 384)
        mock_model.encode.assert_called_once_with(sample_texts, convert_to_numpy=True)

    @patch('open_classifier.models.embeddings.SentenceTransformer')
    def test_embedding_model_similarity(self, mock_transformer):
        """Test embedding model similarity computation."""
        # Arrange
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        # Mock embeddings
        query_embedding = np.array([1, 0, 0, 0])
        candidate_embeddings = np.array([
            [1, 0, 0, 0],  # Perfect match
            [0, 1, 0, 0],  # Orthogonal
            [0.7, 0.7, 0, 0]  # Partial match
        ])
        
        mock_model.encode.side_effect = [
            query_embedding,
            candidate_embeddings
        ]
        
        embedding_model = EmbeddingModel(model_name="test-model")
        
        # Act
        similarities = embedding_model.compute_similarity(
            "query text",
            ["candidate1", "candidate2", "candidate3"]
        )
        
        # Assert
        assert len(similarities) == 3
        assert similarities[0]["similarity"] == pytest.approx(1.0, abs=0.01)  # Perfect match
        assert similarities[1]["similarity"] == pytest.approx(0.0, abs=0.01)  # Orthogonal
        assert similarities[0]["similarity"] > similarities[2]["similarity"] > similarities[1]["similarity"]

    @patch('open_classifier.models.embeddings.SentenceTransformer')
    @patch('open_classifier.models.embeddings.redis.Redis')
    def test_cached_embedding_model(self, mock_redis, mock_transformer, sample_texts):
        """Test cached embedding model functionality."""
        # Arrange
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None  # Cache miss
        
        cached_model = CachedEmbeddingModel(model_name="test-model", redis_url="redis://localhost")
        
        # Act
        embeddings = cached_model.encode(sample_texts)
        
        # Assert
        assert embeddings.shape == (3, 384)
        # Verify cache operations
        assert mock_redis_client.get.call_count == 3  # One per text
        assert mock_redis_client.setex.call_count == 3  # One per text

    @patch('open_classifier.models.embeddings.SentenceTransformer')
    @patch('open_classifier.models.embeddings.redis.Redis')
    def test_cached_embedding_model_cache_hit(self, mock_redis, mock_transformer):
        """Test cached embedding model with cache hit."""
        # Arrange
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        # Simulate cache hit
        cached_embedding = np.random.rand(384).astype(np.float32).tobytes()
        mock_redis_client.get.return_value = cached_embedding
        
        cached_model = CachedEmbeddingModel(model_name="test-model", redis_url="redis://localhost")
        
        # Act
        embeddings = cached_model.encode(["test text"])
        
        # Assert
        assert embeddings.shape == (1, 384)
        mock_model.encode.assert_not_called()  # Should not call model if cache hit


class TestClassificationAgent:
    """Test suite for classification agent implementations."""

    @pytest.fixture
    def sample_labels(self):
        """Sample classification labels."""
        return ["positive", "negative", "neutral"]

    @patch('open_classifier.models.agent.ClassifierModule')
    @patch('open_classifier.models.agent.LangChainClassifier')
    def test_classification_agent_initialization(self, mock_langchain, mock_dspy, sample_labels):
        """Test classification agent initialization."""
        # Arrange
        mock_dspy_instance = MagicMock()
        mock_dspy.return_value = mock_dspy_instance
        mock_langchain_instance = MagicMock()
        mock_langchain.return_value = mock_langchain_instance
        
        # Act
        agent = ClassificationAgent(labels=sample_labels)
        
        # Assert
        assert agent.labels == sample_labels
        assert hasattr(agent, 'dspy_classifier')
        assert hasattr(agent, 'langchain_classifier')

    @patch('open_classifier.models.agent.ClassifierModule')
    @patch('open_classifier.models.agent.LangChainClassifier')
    def test_classification_agent_classify_single(self, mock_langchain, mock_dspy, sample_labels):
        """Test classification agent single classification."""
        # Arrange
        mock_dspy_instance = MagicMock()
        mock_dspy_instance.forward.return_value = {
            "class": "positive", "confidence": 0.8, "explanation": "DSPy result"
        }
        mock_dspy.return_value = mock_dspy_instance
        
        mock_langchain_instance = MagicMock()
        mock_langchain_instance.classify.return_value = {
            "class": "positive", "confidence": 0.9, "explanation": "LangChain result"
        }
        mock_langchain.return_value = mock_langchain_instance
        
        agent = ClassificationAgent(labels=sample_labels)
        
        # Act
        result = agent.classify("This is great!", use_ensemble=True)
        
        # Assert
        assert result["class"] == "positive"
        assert 0.8 <= result["confidence"] <= 0.9  # Ensemble average
        assert "ensemble" in result["metadata"]["strategy"]

    @patch('open_classifier.models.agent.ClassifierModule')
    @patch('open_classifier.models.agent.LangChainClassifier')
    def test_classification_agent_classify_disagreement(self, mock_langchain, mock_dspy, sample_labels):
        """Test classification agent with model disagreement."""
        # Arrange
        mock_dspy_instance = MagicMock()
        mock_dspy_instance.forward.return_value = {
            "class": "positive", "confidence": 0.8, "explanation": "DSPy positive"
        }
        mock_dspy.return_value = mock_dspy_instance
        
        mock_langchain_instance = MagicMock()
        mock_langchain_instance.classify.return_value = {
            "class": "negative", "confidence": 0.7, "explanation": "LangChain negative"
        }
        mock_langchain.return_value = mock_langchain_instance
        
        agent = ClassificationAgent(labels=sample_labels)
        
        # Act
        result = agent.classify("Ambiguous text", use_ensemble=True)
        
        # Assert
        # Should choose higher confidence result or use weighted voting
        assert result["class"] in ["positive", "negative"]
        assert "disagreement" in result["metadata"]["notes"].lower()

    @pytest.mark.asyncio
    @patch('open_classifier.models.agent.ClassifierModule')
    @patch('open_classifier.models.agent.LangChainClassifier')
    async def test_classification_agent_async_classify(self, mock_langchain, mock_dspy, sample_labels):
        """Test classification agent async classification."""
        # Arrange
        mock_dspy_instance = MagicMock()
        mock_dspy_instance.forward.return_value = {
            "class": "neutral", "confidence": 0.75, "explanation": "Neutral sentiment"
        }
        mock_dspy.return_value = mock_dspy_instance
        
        mock_langchain_instance = MagicMock()
        mock_langchain_instance.aclassify = AsyncMock(return_value={
            "class": "neutral", "confidence": 0.8, "explanation": "Neutral assessment"
        })
        mock_langchain.return_value = mock_langchain_instance
        
        agent = ClassificationAgent(labels=sample_labels)
        
        # Act
        result = await agent.aclassify("This is okay", use_ensemble=True)
        
        # Assert
        assert result["class"] == "neutral"
        assert result["confidence"] > 0.75


class TestMultiStepClassificationAgent:
    """Test suite for multi-step classification agent."""

    @pytest.fixture
    def complex_labels(self):
        """Complex hierarchical labels."""
        return {
            "sentiment": ["positive", "negative", "neutral"],
            "topic": ["technology", "sports", "politics", "entertainment"],
            "urgency": ["high", "medium", "low"]
        }

    @patch('open_classifier.models.agent.ClassificationAgent')
    def test_multistep_agent_initialization(self, mock_agent, complex_labels):
        """Test multi-step agent initialization."""
        # Arrange
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        # Act
        agent = MultiStepClassificationAgent(label_hierarchy=complex_labels)
        
        # Assert
        assert agent.label_hierarchy == complex_labels
        assert len(agent.step_agents) == 3  # One per classification dimension

    @patch('open_classifier.models.agent.ClassificationAgent')
    def test_multistep_agent_classify(self, mock_agent, complex_labels):
        """Test multi-step agent classification."""
        # Arrange
        mock_agent_instance = MagicMock()
        mock_agent_instance.classify.side_effect = [
            {"class": "positive", "confidence": 0.9, "explanation": "Very positive"},
            {"class": "technology", "confidence": 0.85, "explanation": "Tech topic"},
            {"class": "low", "confidence": 0.7, "explanation": "Low urgency"}
        ]
        mock_agent.return_value = mock_agent_instance
        
        agent = MultiStepClassificationAgent(label_hierarchy=complex_labels)
        
        # Act
        result = agent.classify("New smartphone features are amazing!")
        
        # Assert
        assert result["sentiment"]["class"] == "positive"
        assert result["topic"]["class"] == "technology"
        assert result["urgency"]["class"] == "low"
        assert "multi_step" in result["metadata"]["strategy"]

    @pytest.mark.asyncio
    @patch('open_classifier.models.agent.ClassificationAgent')
    async def test_multistep_agent_async_classify(self, mock_agent, complex_labels):
        """Test multi-step agent async classification."""
        # Arrange
        mock_agent_instance = MagicMock()
        mock_agent_instance.aclassify = AsyncMock()
        mock_agent_instance.aclassify.side_effect = [
            {"class": "negative", "confidence": 0.8, "explanation": "Negative sentiment"},
            {"class": "sports", "confidence": 0.9, "explanation": "Sports content"},
            {"class": "high", "confidence": 0.75, "explanation": "High urgency"}
        ]
        mock_agent.return_value = mock_agent_instance
        
        agent = MultiStepClassificationAgent(label_hierarchy=complex_labels)
        
        # Act
        result = await agent.aclassify("Breaking: Team loses championship!")
        
        # Assert
        assert result["sentiment"]["class"] == "negative"
        assert result["topic"]["class"] == "sports"
        assert result["urgency"]["class"] == "high"


class TestModelIntegration:
    """Integration tests for model interactions."""

    @patch('open_classifier.models.dspy_classifier.dspy.OpenAI')
    @patch('open_classifier.models.dspy_classifier.dspy.configure')
    @patch('open_classifier.models.langchain_classifier.ChatOpenAI')
    @patch('open_classifier.models.embeddings.SentenceTransformer')
    def test_model_ensemble_integration(self, mock_transformer, mock_chat_openai, mock_configure, mock_openai):
        """Test integration between different model types."""
        # Arrange
        labels = ["positive", "negative", "neutral"]
        
        # Mock DSPy
        mock_dspy_llm = MagicMock()
        mock_openai.return_value = mock_dspy_llm
        dspy_classifier = ClassifierModule(labels=labels)
        dspy_classifier.classify = MagicMock(return_value=MagicMock(
            classification="positive", confidence="0.8", explanation="DSPy positive"
        ))
        
        # Mock LangChain
        mock_langchain_llm = MagicMock()
        mock_chat_openai.return_value = mock_langchain_llm
        langchain_classifier = LangChainClassifier(labels=labels)
        langchain_classifier.chain = MagicMock()
        langchain_classifier.chain.invoke = MagicMock(return_value=MagicMock(
            content='{"classification": "positive", "confidence": 0.9, "explanation": "LangChain positive"}'
        ))
        
        # Mock Embeddings
        mock_embedding_model = MagicMock()
        mock_transformer.return_value = mock_embedding_model
        embedding_model = EmbeddingModel(model_name="test-model")
        
        # Act
        dspy_result = dspy_classifier.forward("Great product!")
        langchain_result = langchain_classifier.classify("Great product!")
        embeddings = embedding_model.encode(["Great product!"])
        
        # Assert
        assert dspy_result["class"] == "positive"
        assert langchain_result["class"] == "positive"
        assert embeddings is not None

    @pytest.mark.benchmark
    @patch('open_classifier.models.embeddings.SentenceTransformer')
    def test_embedding_performance(self, mock_transformer, benchmark):
        """Benchmark embedding model performance."""
        # Arrange
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(100, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        embedding_model = EmbeddingModel(model_name="test-model")
        texts = ["Sample text"] * 100
        
        # Act & Assert
        result = benchmark(embedding_model.encode, texts)
        assert result.shape == (100, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 