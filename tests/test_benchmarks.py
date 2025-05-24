"""
Performance benchmark tests for OpenClassifier.
Measures throughput, latency, memory usage, and scalability.
"""

import pytest
import asyncio
import time
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock
import numpy as np
from typing import List, Dict, Any

from open_classifier.services.classifier_service import ClassifierService
from open_classifier.models.embeddings import EmbeddingModel
from open_classifier.api.models import ClassificationRequest


class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""

    @pytest.fixture
    def classifier_service(self):
        """Create a mocked classifier service for benchmarking."""
        with patch('open_classifier.models.dspy_classifier.ClassifierModule'), \
             patch('open_classifier.models.langchain_classifier.LangChainClassifier'):
            service = ClassifierService()
            # Mock the classify method to return quickly
            service.classify = MagicMock(return_value={
                "class": "positive",
                "confidence": 0.85,
                "explanation": "Mock classification result",
                "metadata": {"model": "mock", "processing_time": 0.001}
            })
            return service

    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts for benchmarking."""
        return [
            "This is an excellent product with outstanding quality!",
            "I'm very disappointed with this purchase and service.",
            "The item is okay, nothing special but does the job.",
            "Absolutely fantastic experience, highly recommended!",
            "Poor quality, would not buy again, waste of money.",
            "Average product, meets basic expectations adequately.",
            "Exceptional value for money, exceeded all expectations!",
            "Terrible customer service, product arrived damaged.",
            "Good quality product, satisfied with the purchase.",
            "Outstanding innovation, revolutionary features included!"
        ] * 10  # 100 texts total

    @pytest.mark.benchmark
    def test_single_classification_latency(self, benchmark, classifier_service):
        """Benchmark single classification latency."""
        text = "This is a sample text for classification."
        labels = ["positive", "negative", "neutral"]
        
        def classify_single():
            return classifier_service.classify(text, labels)
        
        result = benchmark(classify_single)
        
        # Assertions
        assert result["class"] in labels
        assert 0 <= result["confidence"] <= 1
        
        # Performance assertions (adjust based on your requirements)
        assert benchmark.stats.mean < 0.5  # Average < 500ms
        assert benchmark.stats.max < 2.0   # Max < 2 seconds

    @pytest.mark.benchmark
    def test_batch_classification_throughput(self, benchmark, classifier_service, sample_texts):
        """Benchmark batch classification throughput."""
        labels = ["positive", "negative", "neutral"]
        
        def classify_batch():
            results = []
            for text in sample_texts:
                result = classifier_service.classify(text, labels)
                results.append(result)
            return results
        
        results = benchmark(classify_batch)
        
        # Assertions
        assert len(results) == len(sample_texts)
        for result in results:
            assert result["class"] in labels
        
        # Throughput assertions
        texts_per_second = len(sample_texts) / benchmark.stats.mean
        assert texts_per_second > 50  # At least 50 texts/second

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_classification_performance(self, benchmark, classifier_service):
        """Benchmark concurrent classification performance."""
        texts = ["Sample text for concurrent processing"] * 50
        labels = ["positive", "negative", "neutral"]
        
        async def classify_concurrent():
            tasks = []
            for text in texts:
                # Simulate async classification
                task = asyncio.create_task(
                    asyncio.to_thread(classifier_service.classify, text, labels)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        results = await benchmark(classify_concurrent)
        
        # Assertions
        assert len(results) == len(texts)
        
        # Concurrent performance should be better than sequential
        concurrent_time = benchmark.stats.mean
        expected_sequential_time = len(texts) * 0.1  # Assuming 100ms per classification
        assert concurrent_time < expected_sequential_time * 0.5  # At least 2x speedup

    @pytest.mark.benchmark
    def test_memory_usage_classification(self, classifier_service, sample_texts):
        """Test memory usage during classification operations."""
        labels = ["positive", "negative", "neutral"]
        
        # Measure initial memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform classifications
        for text in sample_texts:
            classifier_service.classify(text, labels)
        
        # Measure final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory assertions
        assert memory_increase < 100  # Less than 100MB increase
        assert final_memory < 1000    # Total memory under 1GB

    @pytest.mark.benchmark
    def test_embedding_generation_performance(self, benchmark):
        """Benchmark embedding generation performance."""
        with patch('open_classifier.models.embeddings.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_embeddings = np.random.rand(10, 384)
            mock_model.encode.return_value = mock_embeddings
            mock_transformer.return_value = mock_model
            
            embedding_model = EmbeddingModel(model_name="test-model")
            texts = ["Sample text for embedding"] * 10
            
            def generate_embeddings():
                return embedding_model.encode(texts)
            
            result = benchmark(generate_embeddings)
            
            # Assertions
            assert result.shape == (10, 384)
            
            # Performance assertions
            embeddings_per_second = 10 / benchmark.stats.mean
            assert embeddings_per_second > 100  # At least 100 embeddings/second

    @pytest.mark.benchmark
    def test_similarity_computation_performance(self, benchmark):
        """Benchmark similarity computation performance."""
        with patch('open_classifier.models.embeddings.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            
            # Mock embeddings for query and candidates
            query_embedding = np.random.rand(384)
            candidate_embeddings = np.random.rand(100, 384)
            
            mock_model.encode.side_effect = [query_embedding, candidate_embeddings]
            mock_transformer.return_value = mock_model
            
            embedding_model = EmbeddingModel(model_name="test-model")
            query = "Find similar texts"
            candidates = [f"Candidate text {i}" for i in range(100)]
            
            def compute_similarities():
                return embedding_model.compute_similarity(query, candidates)
            
            result = benchmark(compute_similarities)
            
            # Assertions
            assert len(result) == 100
            
            # Performance assertions
            comparisons_per_second = 100 / benchmark.stats.mean
            assert comparisons_per_second > 1000  # At least 1000 comparisons/second

    @pytest.mark.benchmark
    def test_cache_performance(self, benchmark):
        """Benchmark caching system performance."""
        from open_classifier.utils.cache import LRUCache
        
        cache = LRUCache(maxsize=1000)
        
        def cache_operations():
            # Perform mixed read/write operations
            for i in range(100):
                key = f"key_{i % 50}"  # Create some cache hits
                value = f"value_{i}"
                
                # Write operation
                cache.set(key, value)
                
                # Read operation
                cached_value = cache.get(key)
                
                if cached_value is None:
                    cache.set(key, value)
        
        benchmark(cache_operations)
        
        # Cache should be fast
        assert benchmark.stats.mean < 0.01  # Less than 10ms for 100 operations

    @pytest.mark.benchmark
    def test_json_serialization_performance(self, benchmark, sample_texts):
        """Benchmark JSON serialization/deserialization performance."""
        import json
        
        # Create complex classification results
        results = []
        for i, text in enumerate(sample_texts[:10]):
            result = {
                "text": text,
                "classification": {
                    "class": "positive",
                    "confidence": 0.85 + (i * 0.01),
                    "probabilities": {
                        "positive": 0.85 + (i * 0.01),
                        "negative": 0.10 - (i * 0.005),
                        "neutral": 0.05 - (i * 0.005)
                    }
                },
                "metadata": {
                    "model_used": "ensemble",
                    "processing_time": 0.234 + (i * 0.01),
                    "timestamp": time.time(),
                    "request_id": f"req_{i:06d}"
                }
            }
            results.append(result)
        
        def serialize_deserialize():
            # Serialize to JSON
            json_data = json.dumps(results)
            
            # Deserialize from JSON
            parsed_results = json.loads(json_data)
            
            return parsed_results
        
        parsed_results = benchmark(serialize_deserialize)
        
        # Assertions
        assert len(parsed_results) == len(results)
        assert parsed_results[0]["classification"]["class"] == "positive"

    @pytest.mark.benchmark
    def test_multithreaded_classification(self, benchmark, classifier_service, sample_texts):
        """Benchmark multithreaded classification performance."""
        labels = ["positive", "negative", "neutral"]
        
        def classify_multithreaded():
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(classifier_service.classify, text, labels)
                    for text in sample_texts[:20]  # Use smaller sample for threading test
                ]
                
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                
                return results
        
        results = benchmark(classify_multithreaded)
        
        # Assertions
        assert len(results) == 20
        
        # Threading should provide speedup
        assert benchmark.stats.mean < 2.0  # Should complete in under 2 seconds

    @pytest.mark.benchmark
    def test_stress_test_high_load(self, classifier_service):
        """Stress test with high load simulation."""
        labels = ["positive", "negative", "neutral"]
        texts = ["Stress test text"] * 1000
        
        start_time = time.time()
        successful_classifications = 0
        errors = 0
        
        for text in texts:
            try:
                result = classifier_service.classify(text, labels)
                if result and "class" in result:
                    successful_classifications += 1
                else:
                    errors += 1
            except Exception:
                errors += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert successful_classifications > 950  # 95% success rate
        assert errors < 50  # Less than 5% errors
        assert total_time < 30  # Complete in under 30 seconds
        
        throughput = successful_classifications / total_time
        assert throughput > 30  # At least 30 classifications/second

    @pytest.mark.benchmark
    def test_memory_leak_detection(self, classifier_service):
        """Test for memory leaks during repeated operations."""
        labels = ["positive", "negative", "neutral"]
        text = "Memory leak test text"
        
        # Measure initial memory
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        for _ in range(1000):
            classifier_service.classify(text, labels)
            
            # Periodic garbage collection
            if _ % 100 == 0:
                gc.collect()
        
        # Final memory measurement
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory leak assertions
        assert memory_increase < 50  # Less than 50MB increase for 1000 operations
        
        # Memory should stabilize (not grow linearly)
        memory_per_operation = memory_increase / 1000
        assert memory_per_operation < 0.05  # Less than 50KB per operation

    @pytest.mark.benchmark
    def test_cpu_utilization(self, classifier_service, sample_texts):
        """Test CPU utilization during classification operations."""
        labels = ["positive", "negative", "neutral"]
        
        # Monitor CPU usage
        cpu_percentages = []
        
        def monitor_cpu():
            for _ in range(10):  # Monitor for 10 intervals
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform classifications
        for text in sample_texts[:50]:
            classifier_service.classify(text, labels)
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # CPU utilization assertions
        avg_cpu = np.mean(cpu_percentages)
        max_cpu = np.max(cpu_percentages)
        
        assert avg_cpu < 80  # Average CPU usage under 80%
        assert max_cpu < 95  # Peak CPU usage under 95%

    @pytest.mark.benchmark
    def test_response_time_distribution(self, benchmark, classifier_service):
        """Test response time distribution and percentiles."""
        text = "Response time test text"
        labels = ["positive", "negative", "neutral"]
        
        response_times = []
        
        def measure_response_times():
            for _ in range(100):
                start_time = time.time()
                classifier_service.classify(text, labels)
                end_time = time.time()
                response_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        benchmark(measure_response_times)
        
        # Calculate percentiles
        p50 = np.percentile(response_times, 50)
        p95 = np.percentile(response_times, 95)
        p99 = np.percentile(response_times, 99)
        
        # Response time assertions
        assert p50 < 100   # 50th percentile under 100ms
        assert p95 < 500   # 95th percentile under 500ms
        assert p99 < 1000  # 99th percentile under 1 second


class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [1, 10, 50, 100, 500])
    def test_scalability_batch_sizes(self, benchmark, batch_size):
        """Test performance scaling with different batch sizes."""
        with patch('open_classifier.services.classifier_service.ClassifierService') as mock_service:
            mock_instance = MagicMock()
            mock_instance.classify.return_value = {"class": "positive", "confidence": 0.8}
            mock_service.return_value = mock_instance
            
            service = mock_service()
            texts = [f"Test text {i}" for i in range(batch_size)]
            labels = ["positive", "negative", "neutral"]
            
            def classify_batch():
                results = []
                for text in texts:
                    result = service.classify(text, labels)
                    results.append(result)
                return results
            
            results = benchmark(classify_batch)
            
            # Assertions
            assert len(results) == batch_size
            
            # Performance should scale reasonably
            time_per_item = benchmark.stats.mean / batch_size
            assert time_per_item < 0.01  # Less than 10ms per item

    @pytest.mark.benchmark
    @pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
    def test_scalability_worker_threads(self, benchmark, num_workers):
        """Test performance scaling with different numbers of worker threads."""
        with patch('open_classifier.services.classifier_service.ClassifierService') as mock_service:
            mock_instance = MagicMock()
            mock_instance.classify.return_value = {"class": "positive", "confidence": 0.8}
            mock_service.return_value = mock_instance
            
            service = mock_service()
            texts = ["Test text"] * 100
            labels = ["positive", "negative", "neutral"]
            
            def classify_with_workers():
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(service.classify, text, labels)
                        for text in texts
                    ]
                    
                    results = []
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                    
                    return results
            
            results = benchmark(classify_with_workers)
            
            # Assertions
            assert len(results) == 100
            
            # More workers should generally improve performance (up to a point)
            if num_workers > 1:
                assert benchmark.stats.mean < 5.0  # Should complete faster with multiple workers


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"]) 