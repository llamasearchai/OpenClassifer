# OpenClassifier

A production-ready, high-performance text classification system built with FastAPI, DSPy, and LangChain. OpenClassifier provides enterprise-grade text classification capabilities with advanced features like ensemble learning, intelligent caching, comprehensive monitoring, and scalable deployment options.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Monitoring    │
│     (NGINX)     │────│    (FastAPI)    │────│  (Prometheus)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │                 │
                ┌──────▼──────┐   ┌──────▼──────┐
                │    DSPy     │   │  LangChain  │
                │ Classifier  │   │ Classifier  │
                └─────────────┘   └─────────────┘
                       │                 │
                       └────────┬────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Ensemble Voting     │
                    │   & Result Fusion     │
                    └───────────────────────┘
```

## Core Capabilities

### Advanced Classification Engine
- **Multi-Framework Integration**: Seamlessly combines DSPy and LangChain for robust classification
- **Ensemble Learning**: Intelligent voting mechanisms across multiple models
- **Custom Label Support**: Dynamic label creation and management
- **Confidence Scoring**: Detailed probability distributions for all predictions
- **Batch Processing**: Efficient handling of large-scale classification tasks

### Performance & Scalability
- **High-Performance Caching**: Multi-tier caching with LRU and Redis support
- **Asynchronous Processing**: Non-blocking operations for maximum throughput
- **Load Balancing**: NGINX-based distribution across multiple instances
- **Memory Optimization**: Intelligent resource management and cleanup
- **Horizontal Scaling**: Container-ready architecture for cloud deployment

### Production Features
- **Comprehensive Monitoring**: Prometheus metrics with Grafana dashboards
- **Security**: Rate limiting, authentication, and input validation
- **Error Handling**: Graceful degradation and detailed error reporting
- **Logging**: Structured logging with multiple output formats
- **Health Checks**: Automated system health monitoring

## Quick Start

### Prerequisites
- Python 3.9+
- Poetry (recommended) or pip
- Docker & Docker Compose (for containerized deployment)
- Redis (optional, for distributed caching)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/llamasearchai/OpenClassifer.git
cd OpenClassifer
```

2. **Install dependencies**:
```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

3. **Set up environment**:
```bash
cp env.example .env
# Edit .env with your configuration
```

4. **Verify installation**:
```bash
python verify_installation.py
```

5. **Start the development server**:
```bash
# Using Poetry
poetry run uvicorn open_classifier.main:app --reload

# Or directly
uvicorn open_classifier.main:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Usage Examples

### Basic Classification

```python
import requests

# Single text classification
response = requests.post("http://localhost:8000/api/classify", json={
    "text": "This movie was absolutely fantastic!",
    "labels": ["positive", "negative", "neutral"]
})

result = response.json()
print(f"Classification: {result['class']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Batch Processing

```python
# Batch classification
texts = [
    "Great product, highly recommended!",
    "Poor quality, waste of money.",
    "Average product, nothing special."
]

response = requests.post("http://localhost:8000/api/classify/batch", json={
    "texts": texts,
    "labels": ["positive", "negative", "neutral"]
})

results = response.json()
for i, result in enumerate(results):
    print(f"Text {i+1}: {result['class']} ({result['confidence']:.2f})")
```

### Custom Labels

```python
# Classification with custom labels
response = requests.post("http://localhost:8000/api/classify", json={
    "text": "The weather is sunny and warm today.",
    "labels": ["weather", "sports", "technology", "food"]
})
```

### Similarity Search

```python
# Find similar texts
response = requests.post("http://localhost:8000/api/similarity/search", json={
    "query": "machine learning algorithms",
    "texts": [
        "Deep learning neural networks",
        "Cooking recipes for dinner",
        "Artificial intelligence research",
        "Travel destinations in Europe"
    ],
    "top_k": 2
})

similar_texts = response.json()
```

## Advanced Usage

### Ensemble Configuration

```python
# Configure ensemble voting
response = requests.post("http://localhost:8000/api/classify", json={
    "text": "Sample text for classification",
    "labels": ["label1", "label2"],
    "mode": "ensemble",  # Options: dspy, langchain, ensemble
    "ensemble_strategy": "weighted_voting"  # Options: majority, weighted_voting, confidence_based
})
```

### Performance Monitoring

```python
# Get system metrics
response = requests.get("http://localhost:8000/api/metrics")
metrics = response.json()

print(f"Total classifications: {metrics['total_classifications']}")
print(f"Average latency: {metrics['avg_latency_ms']}ms")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

### WebSocket Streaming

```python
import asyncio
import websockets
import json

async def stream_classifications():
    uri = "ws://localhost:8000/ws/classify"
    async with websockets.connect(uri) as websocket:
        # Send classification request
        await websocket.send(json.dumps({
            "text": "Streaming classification example",
            "labels": ["positive", "negative"]
        }))
        
        # Receive result
        result = await websocket.recv()
        print(json.loads(result))

asyncio.run(stream_classifications())
```

## Production Deployment

### Docker Deployment

1. **Build and run with Docker Compose**:
```bash
docker-compose up -d
```

This starts:
- OpenClassifier API server
- Redis cache
- NGINX load balancer
- Prometheus monitoring
- Grafana dashboards
- ELK stack for logging

2. **Scale the application**:
```bash
docker-compose up -d --scale openclassifier=3
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=openclassifier
```

### Manual Deployment

```bash
# Use the deployment script
./scripts/deploy.sh --environment production --replicas 3
```

## API Reference

### Classification Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Single text classification |
| `/api/classify/batch` | POST | Batch text classification |
| `/api/classify/stream` | POST | Streaming classification |

### Similarity Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/similarity/search` | POST | Find similar texts |
| `/api/similarity/compare` | POST | Compare text similarity |

### Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/list` | GET | List available models |
| `/api/models/load` | POST | Load specific model |
| `/api/cache/clear` | POST | Clear system cache |
| `/api/metrics` | GET | System metrics |

### WebSocket Endpoints

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/ws/classify` | WebSocket | Real-time classification |
| `/ws/metrics` | WebSocket | Live metrics stream |

## Performance Benchmarks

### Latency Performance
- **Single Classification**: < 100ms (95th percentile)
- **Batch Processing**: < 50ms per item (1000 items)
- **Cache Hit**: < 5ms response time

### Throughput Capacity
- **Concurrent Requests**: 1000+ requests/second
- **Batch Size**: Up to 10,000 items per request
- **Memory Usage**: < 2GB for standard deployment

### Scalability Metrics
- **Horizontal Scaling**: Linear performance improvement
- **Load Balancing**: Automatic failover and recovery
- **Resource Efficiency**: 70%+ CPU utilization under load

## Security Features

### Authentication & Authorization
- API key-based authentication
- Role-based access control
- Request signing and validation

### Rate Limiting
- Per-IP rate limiting
- API key-based quotas
- Burst protection

### Input Validation
- Schema validation for all inputs
- SQL injection prevention
- XSS protection

### Security Headers
- CORS configuration
- Security headers (HSTS, CSP, etc.)
- Request/response sanitization

## Monitoring & Observability

### Metrics Collection
- **Application Metrics**: Request latency, throughput, error rates
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Classification accuracy, model performance

### Alerting
- **Performance Alerts**: High latency, low throughput
- **Error Alerts**: Classification failures, system errors
- **Capacity Alerts**: Resource utilization thresholds

### Dashboards
- **Operational Dashboard**: System health and performance
- **Business Dashboard**: Classification metrics and trends
- **Debug Dashboard**: Error analysis and troubleshooting

## Configuration

### Environment Variables

```bash
# Core Settings
APP_NAME=OpenClassifier
DEBUG=false
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Settings
DEFAULT_MODEL=ensemble
ENABLE_CACHING=true
CACHE_TTL=3600

# Performance Tuning
MAX_BATCH_SIZE=1000
REQUEST_TIMEOUT=30
WORKER_CONNECTIONS=1000

# Security
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=100
API_KEY_REQUIRED=false

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30
```

### Model Configuration

```yaml
# config/models.yaml
models:
  dspy:
    enabled: true
    model_name: "gpt-3.5-turbo"
    temperature: 0.1
    max_tokens: 150
    
  langchain:
    enabled: true
    model_name: "gpt-3.5-turbo"
    temperature: 0.1
    
ensemble:
  strategy: "weighted_voting"
  weights:
    dspy: 0.6
    langchain: 0.4
```

## Development

### Setting Up Development Environment

```bash
# Clone and setup
git clone https://github.com/llamasearchai/OpenClassifer.git
cd OpenClassifer

# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with hot reload
uvicorn open_classifier.main:app --reload
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=open_classifier

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run integration tests only
pytest -m benchmark   # Run benchmark tests only
```

### Code Quality

```bash
# Format code
black open_classifier/
isort open_classifier/

# Type checking
mypy open_classifier/

# Linting
flake8 open_classifier/

# Security scanning
bandit -r open_classifier/
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all dependencies are installed
poetry install
# Or
pip install -r requirements.txt
```

**Performance Issues**
```bash
# Check system resources
docker stats
# Monitor application metrics
curl http://localhost:8000/api/metrics
```

**Cache Issues**
```bash
# Clear application cache
curl -X POST http://localhost:8000/api/cache/clear
# Restart Redis
docker-compose restart redis
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with debug server
uvicorn open_classifier.main:app --reload --log-level debug
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/api/status

# Component health
curl http://localhost:8000/api/health/detailed
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive docstrings
- Maintain test coverage above 90%
- Update documentation for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://github.com/llamasearchai/OpenClassifer/wiki](https://github.com/llamasearchai/OpenClassifer/wiki)
- **Issues**: [https://github.com/llamasearchai/OpenClassifer/issues](https://github.com/llamasearchai/OpenClassifer/issues)
- **Discussions**: [https://github.com/llamasearchai/OpenClassifer/discussions](https://github.com/llamasearchai/OpenClassifer/discussions)

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [DSPy](https://github.com/stanfordnlp/dspy) and [LangChain](https://github.com/langchain-ai/langchain)
- Monitoring with [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)
- Containerized with [Docker](https://www.docker.com/)

---

**OpenClassifier** - Production-ready text classification at scale.