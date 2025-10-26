# Project Status

**Last Updated**: 2025-10-26
**Phase**: Phase 1 - Data Engineering
**Completion**: Phase 0 Complete (100%), Phase 1 (90%)
**Next Milestone**: Unit Tests

---

## ✅ Completed Components

### Infrastructure & API
- Docker Compose (PostgreSQL + pgvector + Redis + MLflow)
- FastAPI application with health checks
- Config management (pydantic-settings)
- Logging setup

### Core RAG Modules
- **Embeddings**: BaseEmbedder, LocalEmbedder (sentence-transformers)
  - Centralized dimension configuration via settings
- **Caching**: SemanticCache (langchain-redis), EmbeddingCache
- **Storage**:
  - BaseDocumentStore: InMemoryDocumentStore, PostgresDocumentStore ✅
  - BaseVectorStore: FAISSVectorStore, PgvectorVectorStore ✅
  - Document model with metadata support
  - SQLAlchemy models (DocumentModel, ChunkModel) with pgvector support
  - Clean API: add_documents(), add_chunks(), get_chunk(), get_chunks(), iter_chunks()
- **LLM**: BaseLLM, LocalLLM (Qwen/Qwen2.5-7B-Instruct)
- **Chunking**: BaseChunker, RecursiveChunker (token-aware)
- **Reranking**: BaseReranker, CrossEncoderReranker
- **Pipeline**: RAGPipeline (retrieve → rerank → generate)

### Database & Migrations
- **Alembic**: Configured for database migrations
  - SQLAlchemy models for documents and chunks (JSONB metadata)
  - pgvector integration for embeddings (384 dimensions)
  - Automatic schema versioning with pgvector support
  - Initial migration created and applied ✅

### API Endpoints
- **`GET /health`** - Health check endpoint
- **`POST /embed`** - Text embedding endpoint
- **`POST /query`** - RAG query endpoint (full pipeline)

### Evaluation & Metrics
- **Traditional IR Metrics**: evaluate_retrieval() in utils.py
  - Precision@k, Recall@k, MRR@k, NDCG@k, Hit@k
  - Proper deduplication for accurate metrics
- **RAGAS Framework**: 12_ragas_metrics.ipynb ✅
  - LLM-as-judge metrics (faithfulness, answer_relevancy, answer_correctness)
  - OpenAI GPT-4o-mini with parallel execution (8 workers)
  - Redis caching for LLM calls (7-day TTL)
  - Random sampling with seed for reproducibility
- **Benchmark Results** (100 random samples, deduplicated):
  - IR: P@10=0.346, R@10=0.341, MRR@10=0.612, Hit@10=0.97
  - RAGAS: faithfulness=0.89, answer_relevancy=0.87, answer_correctness=0.62
- **Notebooks**: bioasq-mini dataset experiments
  - Embedder comparisons (04_compare_embedders.ipynb, 08_reranker.ipynb)
  - Chunking strategies, reranker testing
  - Full pipeline evaluation (09_full_pipeline.ipynb)
  - MLflow integration (14_mlflow.ipynb)

### MLOps & Reproducibility
- **DVC (Data Version Control)** ✅
  - Initialized for data versioning
  - Local remote: /tmp/dvc-storage
  - Tracked datasets: rag-mini-bioasq (docs + queries)
  - .dvc files in Git for reproducibility
- **MLflow (Experiment Tracking)** ✅
  - Docker service on http://localhost:5000
  - File-based backend (simple setup)
  - Named volume: mlflow_data
  - ExperimentTracker utility (src/rag/tracking.py)
    - Auto-logs git commit hash
    - Auto-logs DVC data hashes
    - Metric name sanitization (@ → _at_)
  - Experiment naming: {component}-{task}-{dataset}
  - Run naming: {model}-{params}
  - Tags: experiment_type, phase, dataset, components

---

## 🎯 Next Actions

1. **PostgreSQL Integration** - ✅ COMPLETED!
2. **Redis Caching** - ✅ COMPLETED!
3. **RAGAS Evaluation** - ✅ COMPLETED!
   - ✅ Traditional IR metrics (evaluate_retrieval in utils.py)
   - ✅ LLM-as-judge metrics (faithfulness, relevancy, correctness)
   - ✅ Redis caching for LLM calls
   - ✅ Deduplication fixes in evaluation notebooks
4. **DVC Setup** - ✅ COMPLETED!
   - ✅ DVC initialized with local remote
   - ✅ Datasets tracked (bioasq-mini docs + queries)
   - ✅ .dvc files committed to Git
5. **MLflow Setup** - ✅ COMPLETED!
   - ✅ MLflow server running in Docker
   - ✅ ExperimentTracker utility with Git + DVC integration
   - ✅ Experiment naming patterns established
   - ✅ 14_mlflow.ipynb demonstrates full workflow
6. **Unit Tests** - Core module testing (embeddings, storage, retrieval)

---

## 🐛 Current Blockers

None

---

## 📊 Phase Overview

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Foundation | ✅ Complete | 100% |
| Phase 1: Data Engineering | 🟡 In Progress | 90% |
| Phase 2: Advanced Retrieval | ⏳ Pending | 0% |
| Phase 3: Production LLM | ⏳ Pending | 0% |
| Phase 4: Observability | ⏳ Pending | 0% |
| Phase 5: Agents | ⏳ Pending | 0% |
| Phase 6: Deployment | ⏳ Pending | 0% |

---

## 🔑 Key Tech Stack

- **Embeddings**: sentence-transformers (BAAI/bge-small-en-v1.5)
- **Vector DB**: FAISS (dev) → pgvector (production)
- **LLM**: Qwen/Qwen2.5-7B-Instruct (local)
- **API**: FastAPI + Uvicorn
- **DB**: PostgreSQL + Redis
- **Evaluation**: RAGAS (LLM-as-judge) + Traditional IR metrics
- **MLOps**: MLflow (Docker) + DVC (local/network remote) ✅
- **Deployment**: Docker + Kubernetes (future)

---

**For detailed history**: See [ARCHIVE.md](ARCHIVE.md)
**For teaching context**: See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
**For AI instructions**: See [INSTRUCTIONS_FOR_AI.md](INSTRUCTIONS_FOR_AI.md)
