# Biomedical RAG System

A production-grade Retrieval-Augmented Generation (RAG) system designed for biomedical research, with initial focus on PubMed and Google Patents data. Built with industry-standard technologies and architected for evolution toward agentic AI applications.

## Project Overview

**Primary Use Cases**:
- Semantic search and question-answering over biomedical literature (PubMed)
- Patent analysis and prior art search (Google Patents)
- Cross-document synthesis and research assistance
- Future: Multi-agent workflows for complex research tasks

**Core Capabilities**:
- Advanced retrieval with hybrid search (semantic + keyword)
- LLM-powered generation with citation and guardrails
- Scalable ingestion pipelines for large datasets
- Production monitoring and observability
- Extensible architecture for agentic behaviors

---

## Tech Stack

### Core Technologies (Industry Standard)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM Framework** | LangChain + LlamaIndex | LangChain: orchestration, agents; LlamaIndex: advanced RAG retrieval |
| **Vector Database** | pgvector (PostgreSQL) + FAISS | pgvector: production; FAISS: local prototyping |
| **Embeddings** | OpenAI text-embedding-3 / sentence-transformers | Document & query vectorization |
| **API Framework** | FastAPI | REST API, async support |
| **Orchestration** | Apache Airflow | Data pipelines, ETL workflows |
| **Databases** | PostgreSQL + Redis | Metadata storage + caching |
| **Object Storage** | AWS S3 (primary) | Raw documents, data lake |
| **Cloud Platform** | AWS (primary) + GCP (Google Patents BigQuery) | Infrastructure, managed services |
| **Monitoring** | Prometheus + Grafana + LangSmith | Metrics, dashboards, LLM tracing |
| **MLOps** | MLflow + DVC | Experiment tracking, model registry, data versioning |
| **Deployment** | Docker + Kubernetes (EKS) | Containerization, orchestration |
| **CI/CD** | GitHub Actions | Automated testing, deployment |
| **IaC** | Terraform | Infrastructure as code |

### Additional Tools
- **Message Queue**: Redis / RabbitMQ / Kafka (async processing)
- **Reranking**: Cross-encoder models (ms-marco-MiniLM)
- **Guardrails**: Input/output validation, PII detection
- **Model Serving**: vLLM / AWS SageMaker (local/cloud LLM hosting)
- **Drift Detection**: Evidently AI
- **Data Versioning**: DVC (Data Version Control) - Git integration for ML datasets

### Why This Stack?

**Vector Database Choice - pgvector + FAISS**:
- **pgvector**: PostgreSQL extension offering 1.4x better latency than Pinecone at 79% lower cost (self-hosted). Leverages existing PostgreSQL skills and infrastructure.
- **FAISS**: Facebook's vector similarity search library - perfect for local development and prototyping before deploying to pgvector.
- **Migration Path**: Develop locally with FAISS → Deploy to production with pgvector

**LLM Framework - Hybrid Approach**:
- **LangChain**: Industry standard (70%+ market share) for orchestration, agents, and workflows. Best production tooling and ecosystem.
- **LlamaIndex**: Specialized for advanced RAG with 35% better retrieval accuracy (2025 benchmarks). Ideal for document-heavy biomedical applications.
- **Best Practice**: Use both - LlamaIndex for retrieval optimization, LangChain for agent orchestration.

**Cloud Strategy**:
- **AWS**: Primary cloud (34% market share, highest job demand) - S3, EKS, SageMaker, RDS
- **GCP**: Secondary for Google Patents BigQuery Public Dataset access
- Terraform enables multi-cloud flexibility

---

## Project Structure

```
rag-biomedical/
├── src/
│   ├── ingestion/              # Data connectors and preprocessing
│   │   ├── pubmed_loader.py
│   │   ├── patents_loader.py
│   │   ├── chunking.py
│   │   └── preprocessor.py
│   │
│   ├── embeddings/             # Embedding generation & caching
│   │   ├── openai_embedder.py
│   │   ├── local_embedder.py
│   │   └── cache.py
│   │
│   ├── retrieval/              # Vector search & hybrid retrieval
│   │   ├── vector_store.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   │
│   ├── generation/             # LLM integration
│   │   ├── llm_client.py
│   │   ├── prompts.py
│   │   └── guardrails.py
│   │
│   ├── agents/                 # Agentic components (Phase 5)
│   │   ├── base_agent.py
│   │   ├── retrieval_agent.py
│   │   └── synthesis_agent.py
│   │
│   ├── api/                    # FastAPI application
│   │   ├── main.py
│   │   ├── routes/
│   │   └── middleware/
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── metrics.py
│
├── airflow/
│   ├── dags/                   # Ingestion pipelines
│   │   ├── pubmed_ingestion_dag.py
│   │   └── patents_ingestion_dag.py
│   └── config/
│
├── infrastructure/
│   ├── docker/                 # Docker configurations
│   ├── k8s/                    # Kubernetes manifests
│   ├── terraform/              # Cloud infrastructure
│   └── monitoring/             # Prometheus & Grafana configs
│
├── notebooks/                  # Exploratory analysis
├── tests/                      # Unit, integration, e2e tests
├── configs/                    # YAML configurations
└── .github/workflows/          # CI/CD pipelines
```

---

## Implementation Phases

### Phase 0: Core Foundation (Week 1-2)
**Goal**: Production-ready RAG template

**Deliverables**:
- Basic RAG pipeline (embedding → retrieval → generation)
- FastAPI with `/query` endpoint
- Docker Compose local setup
- Simple evaluation framework (MLflow)

**Tech**:
- LangChain + LlamaIndex + OpenAI API / local LLM (Mistral-7B)
- FAISS (local prototype) → pgvector (production)
- FastAPI + PostgreSQL + Redis
- DVC for data versioning

---

### Phase 1: Data Engineering (Week 3-4)
**Goal**: Robust ingestion pipelines for PubMed & Google Patents

**Components**:
1. **Data Connectors**:
   - PubMed: Entrez API → XML parsing
   - Google Patents: GCP BigQuery Public Dataset (patents-public-data)
   - Standardized schema: title, abstract, claims, metadata
   - DVC for data versioning and pipeline reproducibility

2. **Airflow Pipelines**:
   - Incremental data fetching
   - Chunking strategies (recursive character splitter with overlap)
   - Metadata enrichment (MeSH terms, CPC codes)
   - Deduplication (MinHash LSH)

3. **Storage Architecture**:
   - **Raw**: AWS S3 (document storage)
   - **Structured**: PostgreSQL (metadata + relational queries)
   - **Vectors**: pgvector (PostgreSQL extension for embeddings + hybrid search)
   - **Versioning**: DVC tracks data lineage and enables reproducibility

---

### Phase 2: Advanced Retrieval (Week 5-6)
**Goal**: Move beyond naive RAG using LlamaIndex optimizations

**Techniques**:
1. **Hybrid Search**: BM25 (keyword) + Dense Vectors (semantic) via pgvector
2. **LlamaIndex Features**: Query engines, response synthesizers, citation tracking
3. **Reranking**: Cross-encoder models for precision
4. **Query Expansion**: LLM-generated alternative queries
5. **Metadata Filtering**: Date ranges, document types, MeSH terms
6. **Parent-Child Chunking**: Retrieve small chunks, return larger context

**Evaluation** (MLflow + DVC):
- Metrics: NDCG@k, MRR, Recall@k, Precision@k
- Ground truth: Curated Q&A pairs from PubMed reviews (versioned with DVC)
- Experiments: embedding models, chunking strategies, k values
- DVC tracks dataset versions and experiment reproducibility

---

### Phase 3: Production LLM Integration (Week 7-8)
**Goal**: Robust generation with guardrails

**Components**:
1. **LLM Layer**:
   - Primary: OpenAI GPT-4 (fast iteration)
   - Fallback: Self-hosted Mistral-7B (vLLM)
   - Fine-tuning: LoRA adapters on biomedical QA

2. **Prompt Engineering**:
   - System prompts for biomedical accuracy
   - Chain-of-Thought reasoning
   - Citation requirements

3. **Guardrails**:
   - Input: PII detection, prompt injection prevention
   - Output: Hallucination detection (NLI models), citation validation
   - Safety: Medical disclaimers, content filtering

4. **Quality Assurance**:
   - Self-consistency (sample multiple responses)
   - Source citation with `[doc_id:chunk_id]`

---

### Phase 4: Observability & Monitoring (Week 9)
**Goal**: Production-grade monitoring

**Stack**:
- **Metrics** (Prometheus):
  - Latency percentiles (p50, p95, p99)
  - Token usage & cost tracking
  - Retrieval quality scores

- **Dashboards** (Grafana):
  - Request volume by endpoint
  - Error rates (LLM failures, DB timeouts)
  - Embedding drift detection (Evidently AI)

- **Tracing** (LangSmith):
  - End-to-end request traces
  - Chain execution visualization
  - Token usage breakdown

- **Logging**:
  - User queries (anonymized)
  - Retrieved documents (audit trail)
  - Model outputs (human review queue)

---

### Phase 5: Agentic Capabilities (Week 10-12)
**Goal**: Transition from RAG to autonomous agents

**Architecture** (LangGraph):
```
Agent Framework:
├── Planner Agent       # Decomposes complex queries
├── Retrieval Agent     # RAG specialist
├── Analysis Agent      # Document processing (summarization, extraction)
├── Synthesis Agent     # Multi-source synthesis
└── Validation Agent    # Fact-checking
```

**Key Features**:
1. **Tool Use**: API calls (PubMed search, patent classification)
2. **Memory**:
   - Short-term: Conversation history (Redis)
   - Long-term: Vector memory (episodic retrieval)
3. **Multi-step Reasoning**: ReAct / Plan-and-Execute patterns
4. **Human-in-the-Loop**: Approval gates for critical actions

**Example Use Case**:
```
Query: "Find CRISPR gene editing patents filed after 2020 and
        cross-reference with Nature papers on off-target effects"

Agent Flow:
1. Planner: Split → patent search + literature review
2. Retrieval Agent 1: Query Google Patents (metadata filters)
3. Retrieval Agent 2: Search PubMed for Nature papers
4. Analysis Agent: Extract patent citations
5. Synthesis Agent: Find overlaps → generate report
```

---

### Phase 6: Scalability & Deployment (Week 13-14)
**Goal**: Cloud-native, auto-scaling infrastructure

**Stack**:
- **Orchestration**: Kubernetes on AWS EKS (primary) + GKE (Google Patents data access)
- **Model Serving**: vLLM (GPU auto-scaling) or AWS SageMaker
- **API Gateway**: AWS API Gateway (rate limiting, auth, throttling)
- **CI/CD**: GitHub Actions (test → build → deploy to EKS)
- **IaC**: Terraform (AWS primary, GCP secondary for BigQuery)

**Cost Optimization**:
- Redis caching for frequent queries
- Batch embedding generation
- Model routing (small models for simple queries)
- pgvector reduces vector DB costs vs managed solutions (79% cheaper than Pinecone)

---

## MVP Timeline

### Week 1-2: MVP Foundation
- Basic RAG pipeline (LangChain + LlamaIndex + local LLM + FAISS)
- FastAPI `/query` endpoint
- Docker Compose setup (PostgreSQL + Redis + FAISS)
- DVC initialization for data versioning
- Evaluation notebook (MLflow tracking)

### Month 1 Goal
Working prototype on PubMed subset (10K papers)

### Month 2 Goal
Production pipeline with Airflow + monitoring

### Month 3 Goal
Agentic capabilities + patent integration

---

## Vector Database Decision

| Factor | pgvector | FAISS |
|--------|----------|-------|
| **Job Market Demand** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Ease of Setup** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Production Ready** | ⭐⭐⭐⭐⭐ | ⭐⭐ (prototyping only) |
| **Cost (at scale)** | $ (79% cheaper than Pinecone) | Free |
| **Performance** | 1.4x better latency than Pinecone | Excellent for local dev |
| **Skills Transfer** | PostgreSQL (universal skill) | Research/ML prototyping |
| **Hybrid Search** | Native SQL + vector search | Requires additional tooling |

**Our Strategy**:
1. **Local Development**: FAISS (fast prototyping, no infrastructure)
2. **Production**: pgvector on AWS RDS PostgreSQL (cost-effective, performant, leverages PostgreSQL expertise)
3. **Migration Path**: FAISS → pgvector (straightforward, both use similar APIs via LangChain/LlamaIndex)

**Why pgvector + FAISS?**
- pgvector integrates seamlessly with existing PostgreSQL infrastructure
- 79% cost reduction vs managed solutions while maintaining performance
- FAISS perfect for rapid local iteration without cloud dependencies
- Both supported natively by LangChain and LlamaIndex

---

## Key Design Principles

1. **Industry Standards First**: Use technologies with highest job market demand
2. **Modular Architecture**: Easy to swap components (LLM, vector DB, etc.)
3. **Observability by Default**: Metrics, logging, tracing from day one
4. **Reproducibility**: MLflow tracking, versioned datasets, IaC
5. **Scalability**: Design for growth (Kubernetes, auto-scaling, caching)
6. **Extensibility**: Clear path from RAG → Agentic AI

---

## Core Dependencies

```txt
# LLM & RAG Frameworks
langchain==0.1.0
langchain-openai==0.0.5
llama-index==0.9.0              # Advanced RAG capabilities
llama-index-vector-stores-postgres==0.1.0
openai==1.10.0
tiktoken==0.5.2

# Vector Stores
pgvector==0.2.0                 # PostgreSQL extension for production
faiss-cpu==1.7.4                # Local development & prototyping

# API & Backend
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.25

# Databases
psycopg2-binary==2.9.9          # PostgreSQL driver
redis==5.0.1                    # Caching & sessions

# Data Processing & Versioning
pandas==2.2.0
numpy==1.26.3
dvc==3.0.0                      # Data version control
dvc-s3==3.0.0                   # DVC with AWS S3 backend

# Cloud SDKs
boto3==1.34.0                   # AWS SDK (S3, SageMaker, etc.)
google-cloud-bigquery==3.15.0   # GCP BigQuery for Google Patents

# Monitoring & MLOps
prometheus-client==0.19.0
mlflow==2.10.0
evidently==0.4.0                # Drift detection
sentry-sdk==1.40.0

# Orchestration
apache-airflow==2.8.1
apache-airflow-providers-postgres==5.10.0
apache-airflow-providers-amazon==8.16.0

# Testing
pytest==7.4.0
pytest-asyncio==0.21.0
```

---

## Getting Started

```bash
# Clone and setup (to be implemented)
git clone <repo-url>
cd rag-biomedical

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
dvc remote add -d myremote s3://your-bucket/dvc-storage

# Configure environment
cp .env.example .env
# Edit .env with API keys:
#   - OPENAI_API_KEY
#   - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#   - GCP_PROJECT_ID (for BigQuery)
#   - DATABASE_URL (PostgreSQL with pgvector)

# Start local stack (PostgreSQL + pgvector, Redis, FAISS)
docker-compose up -d

# Run database migrations
alembic upgrade head

# Pull versioned data (if available)
dvc pull

# Run API
uvicorn src.api.main:app --reload

# Access API docs
open http://localhost:8000/docs
```

---

## Project Status

**Current Phase**: Phase 0 - Foundation Setup (95% Complete)
**Next Milestone**: RAG Pipeline Integration + `/query` Endpoint + DVC + MLflow

**Completed Components**:
- ✅ Infrastructure (Docker Compose with PostgreSQL + pgvector + Redis)
- ✅ FastAPI application with health checks
- ✅ Embeddings module (sentence-transformers with local models)
- ✅ FAISS vector store (DocumentStore with IndexIDMap)
- ✅ LLM generation (LocalLLM with Qwen/Qwen2.5-7B-Instruct)
- ✅ Text chunking (RecursiveChunker)
- ✅ Reranking (CrossEncoderReranker)
- ✅ Evaluation framework (bioasq-mini dataset experiments)

**In Progress**:
- 🟡 RAG pipeline orchestration (integrate all components)
- 🟡 `/query` API endpoint
- 🟡 DVC for data versioning
- 🟡 MLflow for experiment tracking

**Documentation**:
- 📄 [STATUS.md](STATUS.md) - Current status (read first!)
- 📄 [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) - Tech stack & teaching approach
- 📄 [ARCHIVE.md](ARCHIVE.md) - Historical session notes
- 📄 [INSTRUCTIONS_FOR_AI.md](INSTRUCTIONS_FOR_AI.md) - AI assistant guide

---

## Future Enhancements

- [ ] Multi-modal support (images, chemical structures)
- [ ] Fine-tuned domain models (BioBERT, PubMedBERT)
- [ ] Graph-based knowledge representation
- [ ] Federated learning for privacy-sensitive data
- [ ] Real-time collaboration features
- [ ] Multi-language support

---

## Contributing

(To be added)

## License

(To be added)

---

**Last Updated**: 2025-10-24
