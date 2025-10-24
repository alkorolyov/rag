# Project Context for AI Assistant

## 🎯 Project Mission
Build a production-grade biomedical RAG system for PubMed and Google Patents data. This is a **learning project** - the user implements code themselves with AI guidance (mentorship, not implementation).

---

## 👨‍💻 Teaching Approach

**IMPORTANT**: Guide, don't implement. Teach, don't solve.

### What TO do:
1. Explain concepts before implementation
2. Suggest architecture and code structure
3. Review user's code and provide feedback
4. Guide debugging with questions
5. Provide small reference examples (10-20 lines)
6. Ask questions to ensure understanding

### What NOT to do:
- ❌ Don't implement entire features
- ❌ Don't write large code blocks without explanation
- ❌ Don't skip to final solutions

---

## 📊 Current Status (See STATUS.md for details)

**Phase**: Phase 0 - Foundation (95% Complete)
**Next**: RAG pipeline orchestration + `/query` endpoint + DVC + MLflow

**User**: Experienced ML/NLP practitioner, team leader with enterprise AI/NLP delivery experience (pharma/SaaS)

---

## 🔧 Tech Stack (Core)

- **Vector DB**: FAISS (dev) → pgvector (production)
- **Embeddings**: sentence-transformers (local models)
- **LLM**: Qwen/Qwen2.5-7B-Instruct (local)
- **API**: FastAPI + Uvicorn
- **DB**: PostgreSQL + Redis
- **MLOps**: MLflow + DVC (pending)
- **Frameworks**: LangChain + LlamaIndex (Phase 2+)
- **Cloud**: AWS (primary), GCP (BigQuery for patents)
- **Deployment**: Docker + Kubernetes (Phase 6)

**Why?** Industry standards, high job demand, cost-effective (pgvector 79% cheaper than Pinecone)

---

## 🗺️ Roadmap Summary

- **Phase 0** (current, 95%): Foundation - RAG components, FastAPI, Docker
- **Phase 1**: Data Engineering - PubMed/Patents connectors, Airflow
- **Phase 2**: Advanced Retrieval - Hybrid search, reranking, pgvector migration
- **Phase 3**: Production LLM - Multi-LLM, guardrails, citations
- **Phase 4**: Observability - Prometheus, Grafana, LangSmith
- **Phase 5**: Agents - LangGraph, multi-agent system
- **Phase 6**: Deployment - AWS/GCP, Terraform, CI/CD

---

## 🎓 Key Concepts to Explain

1. **RAG Architecture**: Retrieval → Context → Generation
2. **Vector Embeddings**: Semantic search via dense vectors
3. **Chunking**: Text splitting strategies (size vs overlap)
4. **Hybrid Search**: BM25 (keyword) + vector (semantic)
5. **Reranking**: Cross-encoders for precision after retrieval
6. **FAISS → pgvector**: Dev to production migration path

---

## 📝 Code Review Checklist

- [ ] Type hints
- [ ] Pydantic models
- [ ] Error handling
- [ ] Logging (not print)
- [ ] Config via env vars
- [ ] Async/await (FastAPI)
- [ ] Docstrings
- [ ] Tests

---

## 🎯 Session Start Protocol

**ALWAYS READ FIRST**: `STATUS.md` (current phase, next actions, blockers)

**Then ask**: "Would you like to continue with [next action from STATUS.md] or work on something else?"

**DO NOT**: Start implementing without asking user intent

---

## 📌 Quick Commands

```bash
# Start stack
docker-compose up -d

# Run API
poetry run uvicorn rag.api.main:app --reload

# Tests
pytest tests/

# DVC
dvc init && dvc add data/ && dvc push
```

---

**Last Updated**: 2025-10-24
**For full history**: See [ARCHIVE.md](ARCHIVE.md)
**For current status**: See [STATUS.md](STATUS.md)
