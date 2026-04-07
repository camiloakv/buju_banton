# Production RAG system

<p>
<img src="https://img.shields.io/badge/LangChain-ffffff?style=flat&logo=langchain&logoColor=1b3b3a" />
<img src="https://img.shields.io/badge/ollama-000000?style=flat&logo=ollama&logoColor=white" />
</p>

A FastAPI service that ingests documents, chunks & embeds them, answers questions with citations, logs eval metrics to MLflow, and runs in Docker.

## Data sources:
- [arXiv](https://www.arxiv.org)

## Steps:
- [x] Ingestion and query with manual run:
  - Data Ingestion: `python ingest.py`
  - Query (hardcoded question): `python run.py`
- [ ] ...
