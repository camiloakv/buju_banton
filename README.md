# Buju Banton - Production RAG system

<p align="right"><i>
Raggamuffin music (or simply ragga) is a subgenre of dancehall and reggae music. The instrumentals primarily consist of electronic music with heavy use of sampling.<br>
<br>
en.wikipedia.org/wiki/Ragga
</i></p>

<p>
<img src="https://img.shields.io/badge/LangChain-ffffff?style=flat&logo=langchain&logoColor=1b3b3a" />
<img src="https://img.shields.io/badge/ollama-000000?style=flat&logo=ollama&logoColor=white" />
<img src="https://img.shields.io/badge/MLflow-404D59?style=flat&logo=mlflow" />
<img src="https://img.shields.io/badge/FastAPI-404D59?style=flat&logo=fastapi" />
</p>

<!--
<img src="https://img.shields.io/badge/docker-404D59?style=flat&logo=docker" />
-->

A FastAPI service that ingests documents, chunks & embeds them, answers questions with citations, logs eval metrics to MLflow, and runs in Docker.

## Data sources:
- [arXiv](https://www.arxiv.org)

## Steps:
- [x] Ingestion and query with manual run:
  - Data Ingestion: `python ingest.py`
  - Query (hardcoded question): `python run.py`
- [x] Evaluation through RAGAs and MLflow:
  - Run evaluation: `python evaluate.py`
  - Follow results in MLflow: `mlflow ui`
- [x] Serving with FastAPI:
  - Start the API: `uvicorn api.app:app --reload --port 8000`
  - Test batch endpoint:
    ```
    curl -X POST http://localhost:8000/query \
         -H "Content-Type: application/json" \
         -d '{"question": "What is encoder-decoder attention?", "top_k": 5}'
    ```
  - Test streaming endpoint:
    ```
    curl -N "http://localhost:8000/query/stream?question=What+is+encoder-decoder+attention"
    ```

- [ ] Containerization with Docker
- [ ] ...
