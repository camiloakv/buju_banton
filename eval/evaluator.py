from typing import Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings

from query.pipeline import RAGPipeline
from eval.dataset import build_eval_dataset


def _get_ragas_llm(model: str = "llama3.1"):
    """Point RAGAs at your local Ollama model for scoring."""
    return LangchainLLMWrapper(OllamaLLM(model=model))


def _get_ragas_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Reuse the same embedding model used during indexing."""
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=model_name)
    )


def run_evaluation(
    pipeline: RAGPipeline,
    embedding_model: str = "all-MiniLM-L6-v2",
    llm_model: str = "llama3.1",
) -> dict[str, float]:
    """
    Runs the full RAGAs evaluation loop.
    Returns a dict of metric_name → score (all in [0, 1]).
    """
    eval_dataset = build_eval_dataset()

    # 1. Run pipeline on every question, collect answers + contexts
    records = []
    for row in eval_dataset:
        response = pipeline.run(row["question"])
        records.append({
            "question":         row["question"],
            "ground_truth":     row["ground_truth"],
            "answer":           response.answer,
            # RAGAs expects a list of strings — one string per retrieved chunk
            "contexts":         [doc.content for doc, _ in response.sources],
        })

    ragas_dataset = Dataset.from_list(records)

    # 2. Score with RAGAs (uses Ollama locally — no API calls)
    results = evaluate(
        dataset   = ragas_dataset,
        metrics   = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm        = _get_ragas_llm(llm_model),
        embeddings = _get_ragas_embeddings(embedding_model),
    )

    return results
