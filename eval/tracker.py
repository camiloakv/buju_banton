import mlflow
from datetime import datetime
#from ragas.evaluation import EvaluationResult


def log_eval_run(
#    results:         EvaluationResult,
    results,
    pipeline_config: dict,
    run_name:        str | None = None,
) -> str:
    """
    Logs an evaluation run to MLflow.
    Returns the run_id for reference.

    pipeline_config should include everything that affects quality:
        chunk_size, chunk_overlap, embedding_model, llm_model, top_k
    """
    mlflow.set_experiment("another-rag-evaluation")

    run_name = run_name or datetime.now().strftime("eval_%Y%m%d_%H%M%S")

    # results.scores is a list of dicts, one per sample:
    # [
    #   {"faithfulness": 0.9, "answer_relevancy": 0.8, ...},
    #   {"faithfulness": 0.7, "answer_relevancy": 0.9, ...},
    #   ...
    # ]
    sample_scores = results.scores
    metric_names  = list(sample_scores[0].keys())

    # Average each metric across all samples
    metrics = {
        name: sum(s[name] for s in sample_scores if s[name] is not None)
              / max(sum(1 for s in sample_scores if s[name] is not None), 1)
        for name in metric_names
    }

    with mlflow.start_run(run_name=run_name) as run:

        # Log pipeline config as params
        mlflow.log_params(pipeline_config)

        # Log RAGAs scores as metrics
        for metric_name, score in metrics.items():
            mlflow.log_metric(metric_name, float(score))

        # Composite score — useful for quick ranking across runs
        avg = sum(metrics.values()) / len(metrics)
        mlflow.log_metric("mean_score", avg)

        print(f"\nMLflow run: {run_name}  (id: {run.info.run_id})")
        for metric_name, score in metrics.items():
            print(f"  {metric_name:<22} {score:.3f}")
        print(f"  {'mean_score':<22} {avg:.3f}")

    return run.info.run_id
