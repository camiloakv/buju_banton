import mlflow
from datetime import datetime


def log_eval_run(
    metrics:         dict[str, float],
    pipeline_config: dict,
    run_name:        str | None = None,
) -> str:
    """
    Logs an evaluation run to MLflow.
    Returns the run_id for reference.

    pipeline_config should include everything that affects quality:
        chunk_size, chunk_overlap, embedding_model, llm_model, top_k
    """
    mlflow.set_experiment("rag-evaluation")

    run_name = run_name or datetime.now().strftime("eval_%Y%m%d_%H%M%S")

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
        print(f"  faithfulness:      {metrics.get('faithfulness', 0):.3f}")
        print(f"  answer_relevancy:  {metrics.get('answer_relevancy', 0):.3f}")
        print(f"  context_precision: {metrics.get('context_precision', 0):.3f}")
        print(f"  context_recall:    {metrics.get('context_recall', 0):.3f}")
        print(f"  mean_score:        {avg:.3f}")

    return run.info.run_id
