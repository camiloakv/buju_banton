import mlflow
from datetime import datetime
from ragas.evaluation import EvaluationResult


def log_eval_run(
    results:         EvaluationResult,
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

    # Convert EvaluationResult → plain dict of metric_name: float
    metrics = {k: float(v) for k, v in results.scores.items()}  # ← fix
    print("IN TRACKER -----------------------------------")
    print(len(metrics))
    for metric_name, score in metrics.items():
        print(type(metric_name), type(score))
        print(metric_name)
        print(score)
        print("-----------")

    with mlflow.start_run(run_name=run_name) as run:

        # Log pipeline config as params
        mlflow.log_params(pipeline_config)

        # Log RAGAs scores as metrics
        for metric_name, score in metrics.items():
            mlflow.log_metric(metric_name, score)

        # Composite score — useful for quick ranking across runs
        avg = sum(metrics.values()) / len(metrics)
        mlflow.log_metric("mean_score", avg)

        print(f"\nMLflow run: {run_name}  (id: {run.info.run_id})")
        for metric_name, score in metrics.items():
            print(f"  {metric_name:<22} {score:.3f}")
        print(f"  {'mean_score':<22} {avg:.3f}")

    return run.info.run_id
