from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from query.pipeline import RAGPipeline
from eval.evaluator import run_evaluation
from eval.tracker import log_eval_run

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "balanced"   # resolves to llama3.1
TOP_K           = 5
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50

# Load the pipeline
embedder = Embedder(model_name=EMBEDDING_MODEL)
store    = VectorStore.load("./vector_store")
pipeline = RAGPipeline(embedder=embedder, store=store, top_k=TOP_K, model=LLM_MODEL)

# Run evaluation
print("Running RAGAs evaluation — this takes a few minutes...")
metrics = run_evaluation(pipeline, embedding_model=EMBEDDING_MODEL, llm_model="llama3.1")
print(f"!!!!!!!!!!!!!!!!!!!!!! {type(metrics)}")
print(f"!!!!!!!!!!!!!!!!!!!!!! {type(metrics.scores)}")
print(f"!!!!!!!!!!!!!!!!!!!!!! {metrics.scores}")
for i in metrics.scores:
    print(i)
    print(type(i))
    print("================")

# Log to MLflow
pipeline_config = {
    "embedding_model": EMBEDDING_MODEL,
    "llm_model":       LLM_MODEL,
    "top_k":           TOP_K,
    "chunk_size":      CHUNK_SIZE,
    "chunk_overlap":   CHUNK_OVERLAP,
}
log_eval_run(metrics, pipeline_config)
