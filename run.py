from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from query.pipeline import RAGPipeline

# NEW
from dotenv import load_dotenv

load_dotenv()
# NEW

# Load the index built by ingest.py
embedder = Embedder(model_name="all-MiniLM-L6-v2")
store    = VectorStore.load("./vector_store")

pipeline = RAGPipeline(
    embedder = embedder,
    store    = store,
    top_k    = 5,
    model    = "balanced",  # "fast", "balanced", "reasoning", "long", "structured", "tiny", "multilingual",
)

# Batch mode
response = pipeline.run("What is encoder-decoder attention?")
print(f"\nAnswer:\n{response.answer}")
print(f"\nSources used:")
for doc, score in response.sources:
    print(f"  [{score:.3f}] {doc.source} — chunk {doc.chunk_index}")

# Streaming mode
pipeline.stream("Summarize the key points about attention.")
