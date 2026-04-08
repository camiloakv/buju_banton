from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore
from query.pipeline import RAGPipeline

# Load the index built by ingest.py
embedder = Embedder(model_name="all-MiniLM-L6-v2")
store    = VectorStore.load("./vector_store")

pipeline = RAGPipeline(embedder=embedder, store=store, top_k=5)

# Batch mode
response = pipeline.run("What is the refund policy?")
print(f"\nAnswer:\n{response.answer}")
print(f"\nSources used:")
for doc, score in response.sources:
    print(f"  [{score:.3f}] {doc.source} — chunk {doc.chunk_index}")

# Streaming mode
pipeline.stream("Summarize the key points about returns.")
