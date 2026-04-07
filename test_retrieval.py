from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore

embedder = Embedder(model_name="all-MiniLM-L6-v2")
store    = VectorStore.load("./vector_store")

query   = "What is encoder-decoder attention?"
q_vec   = embedder.embed_query(query)
results = store.search(q_vec, top_k=3)

for doc, score in results:
    print(f"\n[score={score:.4f}] {doc.source} — chunk {doc.chunk_index}")
    print(doc.content[:300])
    print("...")
