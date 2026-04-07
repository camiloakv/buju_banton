from ingestion.chunker import Chunker
from ingestion.embedder import Embedder

chunker  = Chunker(chunk_size=500, chunk_overlap=50)
embedder = Embedder(model_name="all-MiniLM-L6-v2")

# 1. Load and chunk all files
print("Chunking documents...")
docs = chunker.chunk_directory("./data")       # put your PDFs/TXTs here
print(f"Total chunks: {len(docs)}")

# 2. Embed all chunks
print("\nEmbedding chunks...")
embeddings = embedder.embed_documents(docs)    # shape: (N, 384)

print(f"\nDone. Embedding matrix shape: {embeddings.shape}")
print(f"Sample vector (first 5 dims): {embeddings[0][:5]}")
