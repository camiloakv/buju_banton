from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from ingestion.vector_store import VectorStore

chunker  = Chunker(chunk_size=500, chunk_overlap=50)
embedder = Embedder(model_name="all-MiniLM-L6-v2")

# 1. Chunk
print("Chunking documents...")
docs = chunker.chunk_directory("./data")
print(f"Total chunks: {len(docs)}\n")

# 2. Embed
print("Embedding chunks...")
embeddings = embedder.embed_documents(docs)

# 3. Index
print("\nBuilding vector store...")
store = VectorStore(dimension=embedder.dimension)
store.add(docs, embeddings)

# 4. Persist
store.save("./vector_store")
print("Done.")
