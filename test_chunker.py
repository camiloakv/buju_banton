# test_chunker.py  (drop a sample .txt file next to this)
from ingestion.chunker import Chunker

chunker = Chunker(chunk_size=500, chunk_overlap=50)
docs = chunker.chunk_file("./data/raw/txt/sample.txt")

print(f"Total chunks: {len(docs)}")
print(f"\n--- Chunk 0 ---\n{docs[0].content}")
print(f"\n--- Chunk 1 (starts with overlap) ---\n{docs[1].content[:100]}")
