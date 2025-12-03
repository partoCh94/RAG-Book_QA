from qdrant_client import QdrantClient

qdrant = QdrantClient(url="http://localhost:6334")


collection = "rag_books"

try:
    qdrant.delete_collection(collection)
    print(f"Deleted collection: {collection}")
except Exception as e:
    print("Error:", e)
