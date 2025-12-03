# ingest.py
import os
from pypdf import PdfReader
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"


load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6335")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_books")

# مدل سبک
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n\n"
    return text


def create_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        print(f"Collection {COLLECTION_NAME} exists ✔️")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(
            size=VECTOR_SIZE,
            distance=rest.Distance.COSINE
        )
    )
    print(f"Created collection: {COLLECTION_NAME}")


def main(pdf_path: str, metadata: dict = None):
    if metadata is None:
        metadata = {}

    print("Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = splitter.split_text(text)
    print(f"Chunks: {len(docs)}")

    print("Loading local embedding model...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=False,
    )

    create_collection(qdrant)

    print("Upserting vectors...")
    batch = []
    batch_size = 64

    for idx, chunk in tqdm(enumerate(docs), total=len(docs)):
        emb = embedder.encode(chunk).tolist()

        point = rest.PointStruct(
            id=idx,
            vector=emb,
            payload={
                "text": chunk,
                "chunk_id": idx,
                "source": metadata.get("source", os.path.basename(pdf_path)),
            }
        )
        batch.append(point)

        if len(batch) >= batch_size:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch = []

    if batch:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="path to PDF")
    parser.add_argument("--source", default=None)
    args = parser.parse_args()

    main(args.pdf, metadata={"source": args.source})
