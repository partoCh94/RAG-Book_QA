# src/ingest.py
"""
Ingest with guaranteed local fallback producing 1536-dim embeddings.

Requirements:
 - If local fallback used, model = "intfloat/multilingual-e5-large-instruct" (1536 dims)
 - If that model can't be loaded, script prints diagnostic and exits.
"""
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

from pypdf import PdfReader
from dotenv import load_dotenv
import time, random
from typing import List

load_dotenv()

# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_books")

# Embedding settings
OPENAI_EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")  # 1536 dims
LOCAL_EMBED_MODEL = os.environ.get("LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-large-instruct")
EXPECTED_DIM = 1536

# Other settings
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", 16))
MAX_EMBED_RETRIES = int(os.environ.get("MAX_EMBED_RETRIES", 5))

# lazy holder
_st_model = None

# --------------------------
# PDF extract
# --------------------------
def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

# --------------------------
# Qdrant helpers
# --------------------------
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

def create_qdrant_collection(client: QdrantClient, name: str, vector_size: int = EXPECTED_DIM):
    try:
        collections = [c.name for c in client.get_collections().collections]
    except Exception as e:
        print("Failed to connect to Qdrant at", QDRANT_URL)
        print("Error:", e)
        raise
    if name in collections:
        print(f"Collection {name} exists ✔️")
        return
    client.recreate_collection(
        collection_name=name,
        vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
    )
    print(f"Created collection: {name} (dim={vector_size})")

# --------------------------
# Embedding helpers
# --------------------------
def load_local_model():
    """
    Load the precise local model that outputs EXPECTED_DIM.
    Uses trust_remote_code=True to support models that require it.
    """
    global _st_model
    if _st_model is not None:
        return _st_model

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers: pip install sentence-transformers") from e

    print(f"Loading local embedding model: {LOCAL_EMBED_MODEL} (this may take a while)...")
    _st_model = SentenceTransformer(LOCAL_EMBED_MODEL, device="cpu", trust_remote_code=True)
    # quick dim check
    emb = _st_model.encode("test", convert_to_numpy=True)
    dim = len(emb)
    if dim != EXPECTED_DIM:
        raise RuntimeError(f"Local model loaded but produced dim={dim}; expected {EXPECTED_DIM}. Model: {LOCAL_EMBED_MODEL}")
    print(f"Local model loaded OK — dimension = {dim}")
    return _st_model

def embed_with_openai(embedder, texts: List[str]):
    # embed_documents expects list and returns list of vectors
    return embedder.embed_documents(texts)

def embed_with_local(texts: List[str]):
    model = load_local_model()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embs.tolist()

def get_embeddings(embedder, texts: List[str]):
    """
    Try OpenAI embeddings; on quota/billing fallback to guaranteed-local (EXPECTED_DIM).
    Retries for transient errors.
    """
    for attempt in range(MAX_EMBED_RETRIES):
        try:
            return embed_with_openai(embedder, texts)
        except Exception as e:
            s = str(e).lower()
            # billing/quota -> immediate fallback to local
            if "insufficient_quota" in s or "quota" in s or "billing" in s:
                print("⚠️ OpenAI quota/billing issue detected. Switching to local fallback (guaranteed 1536).")
                return embed_with_local(texts)
            # rate-limits or transient -> backoff retry
            if "429" in s or "rate limit" in s or "rate_limit" in s:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit detected. Retry {attempt+1}/{MAX_EMBED_RETRIES} in {wait:.1f}s")
                time.sleep(wait)
                continue
            # other -> retry few times then fail
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"OpenAI error: {e}. Retry {attempt+1}/{MAX_EMBED_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError("Failed to get embeddings after retries.")

# --------------------------
# Main ingest
# --------------------------
def main(pdf_path: str, metadata: dict = None):
    if metadata is None:
        metadata = {}

    print("Extracting text...")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting into chunks...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.split_text(text)
    print(f"Chunks: {len(docs)}")

    print("Initializing embeddings (OpenAI)...")
    from langchain_openai import OpenAIEmbeddings
    embedder = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=OPENAI_EMBED_MODEL)

    print("Connecting to Qdrant...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=False)

    create_qdrant_collection(qdrant, COLLECTION_NAME, vector_size=EXPECTED_DIM)

    print("Upserting vectors to Qdrant...")
    batch_size = 64
    batch = []
    from tqdm import tqdm

    for start in range(0, len(docs), EMBED_BATCH_SIZE):
        slice_texts = docs[start:start + EMBED_BATCH_SIZE]
        try:
            embeddings = get_embeddings(embedder, slice_texts)
        except RuntimeError as e:
            print("Embedding failed:", e)
            return

        # validate dims and upsert
        for offset, emb in enumerate(embeddings):
            if len(emb) != EXPECTED_DIM:
                raise ValueError(f"Embedding dimension mismatch: expected {EXPECTED_DIM}, got {len(emb)}. Aborting.")
            i = start + offset
            payload = {
                "text": slice_texts[offset],
                "chunk_id": i,
                "source": metadata.get("source", os.path.basename(pdf_path))
            }
            batch.append(rest.PointStruct(id=i, vector=emb, payload=payload))
            if len(batch) >= batch_size:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
                batch = []

    if batch:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    print("✅ Done.")

# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="path to PDF")
    parser.add_argument("--source", default=None)
    args = parser.parse_args()
    main(args.pdf, metadata={"source": args.source})
