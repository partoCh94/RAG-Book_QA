# rag_chain.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

load_dotenv()

# --------------------------------------------------
# Environment variables
# --------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_books")

LLM_MODEL = os.environ.get("LLM_MODEL", "phi")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Embeddings (local, offline-friendly)
# --------------------------------------------------
class LocalMiniLMEmbeddings(Embeddings):
    def __init__(self):
        import os
       # cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        print("Loading embedding model...")
        # Load model - will download if cache was cleared
        self.model = SentenceTransformer(
            EMBED_MODEL_NAME,
           # cache_folder=cache_dir,
            device="cpu"
        )
        print("[OK] Loaded embedding model")

    def embed_documents(self, texts):
        return self.model.encode(list(texts)).tolist()

    def embed_query(self, text):
        return self.model.encode([text]).tolist()[0]


# --------------------------------------------------
# Build retriever (Qdrant only, with local embeddings)
# --------------------------------------------------
def build_retriever(top_k: int = 4):
    embeddings = LocalMiniLMEmbeddings()
    
    print("[*] Connecting to Qdrant vectorstore...")

    try:
        # Try langchain_qdrant first (newer, better supported)
        try:
            from langchain_qdrant import Qdrant as QdrantVectorstore
            print("  [->] Using langchain_qdrant (new)")
        except ImportError:
            from langchain_community.vectorstores import Qdrant as QdrantVectorstore
            print("  [->] Using langchain_community (old/deprecated)")
        
        # Use HTTP with explicit proxy bypass for localhost
        import os
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            timeout=30,
            check_compatibility=False,
        )
        
        # Test connection by getting collection info
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        print(f"  [OK] Connected to Qdrant (HTTP)")
        print(f"      Collection: {COLLECTION_NAME}")
        print(f"      Documents: {collection_info.points_count}")
        
        vectorstore = QdrantVectorstore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME,
            embeddings=embeddings,
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        print("  [OK] Retriever ready")
        return retriever
        
    except Exception as e:
        print(f"  [ERROR] Failed to connect to Qdrant: {type(e).__name__}")
        print(f"          Error: {str(e)[:100]}")
        print(f"          Please ensure:")
        print(f"          1. Docker is running: docker ps")
        print(f"          2. Qdrant is started: docker start qdrant")
        print(f"          3. Collection has documents: python src/ingest.py books/thinkpython2.pdf")
        raise RuntimeError(f"Qdrant connection failed: {e}")



# --------------------------------------------------
# Build RAG QA chain (with Ollama Local LLM)
# --------------------------------------------------
def build_rag_qa_chain(top_k: int = 4):
    retriever = build_retriever(top_k)

    # Use Ollama for local LLM (no API key needed!)
    try:
        from langchain_community.llms import Ollama
        
        print("[*] Initializing Ollama LLM...")
        llm = Ollama(
            model=LLM_MODEL,
            base_url="http://localhost:11434",  # default Ollama port
            temperature=0,
            timeout=120,  # 2-minute timeout
            num_predict=256,  # limit output length
        )
        print(f"[OK] Ollama connected (model: {LLM_MODEL})")
    except Exception as e:
        print(f"[ERROR] Ollama not available: {e}")
        print("[*] Make sure Ollama is running: ollama serve")
        raise

    # Use SimpleRetrievalQA which works reliably across versions
    class SimpleRetrievalQA:
        def __init__(self, retriever, llm):
            self.retriever = retriever
            self.llm = llm

        def _get_answer(self, query: str):
            try:
                docs = self.retriever.invoke(query)
            except Exception:
                try:
                    docs = self.retriever.get_relevant_documents(query)
                except Exception:
                    docs = []

            if not docs:
                return {
                    "result": f"No documents found for: {query}",
                    "source_documents": [],
                }

            context = "\n\n".join(d.page_content for d in docs)

            prompt_text = (
                "Use the context below to answer the question.\n"
                "If the answer is not in the context, say 'I don't know'.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )

            try:
                response = self.llm.invoke(prompt_text)
                answer = response if isinstance(response, str) else str(response)
            except TimeoutError:
                answer = "Request timed out. Please try again."
            except Exception as err:
                answer = f"Error: {type(err).__name__}: {str(err)[:100]}"

            return {
                "result": answer,
                "source_documents": docs,
            }

        def run(self, query: str):
            return self._get_answer(query)

        def invoke(self, query: str):
            return self._get_answer(query)

        def __call__(self, query: str):
            return self._get_answer(query)

    print("[OK] Using SimpleRetrievalQA with Ollama")
    return SimpleRetrievalQA(retriever, llm)
