# rag_chain.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Defensive import: prefer langchain_community.vectorstores.Qdrant, fall back to langchain_qdrant.Qdrant
try:
    from langchain_community.vectorstores import Qdrant as LangChainQdrant
except Exception:
    try:
        from langchain_qdrant import Qdrant as LangChainQdrant
    except Exception:
        LangChainQdrant = None

from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import openai

load_dotenv()

# -------------------------------
# ENV variables
# -------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "rag_books")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------------
# Embedding class
# -------------------------------
class LocalMiniLMEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def embed_documents(self, texts):
        return self.model.encode(list(texts)).tolist()

    def embed_query(self, text):
        # return a single vector for the query
        return self.model.encode([text]).tolist()[0]


# -------------------------------
# Build retriever
# -------------------------------
def build_retriever(top_k=4):
    qdrant = QdrantClient(url=QDRANT_URL, prefer_grpc=False, api_key=QDRANT_API_KEY)

    # Robust shim: expose client.search by trying a sequence of possible underlying methods.
    # The wrapper will attempt each candidate with the same args/kwargs until one succeeds.
    def _make_search_wrapper(client):
        candidates = []
        for name in ("search", "search_points", "search_request", "search_vectors",
                     "search_matrix_pairs", "search_matrix_offsets"):
            fn = getattr(client, name, None)
            if callable(fn):
                candidates.append((name, fn))

        if not candidates:
            raise RuntimeError(
                "Incompatible qdrant-client: no search-like methods found on QdrantClient.\n"
                "Recommended fix: upgrade qdrant-client and langchain-qdrant:\n\n"
                "    pip install -U \"qdrant-client>=1.7.0\" langchain-qdrant\n\n"
                "If you cannot upgrade, paste the output of `pip show qdrant-client` and "
                "the list of QdrantClient attributes and I'll craft a shim."
            )

        def _search(*args, **kwargs):
            last_exc = None
            for name, fn in candidates:
                try:
                    return fn(*args, **kwargs)
                except TypeError as e:
                    # signature mismatch — try next candidate
                    last_exc = e
                    continue
                except Exception as e:
                    # method exists but call failed; keep last exception and try next
                    last_exc = e
                    continue
            # If we get here, none of the candidate calls succeeded
            raise RuntimeError(
                "QdrantClient: attempted search candidates "
                f"{[n for n,_ in candidates]} but all failed. Last error: {last_exc}"
            ) from last_exc

        client.search = _search

    _make_search_wrapper(qdrant)

    embeddings = LocalMiniLMEmbeddings()

    vectorstore = LangChainQdrant(
        client=qdrant,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings   # ← درست برای langchain_qdrant
    )

    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# -------------------------------
# Main RAG Chain
# -------------------------------
def build_rag_qa_chain(top_k=4):
    retriever = build_retriever(top_k)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0
    )

    # Try LangChain RetrievalQA (if available)
    try:
        from langchain.chains import RetrievalQA

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        qa.retriever = retriever
        return qa

    except Exception:
        # Fallback QA
        def _fetch_documents_from_retriever(retriever, query):
            """
            Unified document fetcher that supports all LangChain retriever versions.
            """
            # New LCEL retrievers
            if hasattr(retriever, "invoke"):
                return retriever.invoke(query)

            # Old retrievers
            if hasattr(retriever, "get_relevant_documents"):
                return retriever.get_relevant_documents(query)

            # Older vectorstores
            if hasattr(retriever, "similarity_search"):
                return retriever.similarity_search(query)

            raise AttributeError(
                f"The retriever object of type {type(retriever)} does not support "
                "`invoke`, `get_relevant_documents`, or `similarity_search`."
            )

        class SimpleRetrievalQA:
            def __init__(self, retriever, model_name, api_key, return_source_documents=True):
                self.retriever = retriever
                self.model = model_name
                self.api_key = api_key
                self.return_source_documents = return_source_documents

            @classmethod
            def from_chain_type(cls, llm=None, retriever=None, return_source_documents=True):
                model_name = getattr(llm, "model", LLM_MODEL)
                api_key = os.environ["OPENAI_API_KEY"]
                return cls(retriever, model_name, api_key, return_source_documents)

            def run(self, query: str):
                docs = _fetch_documents_from_retriever(self.retriever, query)

                passages = []
                for i, d in enumerate(docs):
                    text = getattr(d, 'page_content', None) or getattr(d, 'content', None) or str(d)
                    passages.append(f"[{i}] {text}")

                system_prompt = (
                    "You are a helpful assistant. Use the passages to answer. "
                    "If unsure, say 'I don't know'."
                )

                user_prompt = (
                    f"Question: {query}\n\nPassages:\n" + "\n".join(passages)
                )

                openai.api_key = self.api_key

                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0
                )

                answer = resp["choices"][0]["message"]["content"]

                if self.return_source_documents:
                    return {"result": answer, "source_documents": docs}
                return answer

        return SimpleRetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
