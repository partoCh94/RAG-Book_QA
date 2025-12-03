# app.py
import streamlit as st
from rag_chain import build_rag_qa_chain

st.set_page_config(page_title="RAG Book-QA", layout="wide")

st.title("📘 RAG Book-QA")

with st.sidebar:
    st.header("تنظیمات")
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=4)
    st.write("قبل از پرسیدن سؤال، مطمئن شو Qdrant و دیتا ingest شده باشد.")

# -----------------------------------
# Build chain only once
# -----------------------------------
if "qa_chain" not in st.session_state or \
   st.session_state.get("qa_top_k") != top_k:
    with st.spinner("در حال ساخت RAG chain ..."):
        st.session_state.qa_chain = build_rag_qa_chain(top_k=top_k)
        st.session_state.qa_top_k = top_k

qa = st.session_state.qa_chain

# -----------------------------------
# User query
# -----------------------------------
query = st.text_input("سؤال خودت را بنویس...")

if st.button("پرسش") and query:
    with st.spinner("در حال پردازش..."):
        try:
            result = qa.run(query)
        except Exception as e:
            st.error(f"Error running QA chain: {e}")
            raise

        # Normalize result
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer")
            source_docs = result.get("source_documents", [])
        else:
            answer = str(result)
            source_docs = []

        # Output
        st.subheader("پاسخ")
        st.write(answer)

        st.subheader("منابع")
        for i, doc in enumerate(source_docs):
            with st.expander(f"Chunk {i}"):
                content = getattr(doc, "page_content", None) \
                        or getattr(doc, "content", None) \
                        or str(doc)
                st.write(content)

                metadata = getattr(doc, "metadata", None)
                if metadata:
                    st.write(metadata)
