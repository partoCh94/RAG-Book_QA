# app.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(page_title="RAG Book-QA", layout="wide")
st.title("[DOC] RAG Book-QA")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=4)
    st.write("Make sure Qdrant and data are ingested before asking.")

# Lazy import to avoid issues
st.write("[*] Importing RAG chain...")
try:
    from rag_chain import build_rag_qa_chain
    st.write("[OK] RAG module imported")
except Exception as e:
    st.error(f"Failed to import: {e}")
    st.stop()

# Build chain once
st.write("[*] Building RAG chain...")
if "qa_chain" not in st.session_state or \
   st.session_state.get("qa_top_k") != top_k:
    try:
        qa = build_rag_qa_chain(top_k=top_k)
        st.session_state.qa_chain = qa
        st.session_state.qa_top_k = top_k
        st.write("[OK] RAG chain built!")
    except Exception as e:
        st.error(f"[ERROR] Chain build failed: {type(e).__name__}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()
else:
    qa = st.session_state.qa_chain
    st.write("[OK] Using cached chain")

st.divider()

# User query
st.write("### Ask a question:")
query = st.text_input("Your question...")

if st.button("Ask") and query:
    with st.spinner("Processing..."):
        try:
            result = qa.run(query)
        except (AttributeError, TypeError):
            try:
                result = qa(query)
            except Exception:
                result = qa.invoke(query)

        # Normalize result
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer")
            source_docs = result.get("source_documents", [])
        else:
            answer = str(result)
            source_docs = []

        # Output
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, doc in enumerate(source_docs):
            with st.expander(f"Document {i+1}"):
                content = getattr(doc, "page_content", None) \
                        or getattr(doc, "content", None) \
                        or str(doc)
                st.write(content)

                metadata = getattr(doc, "metadata", None)
                if metadata:
                    st.write("**Metadata:**")
                    st.write(metadata)
