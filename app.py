from __future__ import annotations

import hashlib

import streamlit as st

from src.config import settings
from src.generator import ExtractiveGenerator, TransformersGenerator
from src.pdf_processor import extract_pdf
from src.rag import RAGPipeline
from src.vector_store import FaissVectorStore, MiniLMEncoder


st.set_page_config(page_title="DocuChat AI", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_encoder():
    return MiniLMEncoder(settings.embedding_model)


@st.cache_resource(show_spinner="Loading the language model…")
def get_generator():
    if settings.backend == "transformers":
        adapter = str(settings.adapter_path) if settings.adapter_path.exists() else None
        return TransformersGenerator(settings.model_id, adapter)
    return ExtractiveGenerator()


def process_pdf(file_bytes: bytes):
    chunks, pages = extract_pdf(file_bytes)
    store = FaissVectorStore(get_encoder())
    store.build(chunks)
    return RAGPipeline(store, get_generator(), settings.top_k), chunks, pages


st.title("📄 DocuChat AI")
st.write("Upload a text-based PDF and ask grounded questions about its content.")

with st.sidebar:
    st.header("Configuration")
    st.code(
        f"LLM: {settings.model_id}\n"
        f"Embeddings: all-MiniLM-L6-v2\n"
        f"Retrieval: top-{settings.top_k}\n"
        f"Backend: {settings.backend}"
    )
    st.info("The default extractive mode is fast. Set GENERATOR_BACKEND=transformers to use the LLM.")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    file_bytes = uploaded.getvalue()
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("fingerprint") != fingerprint:
        try:
            with st.spinner("Cleaning, chunking, embedding, and indexing the document…"):
                pipeline, chunks, pages = process_pdf(file_bytes)
            st.session_state.update(
                fingerprint=fingerprint, pipeline=pipeline, chunks=chunks, pages=pages
            )
        except Exception as error:
            st.error(str(error))
            st.stop()

    st.success(
        f"Processed {st.session_state.pages} pages into {len(st.session_state.chunks)} chunks."
    )
    question = st.text_input("Ask a question", placeholder="What is the main conclusion?")
    if st.button("Get answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching the document…"):
                result = st.session_state.pipeline.ask(question)
            st.subheader("Answer")
            st.write(result.text)
            st.caption(f"Generated in {result.latency_seconds:.2f} seconds")
            with st.expander("Retrieved evidence"):
                for chunk, score in zip(result.sources, result.scores):
                    st.markdown(f"**Page {chunk.page} · similarity `{score:.3f}`**")
                    st.write(chunk.text)
else:
    st.info("Upload a PDF to begin. Scanned image-only PDFs are not supported in this simple version.")

