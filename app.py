import os
import traceback
import streamlit as st
from typing import Optional
from dotenv import load_dotenv
from utils.pdf_loader import extract_text_from_pdf, split_text_into_chunks
from utils.vector_store import create_embedding_model, build_faiss_index, retrieve_similar_chunks
from utils.chatbot import create_chat_model, generate_answer

# Load environment variables from .env file
load_dotenv()


st.set_page_config(
    page_title="Advanced PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)


def init_session_state() -> None:
    """Initialize all required Streamlit session state values."""
    defaults = {
        "history": [],
        "chunks": [],
        "pdf_name": None,
        "vector_index": None,
        "vector_ready": False,
        "processing_status": "No PDF loaded.",
        "retrieval_results": [],
        "error_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_api_key() -> Optional[str]:
    """Read the Google API key from environment variables."""
    return os.getenv("GOOGLE_API_KEY")


@st.cache_resource
def load_embedding_model(api_key: Optional[str]):
    return create_embedding_model(api_key=api_key)


@st.cache_resource
def load_chat_model(api_key: Optional[str]):
    return create_chat_model(api_key=api_key)


def process_pdf(uploaded_file, api_key: Optional[str]) -> None:
    """Extract text from the uploaded PDF, split into chunks, and build the FAISS index."""
    try:
        st.session_state.processing_status = "Reading PDF and extracting text..."
        raw_text = extract_text_from_pdf(uploaded_file)

        if not raw_text:
            raise ValueError("PDF did not contain readable text.")

        st.session_state.processing_status = "Splitting PDF text into chunks..."
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        chunks = split_text_into_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            raise ValueError("Failed to split the PDF into searchable chunks.")

        st.session_state.processing_status = "Creating FAISS vector index..."
        embedding_model = load_embedding_model(api_key)

        if embedding_model is None:
            raise ValueError(
                "Embedding model failed to initialize. Check your GOOGLE_API_KEY."
            )

        index, total_chunks = build_faiss_index(chunks, embedding_model)

        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chunks = chunks
        st.session_state.vector_index = index
        st.session_state.vector_ready = total_chunks > 0
        st.session_state.processing_status = (
            f"✅ Loaded {total_chunks} chunks and built the vector database."
        )
        st.session_state.error_message = None
    except Exception as error:
        st.session_state.chunks = []
        st.session_state.vector_index = None
        st.session_state.vector_ready = False
        st.session_state.processing_status = "❌ Failed to process PDF."
        st.session_state.error_message = str(error)
        raise


def render_sidebar() -> None:
    """Render sidebar information and controls."""
    st.sidebar.title("🤖 PDF RAG Chatbot")
    st.sidebar.markdown(
        "This app ingests PDF content, builds a FAISS vector index, and uses "
        "**Google Gemini 2.5 Flash** for retrieval-augmented responses. "
        "Upload a PDF, ask questions, and keep the conversation flowing."
    )

    if st.sidebar.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.sidebar.success("Chat cleared.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Deployment notes")
    st.sidebar.markdown(
        "- Set `GOOGLE_API_KEY` in your `.env` file before running.\n"
        "- Gemini 2.5 Flash is used for response generation.\n"
        "- Google `text-embedding-004` is used for PDF retrieval.\n"
        "- FAISS keeps the most relevant PDF chunks available for retrieval."
    )

    st.sidebar.markdown("---")
    api_key = get_api_key()
    st.sidebar.write("**API Key Configured:**", bool(api_key))
    if not api_key:
        st.sidebar.error("❌ Missing GOOGLE_API_KEY in .env file.")
    else:
        st.sidebar.success("✅ Google API key found.")


def render_status_cards() -> None:
    """Render high-level status cards for the current session."""
    pdf_name = st.session_state.pdf_name or "No file"
    chunk_count = len(st.session_state.chunks)
    vector_status = "✅ Ready" if st.session_state.vector_ready else "❌ Not ready"

    col1, col2, col3 = st.columns(3)
    col1.metric("📄 PDF file", pdf_name)
    col2.metric("🔢 Chunk count", chunk_count)
    col3.metric("🗄️ Vector store", vector_status)

    if st.session_state.error_message:
        st.error(f"❌ {st.session_state.error_message}")

    st.info(st.session_state.processing_status)


def render_chat_history() -> None:
    """Render the chat history as chat bubbles."""
    if not st.session_state.history:
        st.write("_Upload a PDF and ask a question to start the conversation._")
        return

    for message in st.session_state.history:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("🤖 Advanced PDF RAG Chatbot")
    st.markdown(
        "Upload a PDF, ask questions, and get answers with citations from the most relevant PDF chunks. "
        "This app keeps your chat history and reuses context to answer follow-up questions."
    )

    api_key = get_api_key()

    if not api_key:
        st.warning("⚠️ Please set your `GOOGLE_API_KEY` in the `.env` file and restart the app.")
        return

    uploaded_file = st.file_uploader("📂 Upload a PDF document", type=["pdf"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name != st.session_state.pdf_name or not st.session_state.vector_ready:
                st.session_state.history = []
                with st.spinner("Processing PDF..."):
                    process_pdf(uploaded_file, api_key)
        except Exception:
            st.exception(traceback.format_exc())

    render_status_cards()

    if st.session_state.vector_ready:
        question = st.chat_input("💬 Ask a question about the uploaded PDF...")
        if question:
            st.session_state.history.append({"role": "user", "content": question})
            with st.spinner("Retrieving relevant PDF context..."):
                try:
                    embedding_model = load_embedding_model(api_key)
                    retrieved_chunks = retrieve_similar_chunks(
                        question,
                        st.session_state.vector_index,
                        st.session_state.chunks,
                        embedding_model,
                        top_k=int(os.getenv("NUM_RETRIEVED_CHUNKS", "4")),
                    )
                    st.session_state.retrieval_results = retrieved_chunks

                    if not retrieved_chunks:
                        raise ValueError(
                            "No matching content was found in the PDF. Try a different query."
                        )

                    chat_model = load_chat_model(api_key)
                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                        chat_model,
                        st.session_state.history,
                    )

                    st.session_state.history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as error:
                    error_message = str(error)
                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": f"Sorry, I could not complete your request. Error: {error_message}",
                        }
                    )
                    st.error(error_message)

    render_chat_history()

    if st.session_state.retrieval_results:
        with st.expander("📋 Retrieved PDF chunks", expanded=False):
            for item in st.session_state.retrieval_results:
                st.markdown(
                    f"**Chunk {item['index'] + 1}** — score: `{item['score']:.3f}`\n\n{item['content']}"
                )


if __name__ == "__main__":
    main()