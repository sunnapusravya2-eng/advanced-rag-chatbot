# Advanced PDF RAG Chatbot

A production-ready Streamlit chatbot that ingests PDF documents, builds a FAISS vector index, and answers user queries using Google Gemini 2.5 Flash.

## Features

- Upload PDF documents through the Streamlit UI.
- Extract text from PDFs using `pypdf`.
- Split text into searchable chunks with `RecursiveCharacterTextSplitter`.
- Generate embeddings for chunks and store them in a FAISS vector database.
- Retrieve the most relevant chunks for any user query.
- Generate answers with Google Gemini 2.5 Flash using retrieved PDF context.
- Maintain chat history in Streamlit session state.
- Display chat conversations in ChatGPT-style bubbles.
- Show upload and vector database processing status.
- Provide a sidebar for model information and chat clearing.

## Project Structure

- `app.py` — main Streamlit application.
- `utils/pdf_loader.py` — PDF text extraction and chunking.
- `utils/vector_store.py` — FAISS index creation and retrieval.
- `utils/chatbot.py` — Gemini 2.5 Flash response generation.
- `requirements.txt` — Python dependencies.
- `README.md` — project overview and setup.

## Setup

1. Clone or copy this repository into your workspace.
2. Create a Python virtual environment.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your Google API key:

```bash
set GOOGLE_API_KEY=YOUR_API_KEY
```

5. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Upload a PDF file.
2. Wait for the PDF to be processed and the FAISS index to build.
3. Ask a question in the chat input.
4. Review the answer and the retrieved PDF chunks.

## Notes

- The app is designed for production-like usage with modular utilities and clear error handling.
- Keep your API key private and do not commit it to source control.
- If you want to reset the conversation, use the "Clear chat history" button in the sidebar.
