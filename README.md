# DocuChat - AI Document Assistant with RAG (Retrieval Augmented Generation)

DocuChat is a Streamlit-based AI-powered application that allows users to chat normally with an AI or upload PDF documents to ask specific questions about their content using Retrieval Augmented Generation (RAG). It leverages language models and vector search for precise document-based Q&A.

---

## Features

- **Normal Chat Mode:** Interact with an AI assistant in general conversation mode.
- **Document Q&A Mode:** Upload one or more PDF files, which are processed into text chunks and embedded into a vector store. Ask questions specifically about the content of these PDFs.
- **Smart Text Chunking:** Uses recursive character splitting to create manageable text chunks from large documents for better context handling.
- **Fast Semantic Search:** Uses FAISS vector store to enable quick similarity search across document chunks.
- **Secure Environment Variables:** Uses `.env` files to safely store API keys and configuration.
- **Easy-to-use Web Interface:** Powered by Streamlit for rapid interaction and user feedback.

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/siddhisys/DocuChat.git
    cd DocuChat
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file for your API keys and configurations (if required):
    ```
    # Example:
    OLLAMA_API_KEY=your_api_key_here
    ```

---

## Usage

Run the Streamlit app:
```bash
streamlit run main.py

---

How to use DocuChat:

Normal Chat Mode:

Simply type your question in the input box and chat with the AI assistant.

Document Q&A Mode:

Upload one or multiple PDF files using the sidebar uploader.

Click "Process PDFs" to extract, chunk, and embed the document contents.

Once processed, ask questions specifically related to the uploaded documents.

Click "Clear PDFs" to reset and upload new files.

Code Overview

main.py contains the full Streamlit application logic.

Uses:

PyPDF2 for PDF text extraction.

langchain for text chunking, embeddings, vector search, and retrieval.

streamlit for UI and caching.

dotenv to load environment variables securely.

Implements caching for LLM and embedding models to improve performance.

Supports two modes:

Normal Chat: Direct LLM interaction.

RAG Q&A: Retrieves relevant document chunks using vector similarity and answers based on retrieved context.

Dependencies

Python 3.8+

streamlit

python-dotenv

PyPDF2

langchain

faiss-cpu

Other dependencies as listed in requirements.txt

Troubleshooting

Ensure Ollama model server is running and accessible.

Verify .env variables are correctly set.

Check PDF files are not corrupted.

If you encounter push errors on GitHub, consider manual file upload as a workaround.
