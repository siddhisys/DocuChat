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
