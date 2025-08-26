# DocuChat - AI Document Assistant with RAG (Retrieval Augmented Generation)
# This application allows users to either chat normally with an AI or upload PDFs and ask questions specifically about those documents

#importing all the necessary tools
import streamlit as st  # Web framework for creating interactive Python apps
from dotenv import load_dotenv  # Loads environment variables from .env file for security
from PyPDF2 import PdfReader  # Library to read and extract text from PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Smart text chunking algorithm
from langchain_community.vectorstores import FAISS  # Facebook's vector database for similarity search
from langchain_ollama import ChatOllama, OllamaEmbeddings  # LangChain connectors for Ollama models
from langchain.chains import RetrievalQA  # Chain that combines retrieval and question-answering

# Load environment variables from .env file (API keys, configurations, etc.)
load_dotenv()

# CACHING FUNCTIONS - Performance Optimization
# These functions use Streamlit's caching to avoid expensive re-initialization

# Initialize LLM globally to avoid recreation
@st.cache_resource  # Caches the model in memory to avoid reloading on every interaction
def get_llm():
    # Create and return a ChatOllama instance with gemma3:1b model
    # temperature=0.7 provides balanced creativity (0=deterministic, 1=very creative)
    return ChatOllama(model="gemma3:1b", temperature=0.7)
    #preventing expensive model loading plus generating natural variety in responses

@st.cache_resource  # Cache the embedding model to avoid reloading
def get_embeddings():
    # Return embedding model that converts text to vectors for similarity search
    # nomic-embed-text is optimized for general text embedding tasks
    return OllamaEmbeddings(model="nomic-embed-text")

# ---------- PDF PROCESSING FUNCTIONS ----------

def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF documents
    Args: pdf_docs - list of uploaded PDF file objects from Streamlit
    Returns: concatenated text string from all PDFs
    """
    text = ""  # Initialize empty string to store all extracted text
    
    if pdf_docs:  # Check if any PDFs were uploaded
        # Loop through each uploaded PDF file
        for pdf in pdf_docs:
            try:
                # Create a PDF reader object for the current file
                pdf_reader = PdfReader(pdf)
                
                # Extract text from each page in the PDF
                for page in pdf_reader.pages:
                    page_text = page.extract_text()  # Extract text from current page
                    if page_text:  # Only add if text was successfully extracted
                        text += page_text + "\n"  # Add page text with newline separator
                        
            except Exception as e:
                # Handle errors gracefully (corrupted PDFs, permissions, etc.)
                st.error(f"Error reading {pdf.name}: {str(e)}")
                
    return text  # Return the complete extracted text

def get_text_chunks(raw_text):
    """
    Split large text into smaller, manageable chunks for processing
    This is crucial because AI models have token limits and smaller chunks improve search precision
    """
    # Return empty list if no text provided
    if not raw_text.strip():
        return []
    
    # Create a RecursiveCharacterTextSplitter - the "smart" text splitter
    # It tries multiple separators in order: paragraphs -> sentences -> words -> characters
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # Maximum characters per chunk (balance between context and precision)
        chunk_overlap=100,   # Characters that overlap between adjacent chunks (prevents info loss)
        length_function=len  # Function to measure chunk length (using character count)
    )
    
    # Split the text into chunks using the configured splitter
    chunks = splitter.split_text(raw_text)
    
    # Return only non-empty chunks (filter out whitespace-only chunks)
    return [chunk for chunk in chunks if chunk.strip()]

def get_vector_store(text_chunks):
    """
    Create a vector store from text chunks using FAISS
    This converts text into mathematical vectors for semantic similarity search
    """
    # Return None if no chunks provided
    if not text_chunks:
        return None
    
    # Get the cached embedding model
    embeddings = get_embeddings()
    
    # Create FAISS vector store from text chunks
    # This process: text -> embedding vectors -> searchable index
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    return vectorstore

# ---------- MAIN STREAMLIT APPLICATION ----------

def main():
    """
    Main function that creates the Streamlit web interface
    """
    
    # Configure the Streamlit page
    st.set_page_config(page_title="DocuChat", page_icon="📚")
    st.header("📚 DocuChat - Chat with PDFs or Normally")

    # INITIALIZE SESSION STATE
    # Session state persists data between Streamlit reruns (when user interacts)
    
    if 'vector_store' not in st.session_state:
        # Store the FAISS vector database containing document embeddings
        st.session_state.vector_store = None
        
    if 'processed' not in st.session_state:
        # Flag to track whether documents have been successfully processed
        st.session_state.processed = False

    # SIDEBAR - PDF UPLOAD AND PROCESSING INTERFACE
    with st.sidebar:
        st.subheader("Upload your PDFs")
        
        # File uploader widget - allows multiple PDF uploads
        pdf_docs = st.file_uploader(
            "Upload PDFs (optional)",  # Label shown to user
            accept_multiple_files=True,  # Allow multiple file selection
            type=['pdf']  # Only accept PDF files
        )
        
        # PROCESS PDFs BUTTON
        if st.button("Process PDFs"):
            if pdf_docs:  # Check if files were uploaded
                
                # Show spinner while processing
                with st.spinner("Processing PDFs..."):
                    try:
                        # STEP 1: Extract text from PDFs
                        st.write("Extracting text...") 
                        raw_text = get_pdf_text(pdf_docs)
                        
                        # Validate that text was extracted
                        if not raw_text.strip():
                            st.error("No text found in PDFs!")
                            return  # Exit function early
                        
                        # Show extraction results
                        st.write(f"Extracted {len(raw_text)} characters")
                        
                        # STEP 2: Create text chunks
                        st.write("Creating chunks...")
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Validate chunk creation
                        if not text_chunks:
                            st.error("Could not create text chunks!")
                            return
                            
                        # Show chunking results
                        st.write(f"Created {len(text_chunks)} chunks")
                        
                        # STEP 3: Create vector store (embeddings)
                        st.write("Creating embeddings...")
                        vector_store = get_vector_store(text_chunks)
                        
                        # STEP 4: Save to session state if successful
                        if vector_store:
                            st.session_state.vector_store = vector_store  # Store vector DB
                            st.session_state.processed = True  # Mark as processed
                            st.success(f"Successfully processed {len(pdf_docs)} PDFs!")
                        else:
                            st.error("Failed to create vector store!")
                            
                    except Exception as e:
                        # Handle any processing errors gracefully
                        st.error(f"Processing error: {str(e)}")
            else:
                # User clicked process without uploading files
                st.warning("Please upload PDF files first!")

        # CLEAR PDFs BUTTON
        if st.button("Clear PDFs"):
            # Reset session state to clear processed documents
            st.session_state.vector_store = None
            st.session_state.processed = False
            st.success("PDFs cleared!")

        # MODE INDICATOR
        st.write("---")  # Visual separator
        
        # Show current operation mode to user
        if st.session_state.processed and st.session_state.vector_store:
            st.success("📄 Document Q&A Mode Active")  # RAG mode
        else:
            st.info("💬 Normal Chat Mode")  # Standard chat mode

    # MAIN CHAT INTERFACE
    # Text input for user questions
    user_question = st.text_input("Ask your question:", key="user_input")

    # PROCESS USER QUESTION
    if user_question and user_question.strip():  # Check if user entered a question
        
        with st.spinner("Thinking..."):  # Show thinking indicator
            try:
                # DECISION POINT: RAG vs Normal Chat
                if st.session_state.processed and st.session_state.vector_store:
                    
                    # RAG MODE - Question Answering from Documents
                    st.write("🔍 Searching documents...")  # Progress indicator
                    
                    # Get the cached language model
                    llm = get_llm()
                    
                    # Create retriever from vector store
                    # k=3 means retrieve top 3 most similar document chunks
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    
                    # Create RetrievalQA chain
                    # This combines document retrieval with question answering
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,  # Language model for generating answers
                        chain_type="stuff",  # Strategy: stuff all context into one prompt
                        retriever=retriever,  # Document retriever
                        return_source_documents=False  # Don't return source docs (just the answer)
                    )
                    
                    # Execute the RAG pipeline
                    result = qa_chain.invoke({"query": user_question})
                    answer = result["result"]  # Extract the generated answer
                    
                    # Display the answer
                    st.write("**📄 Answer (from documents):**")
                    st.write(answer)
                    
                else:
                    
                    # NORMAL CHAT MODE - General conversation
                    st.write("💬 Normal chat response...")  # Progress indicator
                    
                    # Get the cached language model
                    llm = get_llm()
                    
                    # Invoke the model directly with the user question
                    response = llm.invoke(user_question)
                    
                    # Display the response
                    st.write("**🤖 Answer:**")
                    st.write(response.content)  # Access content attribute of response
                    
            except Exception as e:
                # Handle any errors during question processing
                st.error(f"Error: {str(e)}")
                st.write("Please check if Ollama is running and the model is available.")

# APPLICATION ENTRY POINT
if __name__ == "__main__":
    main()  # Run the main application function
