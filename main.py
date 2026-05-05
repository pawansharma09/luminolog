import streamlit as st
import PyPDF2
import docx
import os
import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(page_title="DocuMind AI (No LangChain)", page_icon="🧠", layout="wide")
st.title("🧠 DocuMind: Talk to your Documents")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunk_metadata" not in st.session_state:
    st.session_state.chunk_metadata = []

# --- Load Embedding Model ---
# Using st.cache_resource so the model isn't reloaded on every UI click
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# --- Helper Functions ---

def chunk_text(text, source, page, chunk_size=1000, overlap=200):
    """Custom text chunker to replace LangChain's RecursiveCharacterTextSplitter."""
    chunks = []
    if not text or not text.strip():
        return chunks
        
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append({
            "text": chunk,
            "source": source,
            "page": page
        })
        # Break if we've reached the end
        if end >= text_len:
            break
        # Move forward, stepping back by the overlap amount
        start = end - overlap
        
    return chunks

def process_uploaded_files(uploaded_files):
    """Extracts text, metadata, and chunks from PDF, DOCX, and TXT files."""
    all_chunks = []
    
    for file in uploaded_files:
        filename = file.name
        
        # Process PDF
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    all_chunks.extend(chunk_text(text, filename, i + 1))
                    
        # Process DOCX
        elif filename.endswith('.docx'):
            doc = docx.Document(file)
            text = "\n".join([para.text for para in doc.paragraphs])
            if text:
                all_chunks.extend(chunk_text(text, filename, 1))
                
        # Process TXT
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            if text:
                all_chunks.extend(chunk_text(text, filename, 1))
                
    return all_chunks

def build_faiss_index(chunks):
    """Generates embeddings and builds a raw FAISS index."""
    if not chunks:
        return None
        
    # Extract text to encode
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    embeddings = embedding_model.encode(texts)
    
    # Convert to float32 numpy array as required by FAISS
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index

def retrieve_context(query, k=4):
    """Searches the FAISS index for the most relevant chunks."""
    if not st.session_state.faiss_index or not st.session_state.chunk_metadata:
        return []
        
    # Encode query
    query_vector = embedding_model.encode([query]).astype('float32')
    
    # Search index
    distances, indices = st.session_state.faiss_index.search(query_vector, k)
    
    # Retrieve actual chunks based on indices
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(st.session_state.chunk_metadata):
            results.append(st.session_state.chunk_metadata[idx])
            
    return results

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", placeholder="Paste your API key here...")
    
    st.markdown("---")
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if not api_key:
            st.error("Please provide a Google Gemini API key first.")
        elif not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Chunking text, generating embeddings, and building FAISS index..."):
                # 1. Extract and chunk text
                chunks = process_uploaded_files(uploaded_files)
                
                if chunks:
                    # 2. Build FAISS index and store metadata in session state
                    st.session_state.chunk_metadata = chunks
                    st.session_state.faiss_index = build_faiss_index(chunks)
                    st.success(f"✅ Processed {len(chunks)} chunks into vector store!")
                else:
                    st.error("Could not extract any text from the uploaded files.")

# --- Main Chat UI ---

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Handler
if user_question := st.chat_input("Ask a question about your documents..."):
    
    if not api_key:
        st.info("Please enter your Google Gemini API key in the sidebar.")
        st.stop()
        
    if st.session_state.faiss_index is None:
        st.info("Please upload and process documents first.")
        st.stop()

    # Configure the Gemini SDK
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Add user message to UI and history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Retrieval and Generation
    with st.chat_message("assistant"):
        # 1. Retrieve relevant chunks
        retrieved_chunks = retrieve_context(user_question, k=4)
        
        # 2. Format Context
        formatted_context = ""
        for chunk in retrieved_chunks:
            formatted_context += f"[Source: {chunk['source']}, Page: {chunk['page']}]\n{chunk['text']}\n\n---\n\n"

        # 3. Construct Prompt
        prompt = f"""
        You are an expert AI assistant that answers questions based on the provided context.
        Use the following pieces of retrieved context to answer the user's question. 
        If the answer is not contained in the context, explicitly state "I don't know based on the provided documents."
        
        CRITICAL INSTRUCTION: You must include inline citations to the source document and page number right after you state a fact. 
        Format your citations exactly like this: (filename.ext - page X).
        
        Context:
        {formatted_context}
        
        Question: {user_question}
        
        Answer:
        """
        
        # 4. Stream response from Gemini
        response_placeholder = st.empty()
        full_response = ""
        
        response_stream = model.generate_content(prompt, stream=True)
        
        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                response_placeholder.markdown(full_response + "▌")
            
        # Final update without cursor
        response_placeholder.markdown(full_response)
        
        # 5. Display the expandable source list
        if retrieved_chunks:
            st.markdown("---")
            st.markdown("**🔍 Sources Used:**")
            for i, chunk in enumerate(retrieved_chunks):
                with st.expander(f"Source {i+1}: {chunk['source']} (Page {chunk['page']})"):
                    st.write(chunk['text'])

        # Add assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})   