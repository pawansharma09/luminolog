import streamlit as st
import google.generativeai as genai

# --- CONFIGURATION ---
st.set_page_config(page_title="Luminalog | AI Agent Forensic", layout="wide")
API_KEY = 'YOUR_API_KEY' # Replace with your key
genai.configure(api_key=API_KEY)

# --- SYSTEM PROMPT ---
SYSTEM_INSTRUCTION = """
You are Luminalog, an expert AI Log Analyst. 
1. Your first task is to explain 'The Why': Based on the log, explain the agent's logic and actions.
2. For follow-up questions:
   - If the answer is in the log, cite the log.
   - If the user asks something NOT in the log, start your sentence with: 
     "This is not mentioned in the document, but based on my general knowledge..."
3. Keep the tone technical yet insightful.
"""

def get_gemini_response(contents, prompt, model_name):
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_INSTRUCTION
    )
    # Combine the document context with the user question
    full_prompt = f"CONTEXT DOCUMENT/LOGS:\n{contents}\n\nUSER QUESTION: {prompt}"
    response = model.generate_content(full_prompt)
    return response.text

# --- UI DESIGN ---
st.title("🔦 Luminalog")
st.subheader("Trace the logic behind the logs.")

# Sidebar for Model Selection
with st.sidebar:
    st.info("Select the model you verified earlier.")
    model_choice = st.selectbox("Engine", ["gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"])
    st.divider()
    st.caption("Upload raw agent logs (.txt, .log) or paste them directly.")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Agent Log File", type=['txt', 'log'])
    
with col2:
    pasted_log = st.text_area("Or Paste Log Here", height=150)

# Extract content
log_content = ""
if uploaded_file:
    log_content = uploaded_file.read().decode("utf-8")
elif pasted_log:
    log_content = pasted_log

# --- ANALYSIS SECTION ---
if log_content:
    if "analysis_done" not in st.session_state:
        with st.spinner("Analyzing Agent Intent..."):
            initial_summary = get_gemini_response(log_content, "Explain what happened in these logs and why the agent took these actions.", model_choice)
            st.session_state.analysis_done = initial_summary
    
    st.markdown("### 🔍 Initial Forensic Analysis")
    st.write(st.session_state.analysis_done)
    
    st.divider()
    
    # Chat Interface for follow-up
    query = st.text_input("Ask a specific question about these logs:")
    if query:
        with st.spinner("Consulting Log History..."):
            answer = get_gemini_response(log_content, query, model_choice)
            st.markdown(f"**Response:**\n{answer}")
else:
    st.warning("Please provide a log file or paste text to begin.")
