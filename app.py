import streamlit as st
import os
from dotenv import load_dotenv

# Try to load secrets from Streamlit Cloud, then fall back to local .env
if "AWS_ACCESS_KEY_ID" in st.secrets:
    os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_DEFAULT_REGION"] = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
    os.environ["COHERE_API_KEY"] = st.secrets.get("COHERE_API_KEY", "")
else:
    load_dotenv("tools/.env")

st.set_page_config(page_title="ABB Product Assistant", page_icon="⚡", layout="wide")

# CSS for a more ABB-like look (ABB uses red/white/gray)
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main-title {
        color: #ff0000;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff0000;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

from config.models import get_llm, get_embeddings
from rag.retriever import get_or_create_vector_store
from rag.ingest import load_documents, split_documents
from tools.rag_tools import get_retriever_tool
from agents.faq_agent import build_agent
from agents.discovery_agent import guided_questions

# 1. Initialize Backend
@st.cache_resource
def init_backend():
    embeddings = get_embeddings()
    index_path = "faiss_index"
    
    if os.path.exists(index_path):
        with st.status("Loading knowledge base...", expanded=False):
            vector_store = get_or_create_vector_store(embeddings)
    else:
        with st.status("Building knowledge base (first time setup)...", expanded=True) as status:
            st.write("Loading documents from ABB catalogs...")
            docs = load_documents()
            st.write("Processing text segments...")
            splits = split_documents(docs)
            st.write("Generating embeddings and building index...")
            vector_store = get_or_create_vector_store(embeddings, splits)
            status.update(label="Index built successfully!", state="complete", expanded=False)
            
    tool = get_retriever_tool(vector_store)
    llm = get_llm()
    agent = build_agent(llm, [tool])
    return agent

try:
    agent = init_backend()
except Exception as e:
    st.error(f"Failed to initialize backend: {e}")
    st.stop()

# 2. State Management
if "mode" not in st.session_state:
    st.session_state.mode = "initial"  # "initial", "chat", "discovery"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "discovery_step" not in st.session_state:
    st.session_state.discovery_step = 0
if "discovery_answers" not in st.session_state:
    st.session_state.discovery_answers = []

# 3. Main UI Logic
st.markdown('<h1 class="main-title">⚡ ABB Product Assistant</h1>', unsafe_allow_html=True)
st.subheader("Your expert guide for ABB catalogs and induction motors")

# Initial View: Selection Buttons
if st.session_state.mode == "initial":
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💬 Ask about Product/Category"):
            st.session_state.mode = "chat"
            st.rerun()
            
    with col2:
        if st.button("🔍 Guide me to the right product (Guided Discovery)"):
            st.session_state.mode = "discovery"
            st.session_state.discovery_step = 0
            st.session_state.discovery_answers = []
            st.rerun()

# --- Mode: Chat ---
elif st.session_state.mode == "chat":
    if st.button("⬅️ Back to Home"):
        st.session_state.mode = "initial"
        st.rerun()
        
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about ABB products (e.g. 'What are high voltage induction motors?')"):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": query}]})
                answer = response["messages"][-1].content
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Mode: Discovery ---
elif st.session_state.mode == "discovery":
    if st.button("⬅️ Back to Home"):
        st.session_state.mode = "initial"
        st.rerun()
        
    questions = guided_questions()
    
    if st.session_state.discovery_step < len(questions):
        st.progress((st.session_state.discovery_step) / len(questions))
        current_question = questions[st.session_state.discovery_step]
        
        with st.container():
            st.write(f"### Question {st.session_state.discovery_step + 1}:")
            st.info(current_question)
            
            answer = st.text_input("Your Answer:", key=f"q_{st.session_state.discovery_step}")
            
            if st.button("Next ➡️"):
                if answer:
                    st.session_state.discovery_answers.append(f"Q: {current_question} A: {answer}")
                    st.session_state.discovery_step += 1
                    st.rerun()
                else:
                    st.warning("Please provide an answer before proceeding.")
    else:
        st.success("Analysis complete! Recommending products based on your needs...")
        st.write("---")
        
        # Build the discovery prompt for the agent
        discovery_context = "\n".join(st.session_state.discovery_answers)
        final_query = f"Based on the following user requirements for a product discovery, recommend the most suitable ABB products from your catalogues:\n\n{discovery_context}"
        
        with st.spinner("Searching for the best match..."):
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": final_query}]})
                st.markdown(response["messages"][-1].content)
                
                if st.button("Start New Discovery 🔄"):
                    st.session_state.discovery_step = 0
                    st.session_state.discovery_answers = []
                    st.rerun()
            except Exception as e:
                st.error(f"Recommendation error: {e}")
