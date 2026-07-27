import os
import streamlit as st
from dotenv import load_dotenv

# Import the builder function from rag.py
from rag import build_hr_agent

load_dotenv()

st.set_page_config(page_title="HR Policy Assistant", page_icon="🤖", layout="wide")

st.title("🤖 HR Policy Assistant")
st.write("Upload an HR Policy PDF (up to 200 MB) to ask questions.")

# Sidebar Configuration
with st.sidebar:
    st.header("1. Setup & Upload")
    
    groq_key = os.getenv("GROQ_API_KEY")
    jina_key = os.getenv("JINA_API_KEY")

    if not groq_key or not jina_key:
        st.warning("⚠️ API keys missing in `.env`. Enter them below:")
        groq_key = st.text_input("GROQ API Key", value=groq_key or "", type="password")
        jina_key = st.text_input("JINA API Key", value=jina_key or "", type="password")

    uploaded_file = st.file_uploader("Upload Policy PDF", type=["pdf"])
    process_btn = st.button("Process Document")

# Session State Setup
if "hr_assistant" not in st.session_state:
    st.session_state.hr_assistant = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF click trigger
if process_btn:
    if uploaded_file and groq_key and jina_key:
        with st.spinner("Processing PDF document... Please wait."):
            st.session_state.hr_assistant = build_hr_agent(uploaded_file, groq_key, jina_key)
            st.session_state.messages = []  # Reset chat history for new file
            st.sidebar.success("✅ Policy successfully processed!")
    else:
        st.sidebar.error("Please provide API Keys and upload a PDF first.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Chat Input
if prompt := st.chat_input("Ask a question about the HR policy..."):
    if not st.session_state.hr_assistant:
        st.error("Please upload a PDF and click 'Process Document' first.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent invocation
        with st.chat_message("assistant"):
            with st.spinner("Searching policy..."):
                response = st.session_state.hr_assistant.invoke({
                    "messages": [("user", prompt)]
                })
                reply_content = response["messages"][-1].content
                st.markdown(reply_content)

        # Assistant message
        st.session_state.messages.append({"role": "assistant", "content": reply_content})