import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool

def build_hr_agent(uploaded_file, groq_key: str, jina_key: str):
    """
    Processes an uploaded PDF and returns a configured LangChain HR Agent.
    """
    # 1. Save uploaded file to a temporary file for PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 2. Load and split PDF content
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)

        # 3. Create embeddings & vector store
        embedding_model = JinaEmbeddings(
            jina_api_key=jina_key, 
            model_name="jina-embeddings-v2-base-en"
        )
        vector_store = FAISS.from_documents(chunks, embedding_model)

        # 4. Create retrieval tool
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        search_hr_policy_tool = create_retriever_tool(
            retriever=retriever,
            name="search_hr_policy_tool",
            description="Searches and returns facts from the uploaded company HR policy PDF."
        )

        # 5. Initialize LLM and Agent
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            groq_api_key=groq_key
        )

        system_prompt = """YOU ARE A FRIENDLY HR ASSISTANT.
ALWAYS USE THE search_hr_policy_tool to lookup facts before answering. 
If the fact or answer isn't in the search results, say you don't know instead of guessing."""

        agent = create_agent(
            model=llm,
            tools=[search_hr_policy_tool],
            system_prompt=system_prompt
        )

        return agent

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
