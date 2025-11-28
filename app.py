# app.py
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_experimental.chat_models import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Streamlit Page ---
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄")
st.title("📄 PDF RAG Assistant")
st.markdown(
    "Upload a PDF and ask questions about it. "
    "The assistant provides practical suggestions and insights using a RAG pipeline."
)

# --- Set OpenAI API Key from Streamlit secrets ---
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Please set your OpenAI API key in Streamlit Secrets!")
    st.stop()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    st.info("Processing PDF... This may take a few seconds.")

    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_file_path = tmp_file.name

    # Load & split PDF
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

    # Initialize Chroma vector DB
    if "db" not in st.session_state:
        st.session_state["db"] = Chroma.from_documents(splits, OpenAIEmbeddings())

    db = st.session_state["db"]

    # Define prompt template
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant. Use the context below to answer the question as accurately and concisely as possible.
If the answer is not in the context, say "I could not find the answer in the document."

Context:
{context}

Question: {question}
Answer:"""
)

    # Initialize QA chain
    if "qa" not in st.session_state:
        st.session_state["qa"] = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            retriever=db.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

    qa = st.session_state["qa"]

    # --- User Question ---
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Generating answer..."):
            answer = qa.run(question)
            st.markdown(f"**Answer:** {answer}")
