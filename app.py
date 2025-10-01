# app.py
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- Streamlit Page ---
st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄")
st.title("📄 PDF RAG Assistant")
st.markdown("Upload a PDF and ask questions about it. The assistant will provide practical suggestions and insights.")

# --- OpenAI API Key ---
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

    # --- PDF Upload ---
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if uploaded_file:
        st.info("Processing PDF... This may take a few seconds.")

        # Load and split PDF
        loader = PyPDFLoader(uploaded_file)
        docs = loader.load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

        # Create vector DB
        db = Chroma.from_documents(splits, OpenAIEmbeddings())

        # Define prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an assistant helping improve documents.
Based on the context, suggest what could be implemented, added, or improved.
Be practical and concise (max 3 suggestions).

Context:
{context}

Question: {question}
Answer:"""
        )

        # QA chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
            retriever=db.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

        # --- User Q&A ---
        question = st.text_input("Ask a question about your PDF:")
        if question:
            with st.spinner("Generating answer..."):
                answer = qa.run(question)
                st.markdown(f"**Answer:** {answer}")
