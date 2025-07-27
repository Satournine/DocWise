from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import streamlit as st
import os


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

def get_summary_chain():
    prompt = PromptTemplate(
        input_variables = ["document"],
        template="""
        Summarize the following document as clearly and concisely as possible:

        {document}
        """
    )

    model = init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
    )

    return prompt | model | StrOutputParser()

def get_qa_chain(chunks: list[str]):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    dense_retriever = vectorstore.as_retriever(search_type = "similarity", k=4)

    sparse_retriever = BM25Retriever.from_texts([doc.page_content for doc in chunks])
    sparse_retriever.k = 4

    retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5],
    )
    
    model = init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer the questions. Cite the source (metadata 'source') of each piece of information inline, like [source.txt]. If there is only one source cite it at the end of the paragraph."),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ])

    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=(prompt | model | StrOutputParser())
    )

    return qa_chain