from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv


load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = input("Enter your Groq API key: ")

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

    return LLMChain(llm=model, prompt=prompt)

def get_qa_chain(chunks: list[str]):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type = "similarity", k=4)
    
    model = init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer the questions."),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ])

    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=(prompt | model | StrOutputParser())
    )

    return qa_chain