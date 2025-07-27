from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from langchain.schema import BaseMessage
from chains import get_summary_chain, get_qa_chain
from utils import load_document, chunk_text
import os
from typing import Callable
from langchain_core.documents import Document

class DocAgentState(TypedDict):
    file_paths: List[str]
    messages: Annotated[List[BaseMessage], add_messages] #chat hist
    document_chunks: List[str] #doc chunks
    summary: str #generated sum
    query: str #latest user input
    answer: str #answer to query


def parse_and_chunk_document(state: DocAgentState) -> DocAgentState:
    file_paths = state.get("file_paths")
    if not file_paths:
        raise ValueError("File path missing or file does not exist.")
    
    all_chunks = [] #to merge everything
    for path in file_paths:
        if os.path.exists(path):
            text = load_document(path)
            chunks = chunk_text(text, file_name=os.path.basename(path))
            docs = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]
            all_chunks.extend(docs)
    return{
        **state,
        "document_chunks": all_chunks,
    }
def summarize_document(state: DocAgentState) -> DocAgentState:
    chunks = state.get("document_chunks", [])
    if not chunks:
        raise ValueError("No document chunks to summarize.")

    text = "\n".join([doc.page_content for doc in chunks[:10]])
    chain = get_summary_chain()
    summary = chain.invoke({"document": text})

    return {
        **state,
        "summary": summary,
    }
def answer_query(state: DocAgentState) -> DocAgentState:
    query = state.get("query")
    chunks = state.get("document_chunks")

    if not query or not chunks:
        raise ValueError("Missing query or document.")

    qa = get_qa_chain(chunks)
    result = qa.invoke({"input": query})
    answer_text = result["answer"]
    source_files = list({doc.metadata.get("source") for doc in result.get("context", []) if doc.metadata.get("source")})

    return {
        **state,
        "answer": f"{answer_text}\n\n Sources:{', '.join(source_files)}" if source_files else answer_text,
    }


builder = StateGraph(DocAgentState)

builder.add_node("parse_doc", parse_and_chunk_document)
builder.add_node("summarize", summarize_document)


builder.set_entry_point("parse_doc")
builder.add_edge("parse_doc", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

def get_graph():
    return graph