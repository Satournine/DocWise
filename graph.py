from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Union
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from chains import get_summary_chain, get_qa_chain
from utils import load_document, chunk_text
import os
from typing import Callable

class DocAgentState(TypedDict):
    file_path: str
    messages: Annotated[List[BaseMessage], add_messages] #chat hist
    document_chunks: List[str] #doc chunks
    summary: str #generated sum
    query: str #latest user input
    answer: str #answer to query


def parse_and_chunk_document(state: DocAgentState) -> DocAgentState:
    file_path = state.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise ValueError("File path missing or file does not exist.")
    
    text = load_document(file_path)
    chunks = chunk_text(text)

    return{
        **state,
        "document_chunks": chunks,
    }
def summarize_document(state: DocAgentState) -> DocAgentState:
    chunks = state.get("document_chunks", [])
    if not chunks:
        raise ValueError("No document chunks to summarize.")

    text = "\n".join(chunks[:10])
    chain = get_summary_chain()
    summary = chain.run({"document": text})

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
    answer = qa.invoke({"input": query})

    return {
        **state,
        "answer": answer,
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