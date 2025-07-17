from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Union
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from utils import load_document, chunk_text
import os

class DocAgentState(TypedDict):
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