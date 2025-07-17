import pymupdf
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        doc = pymupdf.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")
    
def chunk_text(text: str, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)