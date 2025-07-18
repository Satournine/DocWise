import streamlit as st
import os
from graph import get_graph
from chains import get_qa_chain
from langchain.schema import HumanMessage
import tempfile

st.set_page_config(page_title="DocWise AI", layout="centered")
st.title("📄 DocWise: Smart Document Assistant")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# File upload
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
if uploaded_file:
    upload_dir = "data"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded")

    # Run document setup graph (parse + summarize)
    graph = get_graph()
    state = {
        "file_path": file_path,
        "query": "",
        "messages": [],
    }
    result = graph.invoke(state)

    st.session_state.document_chunks = result.get("document_chunks", [])
    st.session_state.summary = result.get("summary", "")

    st.success("✅ Document processed")

# Show summary if available
if st.session_state.summary:
    with st.expander("📌 Summary"):
        st.write(st.session_state.summary)

# User input for Q&A
user_input = st.text_input("Ask a question about the document:")
if st.button("Ask") and user_input:
    if not st.session_state.document_chunks:
        st.warning("Please upload and process a document first.")
    else:
        qa_chain = get_qa_chain(st.session_state.document_chunks)
        answer = qa_chain.invoke({"input": user_input})

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("DocWise", answer["answer"]))

# Chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🧠 Chat History")
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")