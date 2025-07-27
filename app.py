import streamlit as st
import os
from graph import get_graph
from chains import get_qa_chain
import streamlit as st
from streamlit.components.v1 import html

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
uploaded_files = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"], accept_multiple_files=True)
if uploaded_files:
    upload_dir = "data"
    os.makedirs(upload_dir, exist_ok=True)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        file_paths.append(file_path)

    st.success("✅ File uploaded")

    # Run document setup graph (parse + summarize)
    graph = get_graph()
    state = {
        "file_paths": file_paths,
        "query": "",
        "messages": [],
    }
    if uploaded_files and not st.session_state.summary:

        result = graph.invoke(state)

        st.session_state.document_chunks = result.get("document_chunks", [])
        st.session_state.summary = result.get("summary", "")

    st.success("✅ Document processed")

all_sources = list({doc.metadata["source"] for doc in st.session_state.document_chunks})
selected_source = st.selectbox("Choose document to query:", ["All Documents"] + sorted(all_sources))

# Show summary if available
if st.session_state.summary:
    with st.expander("📌 Summary"):
        st.write(st.session_state.summary)

# Chat history (scrollable)
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🧠 Chat History")
    st.markdown(
        """
        <div style='max-height: 400px; overflow-y: auto; padding-right:8px;'>
        """,
        unsafe_allow_html=True
    )
    for role, msg in st.session_state.chat_history:
        icon = "🧑‍💻" if role == "You" else "🤖"
        bubble_color = "#634DB2" if role == "You" else "#8371C1"
        st.markdown(
            f"""
            <div style='border-radius:8px; padding:10px; margin:8px 0; background-color:{bubble_color};'>
            <b>{icon} {role}</b><br>{msg}
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

# User input for Q&A at the bottom with Enter-to-send
user_input = st.chat_input("Ask something about the document...")
if user_input:
    if not st.session_state.document_chunks:
        st.warning("Please upload and process a document first.")
    else:
        # Add user message immediately
        st.session_state.chat_history.append(("You", user_input))
        # Add placeholder bot response
        st.session_state.chat_history.append(("DocWise", "✍️ *DocWise is thinking...*"))
        st.rerun()  # Refresh immediately to show both

# After rerun: If last message is placeholder, replace it with real answer
if st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg[0] == "DocWise" and "DocWise is thinking" in last_msg[1]:
        if selected_source != "All Documents":
            filtered_chunks = [doc for doc in st.session_state.document_chunks if doc.metadata["source"] == selected_source]
        else:
            filtered_chunks = st.session_state.document_chunks

        qa_chain = get_qa_chain(filtered_chunks)
        answer = qa_chain.invoke({"input": st.session_state.chat_history[-2][1]})

        # Replace placeholder with real answer
        st.session_state.chat_history[-1] = ("DocWise", answer["answer"])
        st.rerun()