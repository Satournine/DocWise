
# 📄 DocWise AI — Your Personal Document Intelligence Assistant

> Upload. Understand. Ask. Cite.  
> **DocWise AI** brings agentic LLM intelligence to your documents — all in one seamless interface.

![DocWise Preview](preview.gif) <!-- Add a gif or screenshot here -->

---

## 🚀 What is DocWise?

**DocWise** is a fully functional Retrieval-Augmented Generation (RAG) system, built with production-grade AI tools like **LangGraph**, **LangChain**, **FAISS**, and **BM25**, designed to:

- 🧠 **Summarize** documents instantly
- 💬 **Answer questions** based on hybrid retrieval (semantic + keyword)
- 🔍 **Cite sources** from original files
- 🧩 Support **multi-document handling**
- 🧾 Provide **per-document querying**
- ⚡ Feel like a personal assistant in a friendly Streamlit chat UI

---

## 🛠️ Features

| Feature                            | Status   | Description |
|------------------------------------|----------|-------------|
| 🔄 Multi-file Upload               | ✅       | Upload multiple PDFs or .txts simultaneously |
| 🧠 Auto-Summary Generation         | ✅       | Generates concise summaries with LLM |
| 📑 Per-Document Source Control     | ✅       | Select a document to restrict query scope |
| 🧭 Hybrid Retrieval (FAISS + BM25) | ✅       | Combines semantic & sparse search for accuracy |
| 💬 Conversational Chat Interface   | ✅       | Built with Streamlit's chat API |
| 🔐 Environment-Safe API Handling   | ✅       | Uses `.env` or `st.secrets` for keys |
| 📦 Ready for Deployment            | ✅       | Docker-ready or Streamlit Cloud deployable |

---

## 🧱 Stack

- 🧠 **LLM**: LLaMA 3 (via [Groq API](https://console.groq.com/))
- 🔗 **LangGraph**: Agentic workflow control
- 🔍 **Retrievers**: FAISS + BM25 via LangChain Ensemble
- 📄 **Parsing**: PyMuPDF & LangChain TextSplitter
- 🖼️ **Frontend**: Streamlit (custom chat UI)
- 🧪 **Vector DB**: In-memory FAISS
- 🐍 **Python**: 3.11+ (see [`requirements.txt`](./requirements.txt))

---

## 🚢 Deployment

### 1. 🔐 Setup `.env`

```bash
# .env
GROQ_API_KEY=your_groq_api_key
```


### 2. 📦 Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. ▶️ Run Locally

```bash
streamlit run app.py
```


## 📁 Directory Structure

```
docwise/
├── app.py               # Streamlit frontend
├── chains.py            # Summary + QA chain logic
├── graph.py             # LangGraph setup for parsing & summarization
├── utils.py             # Text extraction and chunking
├── requirements.txt
├── .env / st.secrets
└── data/                # Uploaded files (auto-created)
```

---

## 🤖 Why It Matters

- ✅ Built with real-world **retriever logic** (dense + sparse ensemble)
- ✅ Shows understanding of **agentic workflows**
- ✅ Demonstrates **prompt engineering & chaining**
- ✅ Includes **multi-file logic**, **metadata control**, and **citation grounding**
- ✅ Strong UI/UX polish with expandable summaries and scoped querying

