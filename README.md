# Retrieval-Augmented Generation (RAG) Chat Application
### LangGraph · LangChain · Chroma · Streamlit

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to upload `.txt` and `.pdf` documents, store them in a vector database, and interact with their content through a conversational chat interface.

The system retrieves the most relevant text passages from uploaded files and uses a large language model (LLM) to generate context-grounded answers.  
It combines:

- **LangChain** — for document loading, text splitting, embedding, and retrieval  
- **LangGraph** — for orchestration of the retrieval-generation pipeline  
- **Chroma** — as a persistent local vector database  
- **Streamlit** — for an interactive, browser-based chat user interface  
- **OpenAI API (Cornell proxy)** — for embeddings and response generation

Once a document is uploaded, it is parsed, chunked, and embedded **only once**. All embeddings are stored persistently so re-running the app or sending new chat messages does **not** reprocess documents.

---

## eatures

| Feature | Description |
|----------|-------------|
| **Multi-file upload** | Upload multiple `.txt` and `.pdf` documents simultaneously. |
| **One-time processing** | Each document is parsed, chunked, and embedded only once using content-hash caching. |
| **Persistent ChromaDB** | All embeddings are stored locally (`.chroma_*`) and automatically reused across runs. |
| **Context-grounded answers** | Model answers are based solely on retrieved document content. |
| **File-based citation** | Assistant responses include the name(s) of the document(s) used to answer. |
| **LangGraph orchestration** | Simple state graph `{START → retrieve → generate → END}` ensures modularity and clarity. |
| **Backend logging** | Each question, retrieval result, and generated prompt is printed to the terminal for debugging and transparency. |

---

## Running the Application

### Step 1: Environment Setup
If you are running locally instead of in Codespaces:

```bash
pip install -r requirements.txt
```
### Step 2: API Configuration
Set your Cornell OpenAI proxy credentials before starting the application:
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.ai.it.cornell.edu"
```
### Step 3: Launch App 
   ```bash
   streamlit run app.py
   ```  
### Step 4: Upload and Chat
1. Upload one or more .txt or .pdf files.

2. Ask questions about their content.

The assistant retrieves relevant passages and generates a concise, grounded answer.
At the end of the answer, it also lists which uploaded file(s) the information came from.







