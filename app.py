import os
import io
from typing import List, Dict, Any, Optional, TypedDict

import streamlit as st
from pypdf import PdfReader

# LangChain (models + splitters + chroma)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# LangGraph
from langgraph.graph import StateGraph, START, END


# ---------------------------
# App/UI config
# ---------------------------
st.set_page_config(page_title="RAG ChatBot", layout="wide")

COLLECTION_NAME = "rag_collection"
DEFAULT_MODEL_NAME = "openai.gpt-4o"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 0
DEFAULT_TOP_K = 10


if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://api.ai.it.cornell.edu"


# ---------------------------
# Helpers: load & chunk
# ---------------------------

def read_txt(name: str, data: bytes) -> List[Document]:
    text = data.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"doc_name": name, "page": 1})]


def read_pdf(name: str, data: bytes) -> List[Document]:
    out: List[Document] = []
    reader = PdfReader(io.BytesIO(data))
    for i, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            out.append(Document(page_content=t, metadata={"doc_name": name, "page": i}))
    return out


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ". ", "? ", "! ", "\n", " ", ""],  # prefer sentence-ish cuts
    )
    chunks = splitter.split_documents(docs)
    counters = {}
    for d in chunks:
        key = (d.metadata.get("doc_name"), d.metadata.get("page"))
        i = counters.get(key, 0)
        d.metadata["chunk"] = i
        counters[key] = i + 1

    return chunks


# ---------------------------
# Embeddings (cached) + in-memory vectorstore in session
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model="openai.text-embedding-3-large")


# Initialize session state
if "vs" not in st.session_state:
    st.session_state.vs = None  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "doc_names" not in st.session_state:
    st.session_state.doc_names = [] 


# ---------------------------
# RAG graph nodes
# ---------------------------
class RAGState(TypedDict):
    question: str
    chat_history: List[Dict[str, str]]  # [{role, content}]
    retrieved: List[Document]
    answer: str


def node_retrieve(state: RAGState, vs: Chroma, k: int, doc_filter: Optional[List[str]]):
    # Prefer diverse results
    search_kwargs: Dict[str, Any] = {"k": k}
    if doc_filter:
        search_kwargs["filter"] = {"doc_name": {"$in": doc_filter}}

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={**search_kwargs, "lambda_mult": 0.5},  # 0..1, lower = more diversity
    )
    docs = retriever.invoke(state["question"])

    unique, seen = [], set()
    for d in docs:
        key = (d.metadata.get("doc_name"), d.metadata.get("page"), d.page_content.strip()[:200])
        if key not in seen:
            seen.add(key)
            unique.append(d)
    docs = unique

    print("\n=== RETRIEVE ===")
    print("Question:", state["question"])
    print("Filter docs:", doc_filter)
    print("Top-k retrieved:")
    for idx, d in enumerate(docs, 1):
        md = d.metadata or {}
        snippet = d.page_content.strip().replace("\n", " ")
        print(f"{idx}. {md.get('doc_name')} (page {md.get('page')}, chunk {md.get('chunk','?')})")
        print("   ->", snippet[:180], "…")
    print("==============\n")

    return {**state, "retrieved": docs}



def _context_from_docs(docs: List[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = f"{meta.get('doc_name','?')} (p.{meta.get('page','?')})"
        txt = d.page_content.strip()
        blocks.append(f"[Source {i}: {src}]\n{txt}")
    return "\n\n".join(blocks)


def node_generate(state: RAGState, model_name: str):
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    context = _context_from_docs(state.get("retrieved", []))
    system_instructions = (
        "Use ONLY the provided context to answer concisely (<=3 sentences).\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    messages = [
        {"role": "system", "content": system_instructions},
        *state.get("chat_history", []),
        {"role": "user", "content": state["question"]},
    ]

    # --- Debug log to terminal: print the full prompt/messages
    print("\n=== GENERATE ===")
    print(f"Model: {model_name}")
    print("Messages:")
    for m in messages:
        role = m.get("role", "?")
        content = (m.get("content", "") or "").strip()
        print(f"- {role.upper()}: {content}")

    resp = llm.invoke(messages)
    return {**state, "answer": resp.content}


def build_graph(vs: Optional[Chroma], k: int, model_name: str, doc_filter: Optional[List[str]]):
    g = StateGraph(RAGState)

    def _retrieve(s: RAGState):
        return node_retrieve(s, vs, k, doc_filter)

    def _generate(s: RAGState):
        return node_generate(s, model_name)

    g.add_node("retrieve", _retrieve)
    g.add_node("generate", _generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g.compile()


# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <h1 style="color:#0e76a8;">Welcome to the <b>Retrieval-Augmented Chatbot!</b></h1>

    <ol style="font-size:16px; line-height:1.6;">
        <li>Upload one or more <code>.txt</code> or <code>.pdf</code> files below, and use the <i>“Limit retrieval to”</i> selector to choose which files to search.</li>
        <li>Type your question in the chat box at the bottom, and wait for the assistant to respond.</li>
    </ol>

    <p style="font-size:15px; color:#555;">
    <i>Tip:</i> You can verify every answer by checking the citations panel to see the exact text the model referenced.
    </p>
    <hr style="margin-top:25px; margin-bottom:10px;">
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Upload & index (in-memory)
# ---------------------------
embeddings = get_embeddings()

uploaded = st.file_uploader(
    "Upload .txt and/or .pdf (multiple files supported)",
    type=("txt", "pdf"),
    accept_multiple_files=True
)

if uploaded:
    all_docs: List[Document] = []
    with st.status("Indexing…", expanded=True) as status:
        for f in uploaded:
            name = f.name
            data = f.getvalue()
            st.write(f"Processing **{name}** …")
            try:
                if name.lower().endswith(".txt"):
                    docs = read_txt(name, data)
                else:
                    docs = read_pdf(name, data)
                all_docs.extend(docs)
                st.session_state.doc_names.append(name)
            except Exception as e:
                st.error(f"Failed to read {name}: {e}")

        status.update(label="Chunking…")
        chunks = chunk_documents(all_docs, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
        for chunk in chunks:
          print(chunk.page_content)
          print("-----")

        status.update(label=f"Adding {len(chunks)} chunks to Chroma (in-memory)…")
        try:
            if chunks:
                if st.session_state.vs is None:
                    st.session_state.vs = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        collection_name=COLLECTION_NAME,
                    )
                else:
                    st.session_state.vs.add_documents(chunks)
            else:
                st.info("No text extracted to index.")
        except Exception as e:
            st.error(f"Vector store add failed: {e}")

        status.update(label="Done", state="complete")


# ---------------------------
# Doc filter UI
# ---------------------------
known_docs = sorted(set(st.session_state.doc_names))
st.markdown(
    """
    **Limit retrieval to:**  
    Select which uploaded document(s) you want the assistant to search for answers.  
    By default, all uploaded files are included.  
    If you only want the model to look inside a specific document, uncheck the others.
    """
)
selected_docs = st.multiselect(
    label="",
    options=known_docs,
    default=known_docs,
    placeholder="Select document(s) to include in retrieval",
)


# ---------------------------
# Build the LangGraph app (compile) so it respects UI settings
# ---------------------------
app = build_graph(vs=st.session_state.vs, k=DEFAULT_TOP_K, model_name=DEFAULT_MODEL_NAME, doc_filter=selected_docs)


# ---------------------------
# Chat area
# ---------------------------
st.markdown(
    """
    <hr style="border:1px solid #ccc; margin-top:50px; margin-bottom:50px;">
    <h1 style="color:#0e76a8;">Chat with your uploaded documents</h1>
    <p style="font-size:15px; color:#444;">
    Ask questions below. The assistant will analyze your uploaded files and respond based on their content.
    </p>
    """,
    unsafe_allow_html=True
)

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your documents…")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    init_state: RAGState = {
        "question": question,
        "chat_history": st.session_state.chat_history,
        "retrieved": [],
        "answer": "",
    }

    with st.spinner("Thinking…"):
        final_state = app.invoke(init_state)
        answer = final_state.get("answer", "(no answer)")
        docs = final_state.get("retrieved", [])

    with st.chat_message("assistant"):
        st.markdown(answer)
        # Show explicit sources: file names and pages (not generic citations)
        with st.expander("Sources / citations"):
            if not docs:
                st.write("No sources retrieved.")
            for i, d in enumerate(docs, 1):
                meta = d.metadata or {}
                st.markdown(f"**{i}.** `{meta.get('doc_name','?')}` — page {meta.get('page','?')}")
                preview = d.page_content
                st.write(preview)
                st.divider()

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
