# app.py
import os, pickle
from pathlib import Path
import numpy as np
import streamlit as st
from openai import OpenAI
import faiss

# ---------- SETTINGS ----------
INDEX_PATH = Path("index.faiss")
CHUNKS_PATH = Path("chunks.pkl")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"           # fast + inexpensive, adjust as you like
TOP_K = 5                            # number of chunks to retrieve
SYSTEM_PROMPT = """You are a helpful course assistant for this class.
Answer ONLY using the provided course materials. If something is not covered,
say you don't know and suggest where it might be found in the syllabus.
Keep answers concise (5-10 sentences) and include citations to the sources."""
# --------------------------------

# Load index & chunks
@st.cache_resource
def load_resources():
    index = faiss.read_index(str(INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def embed_texts(client, texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data]).astype("float32")
    faiss.normalize_L2(vecs)
    return vecs

def retrieve(client, query, index, chunks, top_k=TOP_K):
    q_vec = embed_texts(client, [query])
    D, I = index.search(q_vec, top_k)
    results = [chunks[i] for i in I[0]]
    return results

def format_context(chunks):
    out = []
    for i, c in enumerate(chunks, 1):
        out.append(f"[{i}] ({c['source']})\n{c['text']}")
    return "\n\n".join(out)

def build_citation_footer(chunks):
    unique = []
    for c in chunks:
        tag = c["source"]
        if tag not in unique:
            unique.append(tag)
    return "Sources: " + " | ".join(unique)

def main():
    st.set_page_config(page_title="Course Q&A", page_icon="🎓")
    st.title("🎓 Course Q&A Assistant")
    st.caption("Answers grounded in your course materials.")

    # Read API key from Streamlit secrets (preferred) or env var
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error(
        "No API key found. Please either:\n"
        "1) Add OPENAI_API_KEY to Streamlit Secrets (if running on Streamlit Cloud), or\n"
        "2) Set the OPENAI_API_KEY environment variable (if running locally)."
    )
    st.stop()


    client = OpenAI(api_key=api_key)

    index, chunks = load_resources()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! Ask me a question about the course."}
        ]

    # Chat UI
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_query = st.chat_input("Type your question…")
    if user_query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieval
        docs = retrieve(client, user_query, index, chunks, TOP_K)
        context = format_context(docs)

        # Call model with context
        prompt = f"""System instructions:
{SYSTEM_PROMPT}

Use these course excerpts to answer:
{context}

User question: {user_query}
Provide a helpful answer and cite like [1], [2] etc. at the end of relevant sentences."""
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = resp.choices[0].message.content
                st.markdown(answer)
                st.caption(build_citation_footer(docs))

        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.sidebar:
        st.header("About")
        st.write(
            "- Answers are limited to your uploaded course materials.\n"
            "- If it’s not in the materials, I’ll say so.\n"
            "- Be concise and cite sources."
        )
        st.divider()
        st.subheader("Instructor controls")
        st.write("Consider adding a class password via `st.secrets` to limit access.")

if __name__ == "__main__":
    main()

