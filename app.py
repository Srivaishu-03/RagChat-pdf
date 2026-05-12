import os
import tempfile
import streamlit as st
from rag2 import load_pdf, build_chunks, build_index, ask_pdf

st.set_page_config(page_title="PDF RAG Chat")
st.title("PDF RAG Chat")

# session state
if "chunks" not in st.session_state:
    st.session_state.chunks = None
    st.session_state.index = None
    st.session_state.total_pages = 0
    st.session_state.history = []

# upload pdf
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded and st.button("Process PDF"):
    with st.spinner("Extracting text and building index..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        pdf_text, total_pages = load_pdf(tmp_path)
        os.unlink(tmp_path)

        chunks = build_chunks(pdf_text, total_pages)
        index, _ = build_index(chunks)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.total_pages = total_pages
        st.session_state.history = []

    st.success(f"Ready! {total_pages} pages · {len(chunks)} chunks")

# chat history
for role, text in st.session_state.history:
    st.chat_message(role).write(text)

# chat input
question = st.chat_input("Ask something about the PDF...")
if question:
    if not st.session_state.chunks:
        st.warning("Please upload and process a PDF first.")
    else:
        st.chat_message("user").write(question)
        with st.spinner("Thinking..."):
            answer = ask_pdf(question, st.session_state.chunks,
                             st.session_state.index, st.session_state.total_pages)
        st.chat_message("assistant").write(answer)
        st.session_state.history += [("user", question), ("assistant", answer)]
        st.rerun()
