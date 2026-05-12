import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline


embed_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = pipeline("text2text-generation", model="google/flan-t5-small")


# ── 1. Load PDF 
def load_pdf(file_path: str) -> tuple[str, int]:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text, total_pages


# ── 2. Chunk text
def split_by_day(text: str) -> list[str]:
    chunks, current = [], ""
    for line in text.split("\n"):
        if line.strip().startswith("Day"):
            if current:
                chunks.append(current.strip())
            current = line + "\n"
        else:
            current += " " + line
    if current:
        chunks.append(current.strip())
    return chunks


def build_chunks(pdf_text: str, total_pages: int) -> list[str]:
    chunks = split_by_day(pdf_text)
    chunks.append(f"This document contains {total_pages} pages.")
    chunks.append(
        "This PDF is a 30-day manual testing guide covering topics like "
        "SDLC, Agile, Smoke testing, Sanity testing, and daily practice tasks."
    )
    return chunks


# ── 3. Build FAISS index 
def build_index(chunks: list[str]):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings


# ── 4. Answer a question 
def ask_pdf(question: str, chunks: list[str], index, total_pages: int) -> str:
    if "how many pages" in question.lower():
        return f"The document contains {total_pages} pages."

    # exact Day-N match
    words = question.lower().split()
    for word in words:
        if word.isdigit():
            for chunk in chunks:
                if f"Day {word}" in chunk:
                    return chunk

    # vector search → LLM
    q_emb = embed_model.encode([question])
    _, I = index.search(np.array(q_emb), k=3)
    context = "\n".join(chunks[i] for i in I[0])

    prompt = (
        "Answer ONLY from the provided document context."

        "If the answer is not present in the document, say: I could not find the answer in the document."
        "Answer the question based on the document below.\n\n"
        f"Document:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly:"
        
    )
    result = llm(prompt, max_new_tokens=150)
    return result[0]["generated_text"]

