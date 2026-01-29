import faiss
import numpy as np
import time
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3
VECTOR_DIM = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_PATH = r"C:\Users\kalya\AppData\Local\nomic.ai\GPT4All\Phi-3-mini-4k-instruct.Q4_0.gguf"

vector_indexes = {}
document_chunks = {}
embedder = SentenceTransformer(EMBEDDING_MODEL)
llm = None

def init_llm():
    global llm
    if llm is None:
        print("🔄 Loading GPT4All model...")
        llm = GPT4All(MODEL_PATH)
        print("✅ GPT4All model loaded!")

def extract_text(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error: {e}")
        return ""

def chunk_text(text: str):
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def ingest_document(file_path: str, doc_id: str):
    print(f"📄 Processing document: {file_path}")
    text = extract_text(file_path)
    if not text:
        raise Exception("❌ No text extracted from PDF")
    
    chunks = chunk_text(text)
    if not chunks:
        raise Exception("❌ No chunks created from document")
    
    print(f"✂️ Creating {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(np.array(embeddings).astype("float32"))
    
    vector_indexes[doc_id] = index
    document_chunks[doc_id] = chunks
    print(f"✅ Processed {len(chunks)} chunks for doc_id: {doc_id}")

def retrieve_chunks(doc_id: str, query: str):
    if doc_id not in vector_indexes:
        print(f"❌ No document found with id: {doc_id}")
        return []
    query_embedding = embedder.encode([query])
    _, indices = vector_indexes[doc_id].search(
        np.array(query_embedding).astype("float32"), TOP_K
    )
    chunks = [document_chunks[doc_id][i] for i in indices[0]]
    print(f"🔍 Retrieved {len(chunks)} relevant chunks")
    return chunks

def generate_answer(context: str, question: str):
    init_llm()
    
    prompt = f"""Use ONLY the context below to answer the question.
If answer not in context, say "Not found in document".

Context:
{context}

Question: {question}

Answer:"""
    
    start = time.time()
    with llm.chat_session():
        response = llm.generate(prompt, max_tokens=512, temp=0.1)
    latency = (time.time() - start) * 1000
    print(f"🤖 Answer generated in {latency:.2f}ms")
    return response, latency
