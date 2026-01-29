RAG-Based Document Question Answering System
1. Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions strictly grounded in the uploaded content. The system is built with FastAPI for the backend, Streamlit for the frontend, and uses FAISS-based vector search for retrieval.

2. Key Features

Upload PDF or TXT documents
Automatic document ingestion and chunking
Vector-based semantic search using FAISS
LLM-powered answer generation using retrieved context
REST API built with FastAPI
Interactive UI using Streamlit
Latency measurement for each query

3. Tech Stack

Backend: FastAPI
Frontend: Streamlit
Vector Store: FAISS
Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
LLM: OpenAI GPT-3.5-Turbo
Language: Python

4. Project Structure

.
rag_core.py
main.py
streamlit_app.py
requirements.txt
README.md

5. System Workflow

User uploads a document
Document text is extracted
Text is chunked with overlap
Chunks are embedded and stored in FAISS
User asks a question
Relevant chunks are retrieved
LLM generates an answer using retrieved context only

6. Chunking Strategy

Chunk size: 500 characters
Chunk overlap: 100 characters

This chunk size balances semantic completeness and retrieval precision while preventing context loss across boundaries.

7. Retrieval Strategy

Both queries and document chunks are embedded using the same embedding model. FAISS performs similarity search and retrieves the top relevant chunks. Only these retrieved chunks are passed to the language model to generate grounded answers.

8. API Endpoints

Upload Document
POST /upload
Accepts PDF or TXT file
Returns confirmation message

Ask Question
POST /query

Request body
{
"question": "Your question"
}

Response
{
"answer": "Generated answer",
"latency_ms": 842.31
}

9. Metrics Tracked

Latency in milliseconds measured from retrieval start to final answer generation.

10. Setup Instructions

Install dependencies
pip install -r requirements.txt

Run backend
uvicorn main:app --reload

Run frontend
streamlit run streamlit_app.py

11. Design Constraints Followed

No default RAG templates used
Minimal file structure
Custom chunking and retrieval logic
Lightweight libraries only
Clear and explainable system design

12. Retrieval Failure Case

When a question is vague or unrelated to the uploaded document, similarity search retrieves weakly related chunks, resulting in less accurate answers. This highlights the importance of query specificity.

13. Limitations

Single-document support
Local vector storage
No authentication or user sessions

14. Future Enhancements

Multi-document querying
Hybrid retrieval using keywords and vectors
Persistent vector database
Cloud deployment

15. Conclusion

This project demonstrates a complete end-to-end RAG-based document QA system with clean API design, measurable performance metrics, and clear architectural decisions suitable for applied AI evaluations.