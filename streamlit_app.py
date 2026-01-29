import streamlit as st
import requests

st.set_page_config(page_title="RAG QA System", layout="wide")
st.title("📄 Local RAG Question Answering")
st.markdown("Powered by **GPT4All Phi-3** + **FAISS**")

API_URL = "http://localhost:8000"

# Sidebar for upload
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")
    
    if uploaded_file is not None:
        files = {"file": uploaded_file}
        with st.spinner("Processing PDF..."):
            response = requests.post(f"{API_URL}/upload/", files=files)
        if response.status_code == 200:
            st.success("✅ PDF processed!")
            st.session_state.ready = True
        else:
            st.error("❌ Upload failed")

# Main area for questions
st.header("💬 Ask Questions")
if 'ready' not in st.session_state:
    st.session_state.ready = False

if st.session_state.ready:
    question = st.text_input("Enter your question about the PDF:")
    if st.button("Generate Answer", type="primary") and question:
        with st.spinner("Generating with GPT4All..."):
            response = requests.post(f"{API_URL}/ask/", json={"question": question})
            if response.status_code == 200:
                result = response.json()
                st.markdown("### 🤖 Answer")
                st.write(result["answer"])
                st.caption(f"⏱️ {result['latency_ms']:.1f}ms")
            else:
                st.error("Query failed")
else:
    st.info("👆 Upload a PDF first (sidebar)")
    st.balloons()
