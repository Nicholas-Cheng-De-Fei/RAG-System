import streamlit as st
import requests
import os
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Processing and RAG QA",
    page_icon="📄",
    layout="wide",
)

# --- Title and Description ---
st.title("📄 PDF Processing and Question Answering")
st.markdown("Upload a PDF, choose a chunking method, and then ask questions based on the document's content.")

# --- PDF Upload ---
st.header("1. Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    if not os.path.exists("temp"):
        os.makedirs("temp")
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # --- Chunking Section ---
    st.header("2. Chunk the Document")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Normal Chunking"):
            with st.spinner("Processing with normal chunking..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/chunk/pdf",
                        json={"document_path": file_path}
                    )
                    response.raise_for_status()  # Raise an exception for bad status codes
                    st.success(response.json().get("message", "Chunking complete!"))
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")

    with col2:
        if st.button("Semantic Chunking"):
            with st.spinner("Processing with semantic chunking..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/chunk/pdf/semantic",
                        json={"document_path": file_path}
                    )
                    response.raise_for_status()
                    st.success(response.json().get("message", "Chunking complete!"))
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")

    with col3:
        if st.button("Layout Chunking"):
            with st.spinner("Processing with layout chunking..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/chunk/pdf/layout",
                        json={"document_path": file_path}
                    )
                    response.raise_for_status()
                    st.success(response.json().get("message", "Chunking complete!"))
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred: {e}")

    # --- RAG Question Answering Section ---
    st.header("3. Ask a Question")
    query = st.text_input("Enter your question:")

    if st.button("Ask"):
        if query:
            with st.spinner("Retrieving answer..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/rag/ask",
                        json={"query": query}
                    )
                    response.raise_for_status()
                    data = response.json()

                    st.subheader("Answer:")
                    st.write(data.get("response", {}).get("content", "No content found."))

                    st.subheader("Retrieved Context:")
                    # The context might be nested in the 'query' part of the response
                    retrieved_query = data.get("query", "")
                    context_start = retrieved_query.find("Context:")
                    if context_start != -1:
                        context = retrieved_query[context_start + len("Context:"):]
                        st.text_area("Context", context, height=200)
                    else:
                        st.warning("Could not extract context from the response.")

                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred while asking the question: {e}")
                except json.JSONDecodeError:
                    st.error("Failed to decode the JSON response from the server.")
        else:
            st.warning("Please enter a question.")

else:
    st.info("Please upload a PDF file to begin.")
