from fastapi import HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import time
from utils.logger import log

def read_pdf_document(document_path: str) -> list:
    """
    Reads the PDF document based on the document path.

    Returns
    -------
    A list of langchain document objects.
    """
    path = Path(document_path)

    # If the path is not absolute, make it absolute
    if (not path.is_absolute()):
        path = path.resolve()
    try:
        loader = PyPDFLoader(str(path))
        return loader.load()
    except ValueError as value_err:
        log.error(f"File path {document_path} is invalid or the file cannot be found.")
        raise HTTPException(status_code=400, detail="Invalid file path or file cannot be found")
    
def native_chunking(documents: list) -> list:
    """
    Spilts the text into multiple chunks based on the natural structure such as breakpoints, new lines etc.

    Returns
    -------
    Returns a list of documents but are smaller in text length than the original document list.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", "."],
    )
    log.info("Chunking process has begun.")
    start = time.perf_counter()

    base_chunks = splitter.split_documents(documents)

    end = time.perf_counter()
    print(f"Chunking process completed, took {end - start:.4f} seconds")
    
    return base_chunks

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model.
    """
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/modernbert-embed-base")
    return embeddings