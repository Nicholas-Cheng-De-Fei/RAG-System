from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import re
import time
from utils.logger import log
from utils.utils import get_envvar

ENV_HUGGINGFACE_EMBEDDING_MODEL = "HUGGINGFACE_EMBEDDING_MODEL"

def read_pdf_document(document_path: str) -> str:
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
        raw_docs = loader.load()
        # Clean the text for each document/page
        cleaned_docs = []
        for doc in raw_docs:
            cleaned_content = clean_text(doc.page_content)
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            cleaned_docs.append(cleaned_doc)

        return cleaned_docs
    except ValueError as value_err:
        log.error(f"File path {document_path} is invalid or the file cannot be found")
        raise HTTPException(status_code=400, detail="Invalid file path or file cannot be found")

def clean_text(text: str) -> str:
    """
    Light text cleaning for PDF content.
    Removes HTML, URLs, extra whitespace, and non-printable characters.
    """
    text = re.sub(r"<[^>]+>", " ", text)              # remove HTML
    text = re.sub(r"http\S+|www\.\S+", "", text)      # remove URLs
    text = re.sub(r"\[[0-9]*\]", "", text)            # remove citation marks [1], [2]
    text = re.sub(r"\s+", " ", text)                  # collapse multiple spaces/newlines
    text = re.sub(r"[^\x20-\x7E]", "", text)          # remove non-ASCII characters
    return text.strip()

def native_chunking(documents: list) -> list:
    """
    Spilts the text into multiple chunks based on the natural structure such as breakpoints, new lines etc.

    Returns
    -------
    Returns a list of documents but are smaller in text length than the original document list.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", "."],
    )
    log.info("Chunking process has begun")
    start = time.perf_counter()

    base_chunks = splitter.split_documents(documents)

    end = time.perf_counter()
    log.info(f"Chunking process completed, took {end - start:.4f} seconds")
    
    return base_chunks

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the embedding model.
    """
    huggingface_embedding_model = get_envvar(ENV_HUGGINGFACE_EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=huggingface_embedding_model)
    return embeddings