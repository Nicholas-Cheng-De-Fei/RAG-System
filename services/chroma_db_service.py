from langchain_chroma import Chroma
from services.document_chunking import get_embedding_model
import time
from utils.logger import log

def connect_to_chroma_db() -> Chroma:
    """
    Establishes a connection with chroma database.
    """
    vector_store = Chroma(
        collection_name="document_store",
        embedding_function= get_embedding_model(),
        persist_directory="./data/chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    log.info("INFO: Connected to chroma DB")

    return vector_store

def disconnect_chroma_db(chroma_db: Chroma) -> None:
    """
    Severs the connection to the chroma DB
    """
    chroma_db.delete_collection()

def embed_and_add_document(documents: list, chroma_db: Chroma) -> None:
    """
    Embeds the documents and adds it into chroma database.
    """
    log.info("INFO: Embedding process has begun")
    start = time.perf_counter()

    chroma_db.add_documents(documents)
    
    end = time.perf_counter()
    log.info(f"INFO: Embedding process completed and has been stored into chroma database, took {end - start:.4f} seconds")

def retrieve(query: str, chroma_db: Chroma, k: int = 20) -> dict:
    """
    Retrieves the top k most relevent documents based on the query.
    """
    retrieved_docs_with_scores = chroma_db.similarity_search_with_score(query, k=k)

    retrieved_docs = [doc for doc, score in retrieved_docs_with_scores if score <= 0.6]
    
    log.info(f"INFO: Retrieved {len(retrieved_docs)} documents")

    return retrieved_docs

        