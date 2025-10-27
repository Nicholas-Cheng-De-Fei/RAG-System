from langchain_chroma import Chroma
from models.document_models import DocumentProcessRequest
from services.document_chunking import read_pdf_document, native_chunking
from services.chroma_db_service import embed_and_add_document

def chunk_document(process_request: DocumentProcessRequest, chroma_db: Chroma) -> None:
    """
    Performs semantic chunking on the given document and stores it in the chroma vector database.

    Returns
    -------
    A list of langchain document objects.
    """
    list_of_documents = read_pdf_document(process_request.document_path)
    
    native_chunks = native_chunking(list_of_documents)

    embed_and_add_document(native_chunks, chroma_db)