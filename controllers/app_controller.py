
from models.document_models import DocumentProcessRequest
from services.document_chunking import read_pdf_document, native_chunking


def chunk_document(process_request: DocumentProcessRequest) -> None:
    """
    Performs semantic chunking on the given document and stores it in the chroma vector database.

    Returns
    -------
    A list of langchain document objects.
    """
    list_of_documents = read_pdf_document(process_request.document_path)

    native_chunks = native_chunking(list_of_documents)

    print(native_chunks[0])