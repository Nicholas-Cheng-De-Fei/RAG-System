from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from models.app_models import DocumentProcessRequest, QueryRequest

from services.document_chunking import layout_chunking, read_pdf_document, native_chunking, read_pdf_document_into_markdown, semantic_chunking
from services.chroma_db_service import embed_and_add_document, multi_retrieve

from services.query_service import query_google_ai
from services.reranking import rerank
from services.query_service import query_google_ai, query_transformation

def chunk_document(process_request: DocumentProcessRequest, chroma_db: Chroma) -> None:
    """
    Performs native chunking on the given document and stores it in the chroma vector database.

    Returns
    -------
    A list of langchain document objects.
    """
    list_of_documents = read_pdf_document(process_request.document_path)
    native_chunks = native_chunking(list_of_documents)
    embed_and_add_document(native_chunks, chroma_db)
    
def chunk_document_semantically(process_request: DocumentProcessRequest, chroma_db: Chroma) -> None:
    """
    Performs semantic chunking on the given document and stores it in the chroma vector database.

    Returns
    -------
    A list of langchain document objects.
    """
    list_of_documents = read_pdf_document(process_request.document_path)
    semantic_chunks = semantic_chunking(list_of_documents)
    embed_and_add_document(semantic_chunks, chroma_db)
    
def chunk_document_with_layout(process_request: DocumentProcessRequest, chroma_db: Chroma) -> None:
    """
    Performs layout-aware chunking on the given document and stores it in the chroma vector database.

    Returns
    -------
    A list of langchain document objects.
    """
    list_of_documents = read_pdf_document_into_markdown(process_request.document_path)
    layout_chunks = layout_chunking(list_of_documents)
    embed_and_add_document(layout_chunks, chroma_db)

def query_ai_model(request: QueryRequest, google_ai: ChatGoogleGenerativeAI) -> dict:
    """
    Sends the query into the ai model and return its response.
    """
    return query_google_ai(request.query, google_ai)

def retrieve_and_query_ai_model(request: QueryRequest, google_ai: ChatGoogleGenerativeAI, chroma_db ) -> dict:
    """
    Retrieves relevant documents and dends it as context to the model
    """
    queries = query_transformation(request.query, google_ai)
    
    context = multi_retrieve(queries, chroma_db)

    if (context):
        documents = [getattr(doc, "page_content", str(doc)) for doc in context]
        reranked_context = rerank(request.query, documents)
    else:
        reranked_context = ["No relevant documents retrieved"]

    context = "\n\n".join(reranked_context)
    query_with_context = f"Context:\n{context}\n\nQuestion:\n{request.query}"

    return query_google_ai(query_with_context, google_ai)