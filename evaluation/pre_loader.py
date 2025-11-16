from models.app_models import DocumentProcessRequest
from services.chroma_db_service import connect_to_chroma_db
from controllers.app_controller import chunk_document, chunk_document_with_layout

def preload_native_chunking(db_name:str, request: DocumentProcessRequest):
    db = connect_to_chroma_db(db_name)
    chunk_document(request, db)

def preload_layout_chunking(db_name:str, request: DocumentProcessRequest):
    db = connect_to_chroma_db(db_name)
    chunk_document_with_layout(request, db)
    
def combi_chunking(request: DocumentProcessRequest):
    db_name1= "normal-combi"
    db_name2 = "layout-combi"
    db1 = connect_to_chroma_db(db_name1)
    db2 = connect_to_chroma_db(db_name2)
    
    chunk_document(request, db1)
    chunk_document_with_layout(request, db2)