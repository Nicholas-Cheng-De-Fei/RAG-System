from controllers.app_controller import chunk_document, query_ai_model, retrieve_and_query_ai_model
from fastapi import FastAPI
from models.app_models import DocumentProcessRequest, QueryRequest
from services.chroma_db_service import connect_to_chroma_db, disconnect_chroma_db, get_document_count
from services.query_service import connect_to_google_ai
from utils.logger import log

async def lifespan(app: FastAPI):
    """
    Set up variables for application. \n
    And before shutting down clean up.
    """
    log.info("Backend server starting up")
    app.state.chroma_db = connect_to_chroma_db()
    app.state.google_ai = connect_to_google_ai()
    yield
    # Anything after yeild is for teardown / cleanup
    log.warning("Disconnecting from chroma DB")
    disconnect_chroma_db(app.state.chroma_db )
    log.warning("Backend server shutting down")

app = FastAPI(title="Study Buddy", lifespan=lifespan)

@app.get("/")
async def root() -> dict:
    """
    Returns the status of the application. If it is not running then this API end point will not work.
    """
    return {"message": "Running"}

@app.post("/chunk/pdf")
async def chunk_pdf_document(process_request: DocumentProcessRequest) -> dict:
    """
    Takes in a PDF document locally and chunks it.
    """
    chunk_document(process_request, app.state.chroma_db)
    get_document_count(app.state.chroma_db)
    return {"message": "Document has been chunked"}

@app.post("/ask")
async def query_ai_modell(request: QueryRequest) -> dict:
    """
    Queries the AI model with no retrieval.
    """
    return query_ai_model(request, app.state.google_ai) 

@app.post("/rag/ask")
async def rag_query_ai_model(request: QueryRequest) -> dict:
    """
    Retrieves and queries the model.
    """
    return retrieve_and_query_ai_model(request, app.state.google_ai, app.state.chroma_db)