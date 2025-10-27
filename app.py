from controllers.app_controller import chunk_document
from fastapi import FastAPI
from models.app_models import DocumentProcessRequest
from services.chroma_db_service import connect_to_chroma_db, disconnect_chroma_db
from utils.logger import log

async def lifespan(app: FastAPI):
    """
    Set up variables for application. \n
    And before shutting down clean up.
    """
    app.state.chroma_db = connect_to_chroma_db()
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
    return {"message": "Document has been chunked"}

@app.post("/ask")
async def query_ai_modell() -> dict:
    """
    Queries the AI model and send back its response
    """
