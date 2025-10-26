from controllers.app_controller import chunk_document
from fastapi import FastAPI
from models.document_models import DocumentProcessRequest

async def lifespan(app: FastAPI):
    """
    Set up variables for application. \n
    And before shutting down clean up.
    """

    yield
    # Anything after yeild is for teardown / cleanup

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
    chunk_document(process_request)
    return {"message": "Document has been chunked"}