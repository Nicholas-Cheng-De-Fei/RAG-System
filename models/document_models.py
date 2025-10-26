from pydantic import BaseModel, Field
from typing import Annotated

class DocumentProcessRequest(BaseModel):
    document_path: Annotated[str, Field(min_length=1)]
