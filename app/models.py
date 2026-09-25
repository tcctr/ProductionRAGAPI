"""Request and response schemas. FastAPI validates requests against these and builds /docs from them."""
from typing import Literal

from pydantic import BaseModel, Field

Version = Literal[16, 17, 18]
DocType = Literal["sql_command", "functions", "indexes_perf"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    version: Version | None = Field(None, description="Only search this version; if omitted, "
                                    "chunks identical across versions are merged into one result")
    doc_type: DocType | None = None
    k: int = Field(5, ge=1, le=50)


class Chunk(BaseModel):
    id: str
    version: int
    versions: list[int] = Field(description="Versions whose copy of this exact chunk text was "
                                "among the retrieved candidates (merged into this result)")
    doc_type: DocType
    page: str
    section_title: str
    heading_path: list[str]
    url: str
    similarity: float
    content: str


class QueryResponse(BaseModel):
    question: str
    chunks: list[Chunk]


class IngestRequest(BaseModel):
    """One documentation page, in the same shape as a data/parsed/docs.jsonl record."""
    version: Version
    doc_type: DocType
    section_title: str = Field(min_length=1)
    page: str = Field(min_length=1, pattern=r"^[\w.-]+\.html$")
    url: str = Field(pattern=r"^https?://")
    text: str = Field(min_length=1, description="Page body as markdown")


class IngestResponse(BaseModel):
    id: str = Field(description='"<version>:<page>"')
    chunks_total: int
    chunks_embedded: int = Field(description="New or changed chunks sent to the embedding server")
    chunks_deleted: int
