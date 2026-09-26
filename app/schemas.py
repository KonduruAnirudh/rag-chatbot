# app/schemas.py
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"


class Source(BaseModel):
    doc_id: str
    filename: str
    chunk_index: int
    score: float
    snippet: str
    page_start: int | None = None
    page_end: int | None = None
    ocr: bool = False


# --- Step 8 (EXPERIMENTAL): graph mode only. Source above and ChatResponse below are
# unchanged, so /api/chat's responses are too.

class GraphLink(BaseModel):
    """One edge that reached a graph passage: the path, the edge, and the sentence it was quoted from."""
    reached_from: str             # an entity the question named
    to: str                       # the edge's other end
    subject: str
    relation: str                 # machine-extracted, about 48% correct: for people, never shown to the model
    object: str
    quote: str                    # verbatim from the passage
    subject_as_written: str
    object_as_written: str
    votes: int
    from_diagram: bool = False    # the quote is a vision model's description of an image


class GraphEvidence(BaseModel):
    linked_entities: list[str]
    links: list[GraphLink]


class GraphSource(Source):
    retrieval: Literal["vector", "graph"]
    graph: GraphEvidence | None = None


class GraphChatResponse(BaseModel):
    """POST /api/chat/graph: ChatResponse's fields, with sources that say how each was found."""
    answer: str
    sources: list[GraphSource]
    session_id: str
    search_query: str
    retrieval_mode: Literal["vector+graph"] = "vector+graph"


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str
    search_query: str