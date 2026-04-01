"""Evaluation router."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..services import eval_service

router = APIRouter(prefix="/eval", tags=["Evaluation"])


class EvalAllRequest(BaseModel):
    k: int = Field(default=3, ge=1, le=10)


class EvalQueryRequest(BaseModel):
    query: str
    relevant_doc_ids: list[str]
    k: int = Field(default=3, ge=1, le=10)


@router.post("/all")
def evaluate_all(req: EvalAllRequest):
    return eval_service.evaluate_all(req.k)


@router.post("/query")
def evaluate_query(req: EvalQueryRequest):
    return eval_service.evaluate_query(req.query, req.relevant_doc_ids, req.k)
