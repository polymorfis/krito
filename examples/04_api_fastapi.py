"""API HTTP de décision, prête pour un serveur CPU d'entreprise.

    uv sync --group examples
    uv run uvicorn --app-dir examples 04_api_fastapi:app --host 0.0.0.0 --port 8000

Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import os

from _modele import MODEL, TEMPLATE
from fastapi import FastAPI
from pydantic import BaseModel, Field

from krito import KritoEngine

engine = KritoEngine.from_onnx(MODEL, threads=int(os.environ.get("KRITO_THREADS", "2")),
                               hypothesis_template=os.environ.get("KRITO_TEMPLATE", TEMPLATE))
app = FastAPI(title="Krito", version="0.3.0",
              description="Décisions typées et confiance chiffrée, sur CPU, sans GPU ni API externe.")


class ClassifyRequest(BaseModel):
    context: str = Field(..., examples=["Mon colis est indiqué livré mais je ne l'ai jamais reçu."])
    options: dict[str, str] = Field(..., min_length=2, examples=[{
        "facturation": "un problème de facturation ou de paiement",
        "livraison": "la livraison ou le suivi d'un colis",
        "compte": "la connexion ou l'accès au compte",
    }])
    threshold: float | None = None
    min_margin: float | None = None
    min_entailment: float | None = Field(None, examples=[0.5])


class OptionOut(BaseModel):
    probability: float
    entailment_prob: float


class ClassifyResponse(BaseModel):
    selected_key: str
    confidence: float
    margin: float
    accepted: bool
    rejection_reason: str | None
    scores: dict[str, OptionOut]


class ClassifyBatchRequest(BaseModel):
    contexts: list[str] = Field(..., min_length=1, max_length=256, examples=[[
        "Mon colis est indiqué livré mais je ne l'ai jamais reçu.", "J'ai été débité deux fois."]])
    options: dict[str, str] = Field(..., min_length=2)
    threshold: float | None = None
    min_margin: float | None = None
    min_entailment: float | None = None


class YesNoRequest(BaseModel):
    context: str = Field(..., examples=["Je veux être remboursé, le produit est arrivé cassé."])
    statement: str = Field(..., examples=["Le client demande un remboursement."])
    threshold: float | None = None
    max_neutral: float | None = Field(None, examples=[0.8])


class YesNoResponse(BaseModel):
    answer: bool
    probability: float
    confidence: float
    neutral_prob: float
    accepted: bool
    rejection_reason: str | None


class ScaleRequest(BaseModel):
    context: str = Field(..., examples=["URGENT : plus aucun client ne peut payer sur le site."])
    levels: list[str] = Field(..., min_length=2, examples=[["pas urgente", "peu urgente", "urgente", "très urgente"]])
    template: str | None = Field(None, examples=["Cette demande est {}."])
    threshold: float | None = None
    max_spread: float | None = None


class ScaleResponse(BaseModel):
    selected_key: str
    index: int
    expected: float
    spread: float
    confidence: float
    accepted: bool
    rejection_reason: str | None


def _classify_response(r) -> ClassifyResponse:
    return ClassifyResponse(
        selected_key=r.selected_key, confidence=r.confidence, margin=r.margin,
        accepted=r.accepted, rejection_reason=r.rejection_reason,
        scores={k: OptionOut(probability=s.probability, entailment_prob=s.entailment_prob) for k, s in r.scores.items()},
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    return _classify_response(engine.classify(req.context, req.options, threshold=req.threshold,
                                              min_margin=req.min_margin, min_entailment=req.min_entailment))


@app.post("/classify_batch", response_model=list[ClassifyResponse])
def classify_batch(req: ClassifyBatchRequest) -> list[ClassifyResponse]:
    results = engine.classify_batch(req.contexts, req.options, threshold=req.threshold,
                                    min_margin=req.min_margin, min_entailment=req.min_entailment)
    return [_classify_response(r) for r in results]


@app.post("/yes_no", response_model=YesNoResponse)
def yes_no(req: YesNoRequest) -> YesNoResponse:
    r = engine.yes_no(req.context, req.statement, threshold=req.threshold, max_neutral=req.max_neutral)
    return YesNoResponse(answer=r.answer, probability=r.probability, confidence=r.confidence,
                         neutral_prob=r.neutral_prob, accepted=r.accepted, rejection_reason=r.rejection_reason)


@app.post("/scale", response_model=ScaleResponse)
def scale(req: ScaleRequest) -> ScaleResponse:
    r = engine.scale(req.context, req.levels, template=req.template, threshold=req.threshold,
                     max_spread=req.max_spread)
    return ScaleResponse(selected_key=r.selected_key, index=r.index, expected=r.expected, spread=r.spread,
                         confidence=r.confidence, accepted=r.accepted, rejection_reason=r.rejection_reason)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
