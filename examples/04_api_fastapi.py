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
app = FastAPI(title="Krito", version="0.2.0",
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


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    r = engine.classify(req.context, req.options, threshold=req.threshold,
                        min_margin=req.min_margin, min_entailment=req.min_entailment)
    return ClassifyResponse(
        selected_key=r.selected_key, confidence=r.confidence, margin=r.margin,
        accepted=r.accepted, rejection_reason=r.rejection_reason,
        scores={k: OptionOut(probability=s.probability, entailment_prob=s.entailment_prob) for k, s in r.scores.items()},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
