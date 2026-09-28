"""Fonction AWS Lambda (ou tout FaaS compatible) exposant Krito, sans PyTorch.

Le moteur est créé au chargement du module : ce coût (≈ 2 s) n'est payé qu'au
démarrage à froid, puis chaque invocation ne fait que l'inférence.

Événement accepté, en appel direct ou via API Gateway (champ « body ») :
    {"context": "...", "options": {"cle": "description", ...}, "min_entailment": 0.5}
"""

from __future__ import annotations

import json
import os

from krito import KritoEngine

engine = KritoEngine.from_onnx(
    os.environ.get("KRITO_MODEL", "/opt/krito/model"),
    threads=int(os.environ.get("KRITO_THREADS", "2")),
    hypothesis_template=os.environ.get("KRITO_TEMPLATE", "Ce texte concerne {}."),
)


def handler(event, context=None):
    payload = json.loads(event["body"]) if isinstance(event.get("body"), str) else event
    try:
        r = engine.classify(
            payload["context"], payload["options"],
            threshold=payload.get("threshold"), min_margin=payload.get("min_margin"),
            min_entailment=payload.get("min_entailment"),
        )
    except KeyError as exc:
        return {"statusCode": 400, "body": json.dumps({"error": f"champ manquant : {exc.args[0]}"}, ensure_ascii=False)}
    except ValueError as exc:
        return {"statusCode": 400, "body": json.dumps({"error": str(exc)}, ensure_ascii=False)}
    body = {
        "selected_key": r.selected_key, "confidence": r.confidence, "margin": r.margin,
        "accepted": r.accepted, "rejection_reason": r.rejection_reason,
        "scores": {k: {"probability": s.probability, "entailment_prob": s.entailment_prob} for k, s in r.scores.items()},
    }
    return {"statusCode": 200, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, ensure_ascii=False)}
