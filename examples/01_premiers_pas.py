"""Premiers pas : classer un message parmi des options décrites en langage naturel.

    uv run python examples/01_premiers_pas.py
"""

from _modele import MODEL, TEMPLATE

from krito import KritoEngine

# Chargement unique (≈ 2 s) : ensuite, chaque décision prend quelques centaines de ms sur CPU.
engine = KritoEngine.from_onnx(MODEL, threads=2, hypothesis_template=TEMPLATE)

options = {
    "facturation": "un problème de facturation ou de paiement",
    "livraison": "la livraison ou le suivi d'un colis",
    "compte": "la connexion ou l'accès au compte",
    "bug": "une panne technique du site ou de l'application",
}

message = "Bonjour, depuis la mise à jour de ce matin l'application plante dès que j'ouvre mes factures."
result = engine.classify(message, options, min_entailment=0.5)

print(f"Message   : {message}")
print(f"Décision  : {result.selected_key} ({result.selected_label})")
print(f"Confiance : {result.confidence:.1%}   marge : {result.margin:.1%}")
print(f"Acceptée  : {result.accepted}" + (f"  ({result.rejection_reason})" if result.rejection_reason else ""))
print()
print(f"{'option':<12} {'probabilité':>11} {'implication':>11}")
for key, score in sorted(result.scores.items(), key=lambda kv: -kv[1].probability):
    print(f"{key:<12} {score.probability:>11.1%} {score.entailment_prob:>11.1%}")
