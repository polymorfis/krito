"""Garde-fous : accepter automatiquement ce qui est sûr, renvoyer le reste à un humain.

- threshold      : confiance relative minimale entre les options ;
- min_margin     : écart minimal avec la deuxième option ;
- min_entailment : implication absolue minimale, qui rejette les messages hors-sujet.

    uv run python examples/02_garde_fous.py
"""

from _modele import MODEL, TEMPLATE

from krito import KritoEngine

engine = KritoEngine.from_onnx(MODEL, threads=2, hypothesis_template=TEMPLATE)

options = {
    "moteur": "le moteur ou un problème de démarrage",
    "freins": "le freinage, les plaquettes ou les disques",
    "pneus": "les pneus ou la suspension",
    "entretien": "l'entretien courant, la révision ou la vidange",
}

messages = [
    "Ma moto fait un bruit de ferraille quand je freine à l'arrière.",
    "C'est le moment de faire la révision des 10 000 km, vous avez un créneau ?",
    "Vous avez un parking pour les clients ?",
    "Quelle est la meilleure recette de pâte à crêpes ?",
]

SHORT = {"confidence": "confiance", "margin": "marge", "entailment": "implication"}

for message in messages:
    r = engine.classify(message, options, threshold=0.6, min_margin=0.2, min_entailment=0.5)
    if r.accepted:
        verdict = f"AUTO    {r.selected_key}"
    else:
        motifs = [SHORT[m.split("_")[0].strip()] for m in r.rejection_reason.split("|")]
        verdict = "HUMAIN  " + ", ".join(motifs) + " trop faibles" if len(motifs) > 1 else f"HUMAIN  {motifs[0]} trop faible"
    print(f"{verdict:<50} | {message}")
