"""Questions oui/non et échelles ordonnées, sans entraînement.

- ``yes_no`` vérifie une affirmation : « Le client demande un remboursement. » ;
- ``scale`` situe un texte sur une échelle ordonnée (urgence, satisfaction, gravité…) et
  donne, en plus du niveau le plus probable, sa position moyenne sur l'échelle.

    uv run python examples/07_oui_non_echelle.py
"""

from _modele import MODEL, TEMPLATE

from krito import KritoEngine

engine = KritoEngine.from_onnx(MODEL, threads=2, hypothesis_template=TEMPLATE)

messages = [
    "Ça fait trois semaines que j'attends mon colis, je veux être remboursé immédiatement !",
    "Merci pour la livraison rapide, tout est parfait.",
    "URGENT : plus personne ne peut se connecter au logiciel de caisse, les magasins sont bloqués.",
    "Pourriez-vous m'envoyer votre catalogue quand vous aurez un moment ?",
]

# Plusieurs affirmations indépendantes (plusieurs réponses « oui » possibles pour un même message)
questions = {
    "remboursement": "Le client demande un remboursement.",
    "mécontent": "Le client est mécontent.",
}
URGENCE = ["pas urgente du tout", "peu urgente", "urgente", "extrêmement urgente"]

for message in messages:
    print(message)
    for name, statement in questions.items():
        r = engine.yes_no(message, statement, max_neutral=0.8)
        answer = ("oui" if r.answer else "non") if r.accepted else "?"
        print(f"  {name:<14} {answer:<4} P(oui) = {r.probability:.0%}")
    s = engine.scale(message, URGENCE, template="Cette demande est {}.")
    print(f"  {'urgence':<14} {s.selected_key} (niveau {s.index + 1}/{len(URGENCE)}, "
          f"moyenne {s.expected + 1:.1f}, confiance {s.confidence:.0%})\n")
