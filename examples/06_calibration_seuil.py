"""Calibrer les garde-fous sur VOS données annotées, pour une précision cible.

L'échelle de la probabilité d'implication dépend du modèle et du gabarit : un seuil
ne se devine pas, il se mesure. Il faut un fichier annoté (texte + bonne catégorie,
« none » pour les messages hors-sujet), idéalement différent des données d'entraînement.

Le fichier est coupé en deux : la première moitié fixe les seuils (``engine.calibrate``),
la seconde vérifie qu'ils tiennent sur des messages qui n'ont pas servi à les choisir.

    uv run python examples/06_calibration_seuil.py examples/data/annotes.csv 0.95
"""

from __future__ import annotations

import sys

from _modele import MODEL, TEMPLATE

from krito import KritoEngine, evaluate_guardrails, load_examples, split_examples

OPTIONS = {
    "billing": "un problème de facturation, de paiement ou de remboursement",
    "shipping": "la livraison, le suivi ou le retour d'un colis",
    "account": "la connexion, le mot de passe ou l'accès au compte",
    "cancel": "une résiliation ou l'arrêt d'un abonnement",
    "bug": "un bug technique ou une panne du site ou de l'application",
    "sales": "les tarifs, les offres ou une demande commerciale",
}


def main(path: str, target: float) -> None:
    engine = KritoEngine.from_onnx(MODEL, threads=4, hypothesis_template=TEMPLATE)
    calibration_set, validation_set = split_examples(load_examples(path), test_size=0.5)

    # Cherche la combinaison (min_entailment, min_margin) qui automatise le plus de bonnes
    # décisions tout en respectant la précision cible parmi les décisions acceptées.
    cal = engine.calibrate(calibration_set, OPTIONS, target_precision=target)
    print(cal)

    results = engine.classify_batch([ex.text for ex in validation_set], OPTIONS)
    check = evaluate_guardrails(results, [ex.label for ex in validation_set], **cal.guardrails)
    print(f"\nVérification sur {len(validation_set)} autres messages : précision {check.precision:.1%}, "
          f"{check.automated:.0%} des messages du domaine traités automatiquement.")
    if check.precision < target:
        print("La précision baisse hors du jeu de calibration : annotez plus d'exemples ou visez plus haut.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/data/annotes.csv",
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.95)
