"""Calibrer les garde-fous sur VOS données annotées, pour une précision cible.

L'échelle de la probabilité d'implication dépend du modèle et du gabarit : un seuil
ne se devine pas, il se mesure. Il faut un fichier annoté (texte + bonne catégorie,
« none » pour les messages hors-sujet), idéalement différent des données d'entraînement.

    uv run python examples/06_calibration_seuil.py examples/data/annotes.csv 0.95
"""

from __future__ import annotations

import csv
import sys

import numpy as np
from _modele import MODEL, TEMPLATE

from krito import KritoEngine

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
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    good, entail, margin = [], [], []
    for row in rows:
        r = engine.classify(row["text"], OPTIONS)
        good.append(row["label"] == r.selected_key)  # un hors-sujet (« none ») n'est jamais « bon »
        entail.append(r.scores[r.selected_key].entailment_prob)
        margin.append(r.margin)
    good, entail, margin = map(np.array, (good, entail, margin))
    n_in = sum(row["label"] != "none" for row in rows)
    print(f"{len(rows)} exemples annotés, dont {len(rows) - n_in} hors-sujet")
    print(f"Sans garde-fou : {good.sum() / n_in:.1%} de bonnes réponses sur les messages du domaine\n")

    # Recherche de la combinaison (min_entailment, min_margin) qui automatise le plus
    # tout en respectant la précision cible parmi les décisions acceptées.
    results = []
    for t_e in (0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9):
        for t_m in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
            acc = (entail >= t_e) & (margin >= t_m)
            if acc.any():
                results.append((good[acc].mean(), good[acc].sum() / n_in, t_e, t_m))
    print(f"{'min_entailment':>15} {'min_margin':>11} {'précision':>10} {'automatisé':>11}")
    ok = sorted((r for r in results if r[0] >= target), key=lambda r: -r[1])
    shown = ok[:5] if ok else sorted(results, key=lambda r: -r[0])[:5]
    for p, c, t_e, t_m in shown:
        print(f"{t_e:>15.2f} {t_m:>11.2f} {p:>10.1%} {c:>11.1%}")
    if ok:
        p, c, t_e, t_m = ok[0]
        print(f"\nRecommandé pour {target:.0%} de précision : min_entailment={t_e}, min_margin={t_m}"
              f" -> {c:.0%} des messages traités automatiquement, le reste en revue humaine.")
    else:
        print(f"\nAucune combinaison n'atteint {target:.0%} : baissez la cible ou fine-tunez le modèle sur votre domaine.")
    print("À valider ensuite sur un second jeu annoté, distinct de celui-ci.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/data/annotes.csv",
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.95)
