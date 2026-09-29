"""Juge de confiance : un seul score, appris sur vos données, au lieu de plusieurs seuils.

Le juge apprend, sur des décisions annotées, la probabilité qu'une décision soit bonne à
partir de ses signaux (confiance, marge, implication absolue, entropie…). On l'entraîne sur
une partie des données, on calibre son seuil sur une autre, et on vérifie sur une troisième.

    uv run python examples/09_juge_confiance.py examples/data/annotes.csv 0.9
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
    judge_set, rest = split_examples(load_examples(path), test_size=0.6)
    calibration_set, validation_set = split_examples(rest, test_size=0.5)

    # Référence : garde-fous classiques, calibrés puis vérifiés
    base = engine.calibrate(calibration_set, OPTIONS, target_precision=target)
    judge = engine.fit_judge(judge_set, OPTIONS)
    with_judge = engine.calibrate(calibration_set, OPTIONS, target_precision=target)  # min_judge_score
    judge.save("juge.json")  # KritoEngine.from_onnx(..., judge=ConfidenceJudge.load("juge.json"))

    results = engine.classify_batch([ex.text for ex in validation_set], OPTIONS)
    labels = [ex.label for ex in validation_set]
    print(f"Vérification sur {len(validation_set)} messages jamais utilisés :")
    for name, cal in (("seuils classiques", base), ("juge de confiance", with_judge)):
        p = evaluate_guardrails(results, labels, **cal.guardrails)
        print(f"  {name:<18} {cal.guardrails} -> précision {p.precision:.1%}, automatisé {p.automated:.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/data/annotes.csv",
         float(sys.argv[2]) if len(sys.argv) > 2 else 0.9)
