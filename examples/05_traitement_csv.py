"""Traitement d'un fichier CSV : chaque ligne reçoit une décision, ou part en revue humaine.

    uv run python examples/05_traitement_csv.py examples/data/messages.csv resultats.csv
"""

from __future__ import annotations

import csv
import sys
import time

from _modele import MODEL, TEMPLATE

from krito import KritoEngine

OPTIONS = {
    "facturation": "un problème de facturation, de paiement ou de remboursement",
    "livraison": "la livraison, le suivi ou le retour d'un colis",
    "compte": "la connexion, le mot de passe ou l'accès au compte",
    "resiliation": "une résiliation ou l'arrêt d'un abonnement",
    "bug": "un bug technique ou une panne du site ou de l'application",
    "commercial": "les tarifs, les offres ou une demande commerciale",
}


def main(src: str, dst: str) -> None:
    engine = KritoEngine.from_onnx(MODEL, threads=4, hypothesis_template=TEMPLATE)
    with open(src, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t0 = time.perf_counter()
    for row in rows:
        r = engine.classify(row["message"], OPTIONS, min_entailment=0.5, min_margin=0.2)
        row.update(categorie=r.selected_key, confiance=f"{r.confidence:.3f}",
                   decision="auto" if r.accepted else "revue_humaine", motif=r.rejection_reason or "")
    elapsed = time.perf_counter() - t0
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    auto = sum(r["decision"] == "auto" for r in rows)
    print(f"{len(rows)} messages en {elapsed:.1f} s ({1000 * elapsed / len(rows):.0f} ms/message) : "
          f"{auto} traités automatiquement, {len(rows) - auto} en revue humaine -> {dst}\n")
    print(f"{'décision':<14} {'catégorie':<12} {'confiance':>9}  message")
    for row in rows[:12]:
        message = row["message"] if len(row["message"]) <= 70 else row["message"][:69] + "…"
        print(f"{row['decision']:<14} {row['categorie']:<12} {float(row['confiance']):>9.0%}  {message}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("examples/data/messages.csv", "resultats.csv")))
