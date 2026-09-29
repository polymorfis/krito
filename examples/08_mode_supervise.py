"""Mode supervisé pour une taxonomie fixe : ``KritoClassifier.fit(exemples)``.

Quand les catégories ne changent pas et que l'on a quelques dizaines d'exemples par
catégorie, un classifieur appris sur des embeddings est précis et rapide : un seul passage
du texte dans le modèle, quel que soit le nombre de catégories. Un juge de confiance est
appris en même temps (validation croisée) pour rejeter les erreurs probables et les hors-sujet.

    uv sync --all-groups
    uv run python examples/08_mode_supervise.py
    KRITO_EMBEDDER=modeles/e5-base uv run python examples/08_mode_supervise.py   # sans PyTorch
"""

from __future__ import annotations

import sys
import time

from _modele import embedder

from krito import KritoClassifier, evaluate_guardrails, load_examples, split_examples

path = sys.argv[1] if len(sys.argv) > 1 else "examples/data/annotes.csv"
train, test = split_examples(load_examples(path), test_size=0.3)

t0 = time.perf_counter()
clf = KritoClassifier(embedder()).fit(train)
print(f"Entraîné sur {len(train)} exemples en {time.perf_counter() - t0:.1f} s : {clf.classes_}")
clf.save("classifieur.npz")  # à recharger avec KritoClassifier.load("classifieur.npz")

t0 = time.perf_counter()
results = clf.classify_batch([ex.text for ex in test])
elapsed = time.perf_counter() - t0
labels = [ex.label for ex in test]
in_domain = [(r, lab) for r, lab in zip(results, labels) if lab is not None]
accuracy = sum(r.selected_key == lab for r, lab in in_domain) / len(in_domain)
print(f"{len(test)} messages de test en {elapsed:.1f} s ; précision sur les messages du domaine : {accuracy:.1%}")

for threshold in (0.5, 0.7, 0.9):
    p = evaluate_guardrails(results, labels, min_judge_score=threshold)
    print(f"  min_judge_score={threshold} : {p.accepted} acceptés, précision {p.precision:.1%}, "
          f"automatisé {p.automated:.0%}")

r = clf.classify("Bonjour, auriez-vous une recette de gâteau au chocolat ?", min_judge_score=0.7)
print(f"\nHors-sujet : {r.selected_key}, juge {r.judge_score:.0%} -> "
      f"{'accepté' if r.accepted else 'revue humaine'}")
