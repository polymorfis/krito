"""Contrôle qualité d'un jeu de données annoté AVANT de l'utiliser pour calibrer, entraîner
ou publier un modèle.

Vérifie :
- les étiquettes (inconnues de la taxonomie, catégories trop peu représentées, part de hors-sujet) ;
- les textes (vides, doublons, trop longs pour le modèle) ;
- la présence probable de **données personnelles** à anonymiser (e-mails, téléphones, IBAN,
  numéros de carte ou de sécurité sociale) : indispensable avant tout partage ou publication ;
- avec ``--second``, l'**accord entre deux annotateurs** (kappa de Cohen) sur les textes communs,
  et la liste des désaccords à arbitrer.

    uv run python experiments/check_annotations.py mes_messages.csv --taxonomy mes_categories.json
    uv run python experiments/check_annotations.py annotateur_a.csv --second annotateur_b.csv

Code de sortie 1 s'il y a une erreur bloquante (étiquette inconnue, texte vide, donnée personnelle).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from krito import load_examples

MIN_PER_CLASS = 30  # en dessous, calibration et fine-tuning sont peu fiables
MAX_WORDS = 350  # ≈ 512 tokens : au-delà, la fin du texte est ignorée par le modèle

PII = {
    "e-mail": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "téléphone": re.compile(r"(?<!\d)(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)"),
    "IBAN": re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b"),
    "carte bancaire": re.compile(r"(?<!\d)(?:\d{4}[\s-]?){3}\d{4}(?!\d)"),
    "n° de sécurité sociale": re.compile(r"(?<!\d)[12][\s]?\d{2}[\s]?\d{2}[\s]?\d{2}[\s]?\d{3}[\s]?\d{3}(?:[\s]?\d{2})?(?!\d)"),
}


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def check(path: str, taxonomy: dict | None, *, text_column: str, label_column: str) -> tuple[list, int]:
    examples = load_examples(path, text_column=text_column, label_column=label_column)
    errors = 0
    counts = Counter(ex.label for ex in examples)
    n_off = counts.pop(None, 0)
    print(f"{path} : {len(examples)} exemples, {len(counts)} catégories, {n_off} hors-sujet "
          f"({n_off / len(examples):.0%})\n")
    width = max(len(str(k)) for k in counts) if counts else 10
    for label, n in counts.most_common():
        flag = f"  ← moins de {MIN_PER_CLASS} exemples" if n < MIN_PER_CLASS else ""
        print(f"  {label:<{width}} {n:>5}{flag}")
    print()

    if taxonomy is not None:
        unknown = sorted(set(counts) - set(taxonomy))
        missing = sorted(set(taxonomy) - set(counts))
        if unknown:
            errors += 1
            print(f"ERREUR  étiquettes absentes de la taxonomie : {unknown}")
        if missing:
            print(f"ATTENTION  catégories sans aucun exemple : {missing}")
    if not 0.05 <= n_off / len(examples) <= 0.3:
        print("ATTENTION  visez 10 à 15 % de hors-sujet : sans eux, impossible de calibrer le rejet "
              "(min_entailment, juge).")

    seen: dict[str, list] = {}
    for i, ex in enumerate(examples):
        seen.setdefault(normalize(ex.text), []).append(i)
    dups = [idx for idx in seen.values() if len(idx) > 1]
    conflicts = [idx for idx in dups if len({examples[i].label for i in idx}) > 1]
    if dups:
        print(f"ATTENTION  {len(dups)} textes en double (à dédoublonner : ils faussent l'évaluation)")
    for idx in conflicts[:10]:
        labels = ", ".join(str(examples[i].label) for i in idx)
        print(f"ERREUR  même texte, étiquettes différentes ({labels}) : {examples[idx[0]].text[:70]!r}")
    errors += bool(conflicts)

    long = [ex for ex in examples if len(ex.text.split()) > MAX_WORDS]
    if long:
        print(f"ATTENTION  {len(long)} textes de plus de {MAX_WORDS} mots : la fin sera ignorée par le modèle")

    pii_hits = Counter()
    for ex in examples:
        for kind, pattern in PII.items():
            if pattern.search(ex.text):
                pii_hits[kind] += 1
    for kind, n in pii_hits.items():
        print(f"ERREUR  {n} textes semblent contenir un(e) {kind} : anonymisez avant de partager ou publier")
    errors += bool(pii_hits)
    return examples, errors


def agreement(a: list, b: list) -> None:
    from sklearn.metrics import cohen_kappa_score

    first = {normalize(ex.text): ex for ex in a}
    pairs = [(first[normalize(ex.text)], ex) for ex in b if normalize(ex.text) in first]
    if not pairs:
        print("\nAucun texte commun entre les deux fichiers.")
        return
    la = [str(x.label) for x, _ in pairs]
    lb = [str(y.label) for _, y in pairs]
    kappa = cohen_kappa_score(la, lb)
    raw = sum(x == y for x, y in zip(la, lb)) / len(pairs)
    verdict = "bon" if kappa >= 0.8 else "moyen : clarifiez le guide d'annotation" if kappa >= 0.6 else \
        "faible : les catégories sont ambiguës, revoyez la taxonomie"
    print(f"\nAccord entre annotateurs sur {len(pairs)} textes communs : {raw:.0%} brut, "
          f"kappa de Cohen {kappa:.2f} ({verdict})")
    pairs_diff = Counter((x, y) for x, y in zip(la, lb) if x != y)
    for (x, y), n in pairs_diff.most_common(8):
        print(f"  {n:>3} × {x} ↔ {y}")
    disagreements = [(x.text, x.label, y.label) for x, y in pairs if x.label != y.label]
    for text, x, y in disagreements[:10]:
        print(f"  à arbitrer : {text[:70]!r} ({x} / {y})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="fichier annoté (.csv, .tsv ou .jsonl)")
    p.add_argument("--taxonomy", help="JSON {etiquette: description}")
    p.add_argument("--second", help="annotation des mêmes textes par une autre personne")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label-column", default="label")
    args = p.parse_args()

    taxonomy = json.loads(Path(args.taxonomy).read_text(encoding="utf-8")) if args.taxonomy else None
    cols = dict(text_column=args.text_column, label_column=args.label_column)
    examples, errors = check(args.data, taxonomy, **cols)
    if args.second:
        print()
        second, errors_b = check(args.second, taxonomy, **cols)
        errors += errors_b
        agreement(examples, second)
    print("\n" + ("Des erreurs bloquantes sont à corriger." if errors else "Aucune erreur bloquante."))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
