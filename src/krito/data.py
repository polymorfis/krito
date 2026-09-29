"""Exemples annotés : chargement, normalisation et découpage.

Un exemple est un couple ``(texte, étiquette)``. L'étiquette ``None`` marque un message
**hors-sujet** (aucune option ne convient) : il ne doit jamais être accepté. Dans les fichiers,
une étiquette vide ou ``none`` a le même sens.

Formats de fichier reconnus (colonnes ``text`` et ``label`` par défaut) :
- ``.csv`` et ``.tsv``, avec une ligne d'en-tête ;
- ``.jsonl`` : un objet JSON par ligne.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

OFF_TOPIC_LABELS = frozenset({"", "none", "hors-sujet", "hors_sujet"})


class Example(NamedTuple):
    """Un texte et sa bonne réponse ; ``label=None`` pour un message hors-sujet."""

    text: str
    label: str | None


def _normalize_label(label) -> str | None:
    if label is None:
        return None
    label = str(label).strip()
    return None if label.lower() in OFF_TOPIC_LABELS else label


def as_examples(examples) -> list[Example]:
    """Normalise des exemples annotés en une liste d'``Example``.

    Accepte :
    - des couples ``(texte, étiquette)`` ou des ``Example`` ;
    - des dictionnaires ``{"text": ..., "label": ...}`` ;
    - un dictionnaire ``{étiquette: [textes...]}`` (``None`` ou ``"none"`` pour les hors-sujet).
    """
    if isinstance(examples, Mapping):
        items = [(text, label) for label, texts in examples.items() for text in _texts(texts, label)]
    else:
        items = []
        for ex in examples:
            if isinstance(ex, Mapping):
                items.append((ex.get("text"), ex.get("label")))
            elif isinstance(ex, (tuple, list)) and len(ex) == 2:
                items.append((ex[0], ex[1]))
            else:
                raise ValueError(f"Exemple non reconnu : {ex!r}. Attendu : (texte, étiquette).")
    out = []
    for text, label in items:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Chaque exemple doit avoir un texte non vide.")
        out.append(Example(text, _normalize_label(label)))
    if not out:
        raise ValueError("Aucun exemple annoté.")
    return out


def _texts(texts, label) -> list[str]:
    if isinstance(texts, str):
        raise ValueError(f"L'étiquette « {label} » doit être associée à une liste de textes, pas à une chaîne.")
    return list(texts)


def load_examples(
    path: str | os.PathLike, *, text_column: str = "text", label_column: str = "label"
) -> list[Example]:
    """Charge un fichier annoté ``.csv``, ``.tsv`` ou ``.jsonl``."""
    path = Path(path)
    suffix = path.suffix.lower()
    with open(path, encoding="utf-8", newline="") as f:
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in f if line.strip()]
        elif suffix in (".csv", ".tsv"):
            rows = list(csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ","))
        else:
            raise ValueError(f"Format non reconnu : {path.name} (attendu .csv, .tsv ou .jsonl).")
    if rows and (text_column not in rows[0] or label_column not in rows[0]):
        raise ValueError(
            f"{path.name} doit contenir les colonnes « {text_column} » et « {label_column} » "
            f"(trouvées : {sorted(rows[0])})."
        )
    return as_examples([(r[text_column], r[label_column]) for r in rows])


def split_examples(
    examples: Iterable, test_size: float = 0.3, *, seed: int = 42
) -> tuple[list[Example], list[Example]]:
    """Découpe stratifiée par étiquette (hors-sujet compris), reproductible.

    Utile pour calibrer les seuils ou entraîner un juge sur une partie des données et
    **vérifier** le résultat sur l'autre : un seuil mesuré sur les mêmes exemples que ceux
    qui l'ont fixé est toujours optimiste.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être compris entre 0 et 1.")
    examples = as_examples(examples)
    rng = random.Random(seed)
    by_label: dict[str | None, list[Example]] = {}
    for ex in examples:
        by_label.setdefault(ex.label, []).append(ex)
    train, test = [], []
    for group in by_label.values():
        group = group[:]
        rng.shuffle(group)
        n_test = round(len(group) * test_size) if len(group) > 1 else 0
        test += group[:n_test]
        train += group[n_test:]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def labels_of(examples: Iterable[Example]) -> list[str]:
    """Étiquettes distinctes du domaine (hors-sujet exclus), dans l'ordre d'apparition."""
    return list(dict.fromkeys(ex.label for ex in examples if ex.label is not None))
