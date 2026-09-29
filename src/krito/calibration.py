"""Calibration des garde-fous sur des exemples annotés.

L'échelle des signaux (``entailment_prob``, marge, score du juge…) dépend du modèle, du
gabarit et du domaine : un seuil ne se devine pas, il se mesure. ``calibrate`` cherche la
combinaison de seuils qui **automatise le plus de bonnes décisions** tout en gardant une
précision cible parmi les décisions acceptées. Les hors-sujet acceptés comptent comme des erreurs.

    cal = engine.calibrate(exemples, options, target_precision=0.95)
    print(cal)
    engine.classify(texte, options, **cal.guardrails)

Un seuil mesuré sur les exemples qui l'ont fixé est optimiste : vérifiez-le sur un second
jeu annoté (``split_examples`` puis ``evaluate_guardrails``).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .results import GUARDRAILS, guardrail_value


@dataclass(frozen=True)
class CalibrationPoint:
    """Effet d'une combinaison de seuils sur un jeu annoté."""

    guardrails: dict[str, float]
    precision: float
    """Part de bonnes décisions parmi les décisions acceptées (hors-sujet acceptés = erreurs)."""
    coverage: float
    """Part de tous les exemples dont la décision est acceptée."""
    automated: float
    """Part des exemples du domaine traités automatiquement ET correctement."""
    accepted: int


@dataclass(frozen=True)
class Calibration:
    """Seuils recommandés et compromis précision / automatisation."""

    guardrails: dict[str, float]
    """À passer tels quels : ``engine.classify(texte, options, **cal.guardrails)``."""
    precision: float
    coverage: float
    automated: float
    target_precision: float
    reached: bool
    """``False`` si aucune combinaison n'atteint la précision cible : les seuils donnent alors
    la meilleure précision possible."""
    n_examples: int
    n_in_domain: int
    frontier: tuple[CalibrationPoint, ...]
    """Compromis optimaux (plus de précision = moins d'automatisation), du plus automatisé au plus précis."""

    def __str__(self) -> str:
        names = list(dict.fromkeys(n for p in self.frontier for n in p.guardrails)) or list(self.guardrails)
        head = "".join(f"{n:>17}" for n in names) + f"{'précision':>11}{'acceptés':>10}{'automatisé':>12}"
        lines = [
            f"{self.n_examples} exemples annotés, dont {self.n_examples - self.n_in_domain} hors-sujet",
            head,
        ]
        for p in self.frontier:
            cells = "".join(f"{p.guardrails[n]:>17.3f}" if n in p.guardrails else f"{'—':>17}" for n in names)
            mark = "  ◄" if p.guardrails == self.guardrails else ""
            lines.append(f"{cells}{p.precision:>11.1%}{p.coverage:>10.1%}{p.automated:>12.1%}{mark}")
        if self.reached:
            lines.append(f"Recommandé pour {self.target_precision:.0%} de précision : {self._kwargs()}")
        else:
            lines.append(f"Précision cible {self.target_precision:.0%} non atteinte : meilleure précision "
                         f"possible avec {self._kwargs()}. Fine-tunez le modèle ou baissez la cible.")
        return "\n".join(lines)

    def _kwargs(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.guardrails.items()) or "aucun garde-fou"


def _signals(results: Sequence, name: str) -> np.ndarray:
    if name not in GUARDRAILS:
        raise ValueError(f"Garde-fou inconnu : {name}. Connus : {', '.join(GUARDRAILS)}.")
    return np.array([guardrail_value(r, name) for r in results], dtype=np.float64)


def _accepts(values: np.ndarray, name: str, limit: float | None) -> np.ndarray:
    if limit is None:
        return np.ones(len(values), dtype=bool)
    return values <= limit if GUARDRAILS[name][1] == "max" else values >= limit


def _good(results: Sequence, labels: Sequence[str | None]) -> tuple[np.ndarray, int]:
    if len(results) != len(labels):
        raise ValueError("Il faut autant d'étiquettes que de décisions.")
    if not results:
        raise ValueError("Aucune décision à calibrer.")
    good = np.array([r.selected_key == label for r, label in zip(results, labels)])
    n_in = sum(label is not None and label in r.scores for r, label in zip(results, labels))
    return good, n_in


def _point(guardrails: dict[str, float], accepted: np.ndarray, good: np.ndarray, n_in: int) -> CalibrationPoint:
    n_acc = int(accepted.sum())
    n_good = int((accepted & good).sum())
    return CalibrationPoint(
        guardrails=guardrails,
        precision=n_good / n_acc if n_acc else 0.0,
        coverage=n_acc / len(good),
        automated=n_good / n_in if n_in else 0.0,
        accepted=n_acc,
    )


def evaluate_guardrails(results: Sequence, labels: Sequence[str | None], **guardrails: float | None) -> CalibrationPoint:
    """Mesure l'effet de seuils donnés sur des décisions annotées (ex. un jeu de validation)."""
    good, n_in = _good(results, labels)
    accepted = np.ones(len(good), dtype=bool)
    for name, limit in guardrails.items():
        accepted &= _accepts(_signals(results, name), name, limit)
    return _point({k: v for k, v in guardrails.items() if v is not None}, accepted, good, n_in)


def _candidates(values: np.ndarray, name: str, grid_size: int) -> list[float | None]:
    """Seuils candidats : quantiles des valeurs observées, arrondis vers le côté permissif."""
    q = np.quantile(values, np.linspace(0, 1, grid_size + 1))
    if GUARDRAILS[name][1] == "max":
        q = np.ceil(q * 1000) / 1000
    else:
        q = np.floor(q * 1000) / 1000
    return [None, *(float(v) for v in np.unique(q))]


def calibrate(
    results: Sequence,
    labels: Sequence[str | None],
    *,
    target_precision: float = 0.95,
    guardrails: Sequence[str] = ("min_entailment", "min_margin"),
    grid_size: int = 20,
) -> Calibration:
    """Cherche les seuils qui maximisent l'automatisation à précision cible.

    results : décisions **sans garde-fou** (``classify_batch`` sans seuil) ;
    labels : bonne option de chaque décision, ``None`` pour un hors-sujet ;
    guardrails : noms des garde-fous à régler (1 à 3 conseillés), parmi
        ``threshold``, ``min_margin``, ``min_entailment``, ``min_similarity``,
        ``min_judge_score``, ``max_spread`` ;
    grid_size : nombre de quantiles essayés par garde-fou.
    """
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision doit être compris entre 0 et 1.")
    if not guardrails:
        raise ValueError("Indiquez au moins un garde-fou à calibrer.")
    guardrails = list(dict.fromkeys(guardrails))
    good, n_in = _good(results, labels)
    values = {name: _signals(results, name) for name in guardrails}
    grids = {name: _candidates(values[name], name, grid_size) for name in guardrails}
    masks = {name: [_accepts(values[name], name, c) for c in grids[name]] for name in guardrails}

    points = []
    for combo in itertools.product(*(range(len(grids[n])) for n in guardrails)):
        accepted = np.logical_and.reduce([masks[n][i] for n, i in zip(guardrails, combo)])
        chosen = {n: grids[n][i] for n, i in zip(guardrails, combo) if grids[n][i] is not None}
        points.append(_point(chosen, accepted, good, n_in))

    ok = [p for p in points if p.accepted and p.precision >= target_precision]
    if ok:
        # Le plus de bonnes décisions automatisées ; à égalité, la meilleure précision, puis le moins de seuils.
        best = max(ok, key=lambda p: (p.automated, p.precision, -len(p.guardrails)))
    else:
        best = max(points, key=lambda p: (p.precision, p.automated, -len(p.guardrails)))

    frontier = []
    for p in sorted(points, key=lambda p: (-p.automated, -p.precision, len(p.guardrails))):
        if p.accepted and (not frontier or p.precision > frontier[-1].precision):
            frontier.append(p)
    if best not in frontier:
        frontier.append(best)
        frontier.sort(key=lambda p: (-p.automated, -p.precision))

    return Calibration(
        guardrails=best.guardrails,
        precision=best.precision,
        coverage=best.coverage,
        automated=best.automated,
        target_precision=target_precision,
        reached=bool(ok),
        n_examples=len(results),
        n_in_domain=n_in,
        frontier=tuple(frontier),
    )
