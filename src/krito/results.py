"""Résultats structurés des primitives de décision et évaluation des garde-fous."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionScore:
    """Détail des scores NLI bruts et de la décision pour une option."""

    entailment: float
    contradiction: float
    neutral: float
    decision_score: float
    probability: float
    entailment_prob: float


@dataclass(frozen=True)
class ClassScore:
    """Scores d'une catégorie pour ``KritoClassifier`` (mode supervisé)."""

    logit: float
    probability: float
    similarity: float
    """Similarité cosinus au plus proche exemple d'entraînement de cette catégorie."""


@dataclass(frozen=True)
class DecisionResult:
    """Résultat structuré de la prise de décision probabiliste."""

    selected_key: str
    selected_label: str
    confidence: float
    margin: float
    accepted: bool
    rejection_reason: str | None
    scores: dict[str, OptionScore | ClassScore]
    judge_score: float | None = None
    """Probabilité que la décision soit bonne selon le juge de confiance, s'il y en a un."""


@dataclass(frozen=True, kw_only=True)
class ScaleResult(DecisionResult):
    """Résultat d'une échelle ordonnée : une décision, plus sa position sur l'échelle."""

    index: int
    """Rang du niveau retenu (0 = premier niveau)."""
    expected: float
    """Rang moyen pondéré par les probabilités : tient compte de l'ordre des niveaux."""
    spread: float
    """Écart-type de la distribution, en niveaux : faible quand le modèle hésite entre voisins."""


@dataclass(frozen=True)
class YesNoResult:
    """Réponse à une question fermée, posée comme une affirmation à vérifier."""

    statement: str
    answer: bool
    probability: float
    """P(oui) : implication contre contradiction, sans tenir compte de « neutre »."""
    confidence: float
    """Probabilité de la réponse donnée : ``max(p, 1 - p)``."""
    neutral_prob: float
    """P(neutre) sur les 3 classes : élevée quand le texte ne permet pas de trancher."""
    accepted: bool
    rejection_reason: str | None
    entailment: float
    contradiction: float
    neutral: float


# Garde-fou -> (motif de rejet, sens). « min » : rejet si valeur < seuil ; « max » : si valeur > seuil.
GUARDRAILS = {
    "threshold": ("confidence_too_low", "min"),
    "min_margin": ("margin_too_low", "min"),
    "min_entailment": ("entailment_too_low", "min"),
    "min_similarity": ("similarity_too_low", "min"),
    "min_judge_score": ("judge_score_too_low", "min"),
    "max_spread": ("spread_too_high", "max"),
    "max_neutral": ("neutral_too_high", "max"),
}


def guardrail_value(result, name: str) -> float:
    """Valeur du signal qu'un garde-fou compare à son seuil, pour un résultat donné."""
    if name == "threshold":
        return result.confidence
    if name == "min_margin":
        return result.margin
    if name == "min_judge_score":
        if result.judge_score is None:
            raise ValueError("min_judge_score demande un juge de confiance (paramètre judge).")
        return result.judge_score
    if name == "max_neutral":
        return result.neutral_prob
    if name == "max_spread":
        return result.spread
    selected = result.scores[result.selected_key]
    if name == "min_entailment":
        if not isinstance(selected, OptionScore):
            raise ValueError("min_entailment ne s'applique qu'au moteur NLI (KritoEngine).")
        return selected.entailment_prob
    if name == "min_similarity":
        if not isinstance(selected, ClassScore):
            raise ValueError("min_similarity ne s'applique qu'au mode supervisé (KritoClassifier).")
        return selected.similarity
    raise ValueError(f"Garde-fou inconnu : {name}. Connus : {', '.join(GUARDRAILS)}.")


def rejection_reasons(values: dict[str, float], limits: dict[str, float | None]) -> list[str]:
    """Motifs de rejet, dans l'ordre de ``limits``, pour les seuils fournis (non ``None``)."""
    reasons = []
    for name, limit in limits.items():
        if limit is None:
            continue
        reason, direction = GUARDRAILS[name]
        value = values[name]
        if direction == "min" and value < limit:
            reasons.append(f"{reason} (<{limit})")
        elif direction == "max" and value > limit:
            reasons.append(f"{reason} (>{limit})")
    return reasons
