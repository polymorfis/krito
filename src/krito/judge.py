"""Juge de confiance optionnel.

Le juge est une petite régression logistique qui apprend, sur des exemples annotés, la
probabilité qu'une décision soit **bonne** (message du domaine *et* bonne option) à partir
des signaux d'une décision : confiance, marge, implication absolue, entropie… Il remplace
plusieurs seuils réglés à la main par un seul score, calibré sur vos données.

Dans l'étude du dépôt (experiments/RESULTS.md, section 4), un juge de ce type fait passer
l'AUROC d'acceptation du zero-shot de 0,76 à 0,91. Cette version n'utilise que les signaux du
modèle de décision lui-même : aucun second modèle à charger.

    judge = ConfidenceJudge().fit(engine.classify_batch(textes, options), etiquettes)
    engine.judge = judge
    engine.classify(texte, options, min_judge_score=0.8)
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .results import ClassScore, DecisionResult

def require_sklearn() -> None:
    """scikit-learn (extra ``krito[learn]``) n'est importé qu'à l'usage : le zero-shot n'en a pas besoin."""
    if importlib.util.find_spec("sklearn") is None:
        raise ImportError("Le juge de confiance et le mode supervisé demandent scikit-learn : "
                          'pip install "krito[learn]".')

SIGNALS = (
    "confidence",       # probabilité relative de l'option retenue
    "margin",           # écart avec la 2e option
    "selected_absolute",  # P(entailment) absolue (NLI) ou similarité au plus proche exemple (supervisé)
    "best_absolute",    # meilleure valeur absolue parmi toutes les options
    "selected_raw",     # score brut : E − C (NLI) ou logit (supervisé)
    "entropy",          # entropie normalisée de la distribution entre options (0 = certain, 1 = uniforme)
)


def logistic_to_dict(model) -> dict:
    """Paramètres d'une régression logistique scikit-learn, en JSON (sans pickle)."""
    return {"C": model.C, "classes": model.classes_.tolist(), "coef": model.coef_.tolist(),
            "intercept": model.intercept_.tolist()}


def logistic_from_dict(data: dict):
    """Reconstruit une régression logistique scikit-learn déjà entraînée."""
    require_sklearn()
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=data["C"])
    model.classes_ = np.array(data["classes"])
    model.coef_ = np.array(data["coef"], dtype=np.float64)
    model.intercept_ = np.array(data["intercept"], dtype=np.float64)
    model.n_features_in_ = model.coef_.shape[1]
    return model


def result_kind(result: DecisionResult) -> str:
    return "supervised" if isinstance(next(iter(result.scores.values())), ClassScore) else "nli"


def decision_signals(result: DecisionResult) -> np.ndarray:
    """Vecteur des signaux (``SIGNALS``) d'une décision."""
    scores = list(result.scores.values())
    probs = np.array([s.probability for s in scores])
    if isinstance(scores[0], ClassScore):
        absolute = [s.similarity for s in scores]
        raw = result.scores[result.selected_key].logit
        selected_abs = result.scores[result.selected_key].similarity
    else:
        absolute = [s.entailment_prob for s in scores]
        raw = result.scores[result.selected_key].decision_score
        selected_abs = result.scores[result.selected_key].entailment_prob
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum() / np.log(len(probs)))
    return np.array([result.confidence, result.margin, selected_abs, max(absolute), raw, entropy])


class ConfidenceJudge:
    """Prédit la probabilité qu'une décision soit bonne.

    Modèle : ``StandardScaler`` + ``LogisticRegression`` de scikit-learn.
    ``C`` : inverse de la régularisation ; le baisser si l'on a peu d'exemples (moins d'une centaine).
    """

    def __init__(self, C: float = 1.0):
        self.C = C
        self.kind: str | None = None
        self._model = None  # sklearn.pipeline.Pipeline

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, results: Sequence[DecisionResult], labels: Sequence[str | None]) -> ConfidenceJudge:
        """Entraîne le juge sur des décisions et leurs bonnes réponses.

        ``labels`` : la bonne option pour chaque décision, ``None`` pour un message hors-sujet
        (une décision acceptée sur un hors-sujet est toujours mauvaise). Les décisions doivent
        venir de données **non vues** par le modèle de décision, sinon le juge apprend une
        confiance trop optimiste.
        """
        if len(results) != len(labels):
            raise ValueError("Il faut autant d'étiquettes que de décisions.")
        if not results:
            raise ValueError("Aucune décision pour entraîner le juge.")
        kinds = {result_kind(r) for r in results}
        if len(kinds) > 1:
            raise ValueError("Toutes les décisions doivent venir du même type de modèle.")
        good = np.array([r.selected_key == label for r, label in zip(results, labels)], dtype=np.int64)
        if good.all() or not good.any():
            raise ValueError(
                "Le juge a besoin de bonnes ET de mauvaises décisions pour apprendre : "
                "ajoutez des exemples difficiles ou hors-sujet."
            )
        X = np.stack([decision_signals(r) for r in results])
        require_sklearn()
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._model = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000)).fit(X, good)
        self.kind = kinds.pop()
        return self

    def score(self, result: DecisionResult) -> float:
        """Probabilité que la décision soit bonne."""
        return float(self.score_many([result])[0])

    def score_many(self, results: Sequence[DecisionResult]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Le juge n'est pas entraîné : appelez fit() d'abord.")
        if not results:
            return np.zeros(0)
        kinds = {result_kind(r) for r in results}
        if kinds != {self.kind}:
            raise ValueError(f"Ce juge a été entraîné sur des décisions « {self.kind} », pas « {kinds.pop()} ».")
        proba = self._model.predict_proba(np.stack([decision_signals(r) for r in results]))
        return proba[:, list(self._model.classes_).index(1)]

    # ------------------------------------------------------------ sérialisation

    def to_dict(self) -> dict:
        if self._model is None:
            raise RuntimeError("Le juge n'est pas entraîné.")
        scaler, logistic = self._model[0], self._model[-1]
        return {
            "format": "krito-judge/1",
            "kind": self.kind,
            "signals": list(SIGNALS),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "logistic": logistic_to_dict(logistic),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConfidenceJudge:
        if data.get("format") != "krito-judge/1" or data.get("signals") != list(SIGNALS):
            raise ValueError("Format de juge non reconnu.")
        require_sklearn()
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = np.array(data["mean"], dtype=np.float64)
        scaler.scale_ = np.array(data["scale"], dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(scaler.mean_)
        logistic = logistic_from_dict(data["logistic"])
        judge = cls(logistic.C)
        judge.kind = data["kind"]
        judge._model = make_pipeline(scaler, logistic)
        return judge

    def save(self, path: str | os.PathLike) -> None:
        """Enregistre le juge en JSON (quelques centaines d'octets, sans pickle)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike) -> ConfidenceJudge:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
