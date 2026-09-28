"""Outils communs aux expériences : données, découpage, métriques.

Protocole (identique pour toutes les méthodes) :
- validation croisée stratifiée 5 plis sur ``support_fr`` (graine fixe) ;
- les exemples ``none`` (hors-sujet) sont répartis dans les plis mais ne sont
  jamais une classe à prédire : ils servent à mesurer la capacité de rejet ;
- le domaine ``rh_fr`` n'est jamais utilisé à l'entraînement : il mesure le
  zero-shot sur des catégories inconnues.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"
SEED = 42
N_FOLDS = 5
NONE = "none"

# Hypothèses canoniques (utilisées à l'évaluation de toutes les méthodes NLI / zero-shot)
HYPOTHESES = {
    "support": {
        "billing": "Ce message concerne un problème de facturation, de paiement ou de remboursement.",
        "shipping": "Ce message concerne la livraison, le suivi ou le retour d'un colis.",
        "account": "Ce message concerne la connexion, le mot de passe ou l'accès au compte.",
        "cancel": "Ce message concerne une résiliation ou l'arrêt d'un abonnement.",
        "bug": "Ce message concerne un bug technique ou une panne du site ou de l'application.",
        "sales": "Ce message concerne les tarifs, les offres ou une demande commerciale.",
    },
    "rh": {
        "leave": "Ce message concerne les congés ou les absences.",
        "payroll": "Ce message concerne le salaire ou la fiche de paie.",
        "equipment": "Ce message concerne le matériel informatique ou l'équipement de travail.",
        "training": "Ce message concerne la formation professionnelle.",
        "expenses": "Ce message concerne les notes de frais ou les remboursements de dépenses.",
    },
}


@dataclass
class Dataset:
    name: str
    texts: list[str]
    labels: list[str]
    classes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.classes = sorted({l for l in self.labels if l != NONE})

    @property
    def y(self) -> np.ndarray:
        """Indice de classe, -1 pour hors-sujet."""
        idx = {c: i for i, c in enumerate(self.classes)}
        return np.array([idx.get(l, -1) for l in self.labels])

    @property
    def hypotheses(self) -> list[str]:
        return [HYPOTHESES[self.name][c] for c in self.classes]


def load(name: str) -> Dataset:
    with open(DATA / f"{name}_fr.tsv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return Dataset(name, [r["text"] for r in rows], [r["label"] for r in rows])


def folds(ds: Dataset):
    """Génère (train_idx, test_idx) ; stratifié sur le label y compris ``none``."""
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    yield from skf.split(np.zeros(len(ds.labels)), ds.labels)


def subsample_per_class(y: np.ndarray, idx: np.ndarray, k: int | None, seed: int) -> np.ndarray:
    """Garde au plus k exemples par classe (hors ``none``) parmi idx."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y[idx]):
        if c < 0:
            continue
        members = idx[y[idx] == c]
        if k is not None and len(members) > k:
            members = rng.choice(members, k, replace=False)
        keep.extend(members.tolist())
    return np.array(sorted(keep))


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------


def evaluate(y_true: np.ndarray, pred: np.ndarray, conf: np.ndarray) -> dict:
    """Métriques de classification et de décision sélective.

    y_true : indice de classe, -1 pour hors-sujet.
    pred   : classe prédite (toujours >= 0 : aucune méthode ne s'abstient d'elle-même).
    conf   : score de confiance, plus haut = plus sûr.

    Une décision est « bonne à accepter » si le message est dans le domaine et
    bien classé. Les erreurs et les hors-sujet devraient être rejetés.
    """
    ind = y_true >= 0
    correct = (pred == y_true) & ind
    out = {
        "accuracy": float(correct[ind].mean()),
        "macro_f1": float(f1_score(y_true[ind], pred[ind], average="macro")),
        # Capacité de la confiance à séparer « bonne décision » de « erreur ou hors-sujet »
        "auroc_accept": float(roc_auc_score(correct, conf)),
        # Capacité de la confiance à séparer dans-le-domaine de hors-sujet
        "auroc_ood": float(roc_auc_score(ind, conf)),
        "ece": expected_calibration_error(conf[ind], correct[ind]),
    }
    out["coverage_at_95"] = coverage_at_precision(correct, ind, conf, 0.95)
    out["coverage_at_90"] = coverage_at_precision(correct, ind, conf, 0.90)
    return out


def coverage_at_precision(correct, ind, conf, target: float) -> float:
    """Part maximale des messages du domaine acceptés tout en gardant
    une précision >= target parmi les décisions acceptées (hors-sujet acceptés
    comptés comme erreurs). Seuil « oracle » : sert à comparer les méthodes,
    pas à estimer une performance de production."""
    order = np.argsort(-conf, kind="stable")
    c = correct[order].astype(float)
    tp = np.cumsum(c)
    n = np.arange(1, len(c) + 1)
    precision = tp / n
    ok = np.where(precision >= target)[0]
    if len(ok) == 0:
        return 0.0
    return float(tp[ok[-1]] / ind.sum())


def expected_calibration_error(conf, correct, bins: int = 10) -> float:
    conf = np.clip(np.asarray(conf, dtype=float), 0, 1)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def mean_metrics(per_fold: list[dict]) -> dict:
    keys = per_fold[0].keys()
    return {k: float(np.mean([m[k] for m in per_fold])) for k in keys} | {
        f"{k}_std": float(np.std([m[k] for m in per_fold])) for k in ("accuracy", "auroc_accept")
    }


def save(name: str, payload: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
