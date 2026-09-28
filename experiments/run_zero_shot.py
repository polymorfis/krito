"""Méthodes sans entraînement, évaluées sur les mêmes plis de test que les méthodes supervisées.

- krito_* : scoring actuel de Krito (softmax des E − C entre options), confiance relative
- abs_*   : même prédiction, confiance absolue = P(entailment) de l'option retenue
- embzs_* : similarité cosinus texte / hypothèse, confiance = cosinus max
"""

from __future__ import annotations

import numpy as np

import features
from lib import evaluate, folds, load, mean_metrics, save, softmax


def nli_scores(logits: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ent, con = logits[..., 0], logits[..., 1]
    rel = softmax(ent - con, axis=1)
    pred = rel.argmax(1)
    p_ent = softmax(logits, axis=2)[..., 0]
    return {
        "krito": (pred, rel.max(1)),
        "abs": (pred, p_ent[np.arange(len(pred)), pred]),
    }


def emb_scores(emb: dict) -> tuple[np.ndarray, np.ndarray]:
    sims = emb["texts"] @ emb["hyps"].T
    return sims.argmax(1), sims.max(1)


def all_methods(dataset: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for key in features.NLI_MODELS:
        for variant, v in nli_scores(features.get("nli", key, dataset)["logits"]).items():
            out[f"{variant}_{key}"] = v
    for key in features.EMB_MODELS:
        out[f"embzs_{key}"] = emb_scores(features.get("emb", key, dataset))
    return out


def main() -> None:
    results = {}
    ds = load("support")
    y = ds.y
    for name, (pred, conf) in all_methods("support").items():
        per_fold = [evaluate(y[te], pred[te], conf[te]) for _, te in folds(ds)]
        results[name] = mean_metrics(per_fold)

    # Transfert : domaine RH, entier (rien n'est entraîné)
    rh = load("rh")
    transfer = {name: evaluate(rh.y, p, c) for name, (p, c) in all_methods("rh").items()}

    save("zero_shot", {"support_cv": results, "rh_transfer": transfer})
    print(f"{'méthode':24} {'acc':>5} {'f1':>5} {'auc_acc':>7} {'auc_ood':>7} {'cov@90':>6} {'ece':>5} | RH acc  auc_acc")
    for name, m in results.items():
        t = transfer[name]
        print(f"{name:24} {m['accuracy']:5.2f} {m['macro_f1']:5.2f} {m['auroc_accept']:7.2f} {m['auroc_ood']:7.2f}"
              f" {m['coverage_at_90']:6.2f} {m['ece']:5.2f} | {t['accuracy']:5.2f} {t['auroc_accept']:7.2f}")


if __name__ == "__main__":
    main()
