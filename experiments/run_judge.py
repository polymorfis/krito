"""Couche de confiance : juge appris + prédiction conforme.

Le juge est une régression logistique qui prédit « cette décision est bonne »
(message du domaine ET bien classé) à partir de signaux issus de plusieurs modèles.
Sa probabilité remplace la confiance brute pour accepter ou rejeter.

Pipelines :
- zs     : zero-shot pur. Prédiction = moyenne des probabilités NLI (mDeBERTa) et
           similarité d'embeddings (e5-base). Aucun modèle entraîné sur le domaine, seul
           le juge l'est. Transfert RH : juge entraîné sur le support, appliqué au RH.
- ft_*   : NLI fine-tuné (logits hors pli) + embeddings zero-shot.
- sup    : embeddings e5-base + régression logistique (catégories fixes), juge entraîné
           en validation croisée imbriquée ; + prédiction conforme.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features
from lib import CACHE, SEED, evaluate, expected_calibration_error, folds, load, mean_metrics, save, softmax

EMB_TAU = 0.02  # température de la similarité cosinus -> probabilités


def zs_signals(nli_logits: np.ndarray, cos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prédiction d'ensemble et matrice de signaux pour le juge."""
    ent, con = nli_logits[..., 0], nli_logits[..., 1]
    p_nli = softmax(ent - con, axis=1)
    p_emb = softmax(cos / EMB_TAU, axis=1)
    p = (p_nli + p_emb) / 2
    pred = p.argmax(1)
    r = np.arange(len(pred))
    p_ent_abs = softmax(nli_logits, axis=2)[..., 0]
    srt = np.sort(p, axis=1)
    cos_srt = np.sort(cos, axis=1)
    X = np.column_stack([
        p[r, pred],                          # confiance d'ensemble
        srt[:, -1] - srt[:, -2],             # marge d'ensemble
        p_nli[r, pred],                      # confiance NLI relative
        p_ent_abs[r, pred],                  # P(entailment) absolue de l'option retenue
        p_ent_abs.max(1),                    # meilleure implication absolue
        ent[r, pred] - con[r, pred],         # logit de décision brut
        cos[r, pred],                        # similarité au libellé retenu
        cos_srt[:, -1] - cos_srt[:, -2],     # marge de similarité
        (p_nli.argmax(1) == p_emb.argmax(1)).astype(float),  # accord NLI / embeddings
        -(p * np.log(p + 1e-12)).sum(1),     # entropie
    ])
    return pred, X


def make_judge():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def good(y, pred):
    return ((pred == y) & (y >= 0)).astype(int)


def run_zero_shot_family(name: str, sup_logits: np.ndarray, rh_logits: np.ndarray) -> dict:
    sup, rh = load("support"), load("rh")
    e_sup, e_rh = features.get("emb", "e5_base", "support"), features.get("emb", "e5_base", "rh")
    pred, X = zs_signals(sup_logits, e_sup["texts"] @ e_sup["hyps"].T)
    pred_rh, X_rh = zs_signals(rh_logits, e_rh["texts"] @ e_rh["hyps"].T)
    y, y_rh = sup.y, rh.y

    raw, judged, judge_ece = [], [], []
    for tr, te in folds(sup):
        j = make_judge().fit(X[tr], good(y[tr], pred[tr]))
        pj = j.predict_proba(X[te])[:, 1]
        raw.append(evaluate(y[te], pred[te], X[te, 0]))
        judged.append(evaluate(y[te], pred[te], pj))
        judge_ece.append(expected_calibration_error(pj, good(y[te], pred[te])))
    j = make_judge().fit(X, good(y, pred))
    pj_rh = j.predict_proba(X_rh)[:, 1]
    return {
        f"{name}": {"support_cv": mean_metrics(raw), "rh": evaluate(y_rh, pred_rh, X_rh[:, 0])},
        f"{name}+juge": {
            "support_cv": mean_metrics(judged) | {"judge_ece_all": float(np.mean(judge_ece))},
            "rh": evaluate(y_rh, pred_rh, pj_rh)
            | {"judge_ece_all": expected_calibration_error(pj_rh, good(y_rh, pred_rh))},
        },
    }


# --------------------------------------------------------------------------
# Pipeline supervisé + juge (CV imbriquée) + prédiction conforme
# --------------------------------------------------------------------------


def sup_fit_predict(X_tr, y_tr, X_te):
    clf = LogisticRegression(C=10, max_iter=2000).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    full = np.zeros((len(X_te), 6))
    full[:, clf.classes_] = proba
    return full


def sup_signals(proba, X_te, X_tr, y_tr, nli_logits, cos_h):
    pred = proba.argmax(1)
    r = np.arange(len(pred))
    srt = np.sort(proba, axis=1)
    sims = X_te @ X_tr.T
    knn_same = np.array([sims[i, y_tr == p].max() for i, p in enumerate(pred)])
    p_ent_abs = softmax(nli_logits, axis=2)[..., 0]
    feats = np.column_stack([
        srt[:, -1], srt[:, -1] - srt[:, -2],
        knn_same, sims.max(1), np.sort(sims, axis=1)[:, -5:].mean(1),
        p_ent_abs[r, pred], cos_h[r, pred],
    ])
    return pred, feats


def run_supervised() -> dict:
    sup = load("support")
    y = sup.y
    E = features.get("emb", "e5_base", "support")
    X, cos_h = E["texts"], E["texts"] @ E["hyps"].T
    nli = features.get("nli", "mdeberta", "support")["logits"]
    alpha = 0.1

    raw, judged, conf_stats = [], [], []
    for fi, (tr, te) in enumerate(folds(sup)):
        tr_in = tr[y[tr] >= 0]
        # Signaux « hors pli » sur le pli d'entraînement (y compris hors-sujet) pour entraîner le juge
        F_tr, G_tr, P_tr = [], [], []
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + fi)
        for itr, ite in inner.split(tr, np.where(y[tr] >= 0, y[tr], 99)):
            a, b = tr[itr], tr[ite]
            a_in = a[y[a] >= 0]
            proba = sup_fit_predict(X[a_in], y[a_in], X[b])
            pred, f = sup_signals(proba, X[b], X[a_in], y[a_in], nli[b], cos_h[b])
            F_tr.append(f); G_tr.append(good(y[b], pred)); P_tr.append((b, proba))
        judge = make_judge().fit(np.vstack(F_tr), np.concatenate(G_tr))

        proba_te = sup_fit_predict(X[tr_in], y[tr_in], X[te])
        pred_te, f_te = sup_signals(proba_te, X[te], X[tr_in], y[tr_in], nli[te], cos_h[te])
        raw.append(evaluate(y[te], pred_te, f_te[:, 0]))
        judged.append(evaluate(y[te], pred_te, judge.predict_proba(f_te)[:, 1]))

        # Prédiction conforme (split) : scores de non-conformité hors pli sur les exemples du domaine
        scores = []
        for b, proba in P_tr:
            m = y[b] >= 0
            scores.extend(1 - proba[m, y[b][m]])
        n = len(scores)
        qhat = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        sets = (1 - proba_te) <= qhat
        ind = y[te] >= 0
        size = sets.sum(1)
        conf_stats.append({
            "coverage": float(sets[ind, y[te][ind]].mean()),
            "mean_set_size": float(size[ind].mean()),
            "singleton_rate": float((size[ind] == 1).mean()),
            "singleton_accuracy": float((proba_te[ind].argmax(1) == y[te][ind])[size[ind] == 1].mean()),
            "ood_singleton_rate": float((size[~ind] == 1).mean()),
        })
    conformal = {k: float(np.mean([c[k] for c in conf_stats])) for k in conf_stats[0]}
    return {
        "sup_e5_base": {"support_cv": mean_metrics(raw)},
        "sup_e5_base+juge": {"support_cv": mean_metrics(judged)},
        "conformal_alpha_0.1": conformal,
    }


def main() -> None:
    out = {}
    out |= run_zero_shot_family(
        "zs_mdeberta+e5",
        features.get("nli", "mdeberta", "support")["logits"],
        features.get("nli", "mdeberta", "rh")["logits"],
    )
    for key in ("minilm_multi", "mdeberta"):
        p_sup, p_rh = CACHE / f"nliFT_{key}_support_oof.npz", CACHE / f"nliFT_{key}_rh.npz"
        if p_sup.exists():
            out |= run_zero_shot_family(f"ft_{key}+e5", np.load(p_sup)["logits"], np.load(p_rh)["logits"])
    out |= run_supervised()
    save("judge", out)

    print(f"{'pipeline':26} {'acc':>5} {'auc_acc':>7} {'auc_ood':>7} {'cov@90':>6} {'cov@95':>6} | RH acc auc_acc cov@90")
    for name, v in out.items():
        if "support_cv" not in v:
            continue
        m = v["support_cv"]
        line = (f"{name:26} {m['accuracy']:5.2f} {m['auroc_accept']:7.2f} {m['auroc_ood']:7.2f}"
                f" {m['coverage_at_90']:6.2f} {m['coverage_at_95']:6.2f}")
        if "rh" in v:
            t = v["rh"]
            line += f" | {t['accuracy']:5.2f} {t['auroc_accept']:7.2f} {t['coverage_at_90']:6.2f}"
        print(line)
    print("conforme (alpha=0.1):", {k: round(v, 3) for k, v in out["conformal_alpha_0.1"].items()})


if __name__ == "__main__":
    main()
