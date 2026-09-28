"""Méthodes supervisées (catégories fixes) avec courbe d'apprentissage.

Pour chaque pli, on garde k exemples par classe du pli d'entraînement
(k = 4, 8, 16, tous ≈ 32). Les hors-sujet ne sont jamais utilisés à l'entraînement.

- tfidf_lr : TF-IDF mots + caractères, régression logistique
- emblr_*  : embeddings figés + régression logistique, confiance = proba max
- embknn_* : même prédiction, confiance = cosinus au plus proche exemple d'entraînement
             de la classe prédite (meilleure détection des hors-sujet)
- setfit   : fine-tuning contrastif de e5-small (façon SetFit) + régression logistique
"""

from __future__ import annotations

import argparse
import copy
import random
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

import features
from lib import SEED, evaluate, folds, load, mean_metrics, save, subsample_per_class

KS = [4, 8, 16, None]


def tfidf_lr(texts_tr, y_tr, texts_te):
    clf = make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode", lowercase=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=True),
        ),
        LogisticRegression(C=10, max_iter=2000),
    )
    clf.fit(texts_tr, y_tr)
    proba = clf.predict_proba(texts_te)
    return clf.classes_[proba.argmax(1)], proba.max(1)


def emb_lr(X_tr, y_tr, X_te):
    clf = LogisticRegression(C=10, max_iter=2000).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    pred = clf.classes_[proba.argmax(1)]
    sims = X_te @ X_tr.T
    knn_conf = np.array([sims[i, y_tr == p].max() for i, p in enumerate(pred)])
    return pred, proba.max(1), knn_conf


# --------------------------------------------------------------------------
# SetFit minimal : paires contrastives + perte cosinus, puis régression logistique
# --------------------------------------------------------------------------

_BASE_ST = None


def setfit_embed(texts_tr, y_tr, texts_all, epochs: int = 1, pairs_per_sample: int = 20):
    from sentence_transformers import SentenceTransformer

    global _BASE_ST
    if _BASE_ST is None:
        _BASE_ST = SentenceTransformer(features.EMB_MODELS["e5_small"], device="cpu")
    model = copy.deepcopy(_BASE_ST).to("cuda")
    prefix = "query: "
    rng = random.Random(SEED)
    by_class = {c: [t for t, l in zip(texts_tr, y_tr) if l == c] for c in set(y_tr)}
    pairs = []
    for t, l in zip(texts_tr, y_tr):
        for j in range(pairs_per_sample):
            if j % 2 == 0:
                pairs.append((t, rng.choice(by_class[l]), 1.0))
            else:
                other = rng.choice([c for c in by_class if c != l])
                pairs.append((t, rng.choice(by_class[other]), 0.0))
    rng.shuffle(pairs)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    bs = 16
    for _ in range(epochs):
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            a = model.tokenize([prefix + p[0] for p in batch])
            b = model.tokenize([prefix + p[1] for p in batch])
            a = {k: v.to("cuda") for k, v in a.items() if torch.is_tensor(v)}
            b = {k: v.to("cuda") for k, v in b.items() if torch.is_tensor(v)}
            ea = model(a)["sentence_embedding"]
            eb = model(b)["sentence_embedding"]
            cos = torch.nn.functional.cosine_similarity(ea, eb)
            target = torch.tensor([p[2] for p in batch], device="cuda")
            loss = torch.nn.functional.mse_loss(cos, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        emb = model.encode([prefix + t for t in texts_all], normalize_embeddings=True, batch_size=32)
    del model, opt
    torch.cuda.empty_cache()
    return emb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-setfit", action="store_true")
    args = p.parse_args()

    ds = load("support")
    y = ds.y
    texts = np.array(ds.texts)
    embs = {k: features.get("emb", k, "support")["texts"] for k in features.EMB_MODELS}

    results: dict[str, dict] = {}
    for k in KS:
        tag = f"k{k or 'all'}"
        per = {}
        for fold_i, (tr, te) in enumerate(folds(ds)):
            tr_k = subsample_per_class(y, tr, k, seed=SEED + fold_i)
            pred, conf = tfidf_lr(texts[tr_k].tolist(), y[tr_k], texts[te].tolist())
            per.setdefault("tfidf_lr", []).append(evaluate(y[te], pred, conf))
            for ek, X in embs.items():
                pred, conf, knn = emb_lr(X[tr_k], y[tr_k], X[te])
                per.setdefault(f"emblr_{ek}", []).append(evaluate(y[te], pred, conf))
                per.setdefault(f"embknn_{ek}", []).append(evaluate(y[te], pred, knn))
            if not args.skip_setfit:
                t0 = time.perf_counter()
                emb = setfit_embed(texts[tr_k].tolist(), y[tr_k].tolist(), ds.texts)
                pred, conf, knn = emb_lr(emb[tr_k], y[tr_k], emb[te])
                per.setdefault("setfit_e5_small", []).append(evaluate(y[te], pred, conf))
                per.setdefault("setfitknn_e5_small", []).append(evaluate(y[te], pred, knn))
                print(f"  setfit {tag} pli {fold_i} : {time.perf_counter() - t0:.0f}s", flush=True)
        for name, lst in per.items():
            results[f"{name}@{tag}"] = mean_metrics(lst)
        print(f"--- {tag}", flush=True)
        for name in per:
            m = results[f"{name}@{tag}"]
            print(f"{name:22} acc={m['accuracy']:.2f}±{m['accuracy_std']:.2f} f1={m['macro_f1']:.2f}"
                  f" auc_acc={m['auroc_accept']:.2f} auc_ood={m['auroc_ood']:.2f}"
                  f" cov@90={m['coverage_at_90']:.2f} cov@95={m['coverage_at_95']:.2f} ece={m['ece']:.2f}", flush=True)

    save("supervised" + ("_nosetfit" if args.skip_setfit else ""), results)


if __name__ == "__main__":
    main()
