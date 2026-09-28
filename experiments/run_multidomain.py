"""Étude multi-domaines : zero-shot réel (un domaine exclu) et écart synthétique → gold.

Entraînement : uniquement les textes générés par le LLM (data/multi, filtrés par accord
du second LLM). Test : textes générés du domaine exclu, jeux gold (autre auteur), et
les jeux support / RH de l'étude précédente (jamais utilisés à l'entraînement).

    uv run python run_multidomain.py                 # tout
    uv run python run_multidomain.py --skip-lodo     # sans les 12 entraînements « domaine exclu »
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import features
from lib import CACHE, DATA, RESULTS, SEED, evaluate, softmax
from run_nli_finetune import NLI
from run_nli_finetune import descriptions as legacy_descriptions
from run_zero_shot import nli_scores

CANONICAL = "Ce texte concerne {}."
TEMPLATES = [
    "Ce texte concerne {}.",
    "Ce message concerne {}.",
    "Le sujet de ce texte est {}.",
    "Il s'agit de {}.",
    "Ce texte porte sur {}.",
]
MODEL = features.NLI_MODELS["mdeberta"]
MAX_LEN = 256  # les e-mails longs dépassent 128 tokens


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------


def load_domains() -> dict[str, dict]:
    """Chaque domaine : descriptions des classes, textes d'entraînement et jeux de test."""
    taxo = json.loads((DATA / "taxonomy.json").read_text())
    gold: dict[str, list] = {}
    with open(DATA / "gold_fr.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            gold.setdefault(r["domain"], []).append((r["label"], r["text"]))
    doms = {}
    for d, t in taxo.items():
        with open(DATA / "multi" / f"{d}.tsv", encoding="utf-8") as f:
            gen = [(r["label"], r["text"]) for r in csv.DictReader(f, delimiter="\t") if r["agree"] == "1"]
        doms[d] = {"classes": t["classes"], "gen": gen, "gold": gold[d]}
    for d in ("support", "rh"):  # jeux de l'étude précédente : test uniquement
        with open(DATA / f"{d}_fr.tsv", encoding="utf-8") as f:
            rows = [(r["label"], r["text"]) for r in csv.DictReader(f, delimiter="\t")]
        doms[d] = {"classes": legacy_descriptions(d), "gen": [], "gold": rows}
    return doms


def xy(dom: dict, split: str) -> tuple[list[str], np.ndarray, list[str]]:
    keys = list(dom["classes"])
    idx = {k: i for i, k in enumerate(keys)}
    rows = dom[split]
    return [t for _, t in rows], np.array([idx.get(l, -1) for l, _ in rows]), keys


def build_pairs(doms: dict, names: list[str], rng: random.Random, n_neg: int = 2):
    pairs = []
    for d in names:
        desc = doms[d]["classes"]
        keys = list(desc)
        for label, text in doms[d]["gen"]:
            negs = keys if label == "none" else [k for k in keys if k != label]
            if label != "none":
                pairs.append((text, rng.choice(TEMPLATES).format(desc[label]), 0))
            for k in rng.sample(negs, min(n_neg, len(negs))):
                pairs.append((text, rng.choice(TEMPLATES).format(desc[k]), 1))
    rng.shuffle(pairs)
    return pairs


# --------------------------------------------------------------------------
# Évaluation
# --------------------------------------------------------------------------


def eval_nli(nli: NLI, dom: dict, split: str, template: str = CANONICAL) -> dict:
    texts, y, keys = xy(dom, split)
    hyps = [template.format(dom["classes"][k]) for k in keys]
    pred, conf = nli_scores(nli.logits(texts, hyps))["abs"]
    return evaluate(y, pred, conf)


def eval_emb(st, dom: dict, split: str) -> dict:
    texts, y, keys = xy(dom, split)
    E = st.encode(["query: " + t for t in texts], normalize_embeddings=True, batch_size=64)
    H = st.encode(["query: " + CANONICAL.format(dom["classes"][k]) for k in keys], normalize_embeddings=True)
    sims = E @ H.T
    return evaluate(y, sims.argmax(1), sims.max(1))


def eval_supervised(st, dom: dict) -> dict:
    """e5-base + régression logistique entraînée sur les textes générés, testée sur le gold."""
    tr_texts, tr_y, _ = xy(dom, "gen")
    te_texts, te_y, _ = xy(dom, "gold")
    m = tr_y >= 0
    Xtr = st.encode(["query: " + t for t in np.array(tr_texts)[m]], normalize_embeddings=True, batch_size=64)
    Xte = st.encode(["query: " + t for t in te_texts], normalize_embeddings=True, batch_size=64)
    clf = LogisticRegression(C=10, max_iter=3000).fit(Xtr, tr_y[m])
    pred = clf.predict(Xte)
    sims = Xte @ Xtr.T
    knn = np.array([sims[i, tr_y[m] == p].max() for i, p in enumerate(pred)])
    return evaluate(te_y, pred, knn)


def train(doms: dict, names: list[str], seed: int, epochs: int) -> NLI:
    torch.manual_seed(seed)
    nli = NLI(MODEL, fp32=True, max_length=MAX_LEN)
    t0 = time.perf_counter()
    pairs = build_pairs(doms, names, random.Random(seed))
    nli.train(pairs, epochs=epochs)
    print(f"  entraîné sur {len(names)} domaines, {len(pairs)} paires, {time.perf_counter() - t0:.0f}s", flush=True)
    return nli


def fmt(m: dict) -> str:
    return f"acc={m['accuracy']:.2f} auc_ood={m['auroc_ood']:.2f} cov@90={m['coverage_at_90']:.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-lodo", action="store_true")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lodo-only", action="store_true", help="Second processus : uniquement les entraînements")
    args = p.parse_args()
    doms = load_domains()
    new = [d for d in doms if doms[d]["gen"]]
    out_path = RESULTS / "multidomain.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}

    def save():
        # Plusieurs processus (un par GPU) peuvent écrire : on fusionne avec la version sur disque.
        with open(CACHE / "multidomain.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            disk = json.loads(out_path.read_text()) if out_path.exists() else {}
            for k, v in res.items():
                if k == "lodo":
                    disk.setdefault("lodo", {}).update(v)
                else:
                    disk[k] = v
            out_path.write_text(json.dumps(disk, indent=2, ensure_ascii=False))
            res.update(disk)

    def claim(name: str) -> bool:
        """Réserve un entraînement pour ce processus (création exclusive d'un fichier)."""
        try:
            os.close(os.open(CACHE / f"claim_{name}", os.O_CREAT | os.O_EXCL))
            return True
        except FileExistsError:
            return False

    print({d: (len(doms[d]["gen"]), len(doms[d]["gold"])) for d in doms}, flush=True)

    # 1. Zero-shot (aucun entraînement)
    if not args.lodo_only and "zs_mdeberta" not in res:
        base = NLI(MODEL, fp32=True, max_length=MAX_LEN)
        res["zs_mdeberta"] = {d: {"gold": eval_nli(base, doms[d], "gold")} |
                              ({"gen": eval_nli(base, doms[d], "gen")} if doms[d]["gen"] else {}) for d in doms}
        del base
        save()
    if not args.lodo_only and ("zs_e5_base" not in res or "sup_e5_base" not in res):
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(features.EMB_MODELS["e5_base"], device="cuda")
        res["zs_e5_base"] = {d: {"gold": eval_emb(st, doms[d], "gold")} for d in doms}
        res["sup_e5_base"] = {d: {"gold": eval_supervised(st, doms[d])} for d in new}
        del st
        save()

    # 2. Modèle de l'étude précédente (fine-tuné sur le support seul)
    if not args.lodo_only and "ft_support_only" not in res:
        old = CACHE.parent / "models" / "krito-nli-fr-mdeberta" / "hf"
        if old.exists():
            nli = NLI(str(old), fp32=True, max_length=MAX_LEN)
            res["ft_support_only"] = {d: {"gold": eval_nli(nli, doms[d], "gold", "Ce message concerne {}.")}
                                      for d in doms if d != "support"}
            del nli
            save()

    # 3. Un domaine exclu : entraînement sur les 11 autres domaines générés
    if not args.skip_lodo:
        res.setdefault("lodo", {})
        for i, held in enumerate(new):
            if held in res["lodo"] or not claim(f"lodo_{held}"):
                continue
            print(f"[domaine exclu {i + 1}/{len(new)}] {held}", flush=True)
            nli = train(doms, [d for d in new if d != held], SEED + i, args.epochs)
            res["lodo"][held] = {"gold": eval_nli(nli, doms[held], "gold"), "gen": eval_nli(nli, doms[held], "gen")}
            print(f"  {held}: gold {fmt(res['lodo'][held]['gold'])} | gen {fmt(res['lodo'][held]['gen'])}", flush=True)
            del nli
            torch.cuda.empty_cache()
            save()

    # 4. Tous les domaines générés -> gold (dont support / RH, jamais vus)
    if "ft_all" not in res and claim("ft_all"):
        print("[tous les domaines]", flush=True)
        nli = train(doms, new, SEED, args.epochs)
        res["ft_all"] = {d: {"gold": eval_nli(nli, doms[d], "gold")} for d in doms}
        save()
        out = CACHE.parent / "models" / "krito-nli-fr-multi" / "hf"
        nli.model.save_pretrained(out)
        nli.tok.save_pretrained(out)

    # Tableau récapitulatif (gold)
    rows = ["zs_e5_base", "zs_mdeberta", "ft_support_only", "lodo", "ft_all", "sup_e5_base"]
    print(f"\n{'domaine':14}" + "".join(f"{r:>17}" for r in rows))
    for d in doms:
        line = f"{d:14}"
        for r in rows:
            m = res.get(r, {}).get(d, {}).get("gold")
            line += f"{m['accuracy']:>9.2f}/{m['auroc_ood']:.2f}" if m else f"{'—':>17}"
        print(line)


if __name__ == "__main__":
    main()
