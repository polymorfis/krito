"""Modèle TextCNN entraîné de zéro : évaluation multi-domaines et export du modèle livré avec Krito.

Même protocole que run_multidomain.py (textes générés filtrés pour l'entraînement, jeux
gold d'un autre auteur pour le test), sans aucun poids pré-entraîné :

- lodo   : entraîné sur 11 domaines, testé sur le 12e (zero-shot réel, tokenizer compris) ;
- ft_all : entraîné sur les 12 domaines, testé sur les gold (dont support / RH, jamais vus).

    uv run python run_textcnn.py                # tout, puis export dans src/krito/textcnn/krito-textcnn-fr
    uv run python run_textcnn.py --skip-lodo    # entraînement final et export uniquement
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from krito.textcnn import DEFAULT_TEXTCNN_MODEL, TextCNNBackend
from krito.textcnn.train import Domain, evaluate, export_onnx, train
from lib import DATA, RESULTS, SEED, softmax
from lib import evaluate as metrics

CANONICAL = "Ce texte concerne {}."
LEGACY = {  # descriptions des jeux de l'étude 1 (lib.HYPOTHESES sans « Ce message concerne » ni le point)
    d: {k: h[len("Ce message concerne "):-1] for k, h in hyps.items()}
    for d, hyps in __import__("lib").HYPOTHESES.items()
}


def load_domains() -> tuple[list[Domain], dict[str, Domain]]:
    """Domaines d'entraînement (textes générés, accord des deux LLM) et jeux gold par domaine."""
    taxo = json.loads((DATA / "taxonomy.json").read_text())
    gold: dict[str, list] = {}
    with open(DATA / "gold_fr.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            gold.setdefault(r["domain"], []).append((r["label"], r["text"]))
    train_doms, gold_doms = [], {}
    for d, t in taxo.items():
        with open(DATA / "multi" / f"{d}.tsv", encoding="utf-8") as f:
            gen = [(r["label"], r["text"]) for r in csv.DictReader(f, delimiter="\t") if r["agree"] == "1"]
        train_doms.append(Domain(d, t["classes"], gen))
        gold_doms[d] = Domain(d, t["classes"], gold[d])
    for d in ("support", "rh"):  # jeux de l'étude 1 : test uniquement
        with open(DATA / f"{d}_fr.tsv", encoding="utf-8") as f:
            rows = [(r["label"], r["text"]) for r in csv.DictReader(f, delimiter="\t")]
        gold_doms[d] = Domain(d, LEGACY[d], rows)
    return train_doms, gold_doms


def score(backend, dom: Domain) -> dict:
    """Métriques de lib.evaluate avec la confiance absolue P(entailment) de l'option retenue."""
    r = evaluate(backend, dom, CANONICAL)
    logits = r["logits"]
    pred = (logits[..., 0] - logits[..., 1]).argmax(1)
    conf = softmax(logits, axis=2)[np.arange(len(pred)), pred, 0]
    return metrics(r["y"], pred, conf)


def fmt(m: dict) -> str:
    return f"acc={m['accuracy']:.2f} auc_ood={m['auroc_ood']:.2f} cov@90={m['coverage_at_90']:.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-lodo", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", type=Path, default=DEFAULT_TEXTCNN_MODEL)
    args = p.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train_doms, gold = load_domains()
    out_path = RESULTS / "textcnn.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}

    if not args.skip_lodo:
        res["lodo"] = {}
        for i, held in enumerate(train_doms):
            print(f"[domaine exclu {i + 1}/{len(train_doms)}] {held.name}", flush=True)
            backend = train([d for d in train_doms if d is not held], epochs=args.epochs, seed=SEED + i, log=lambda *_: None)
            res["lodo"][held.name] = {"gold": score(backend, gold[held.name]), "gen": score(backend, held)}
            print(f"  gold {fmt(res['lodo'][held.name]['gold'])} | gen {fmt(res['lodo'][held.name]['gen'])}", flush=True)
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("[tous les domaines]", flush=True)
    t0 = time.perf_counter()
    backend = train(train_doms, epochs=args.epochs, seed=SEED)
    res["train_seconds"] = round(time.perf_counter() - t0, 1)
    res["ft_all"] = {d: {"gold": score(backend, g)} for d, g in gold.items()}
    for d, m in res["ft_all"].items():
        print(f"  {d:14} {fmt(m['gold'])}")

    if args.out.exists():
        shutil.rmtree(args.out)
    export_onnx(backend, args.out)
    onnx = TextCNNBackend(args.out)
    res["ft_all_onnx"] = {d: {"gold": score(onnx, g)} for d, g in gold.items()}
    res["params"] = backend.model.num_parameters()
    res["size_mb"] = round(sum(f.stat().st_size for f in args.out.iterdir()) / 1e6, 2)
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    new = [d.name for d in train_doms]
    for key in ("lodo", "ft_all", "ft_all_onnx"):
        if key in res:
            acc = np.mean([res[key][d]["gold"]["accuracy"] for d in new])
            auc = np.mean([res[key][d]["gold"]["auroc_ood"] for d in new])
            print(f"{key:12} moyenne 12 domaines : acc {acc:.1%}  auroc_ood {auc:.1%}")


if __name__ == "__main__":
    main()
