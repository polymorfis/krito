"""Référence haute : un LLM local (via Ollama) en classification zero-shot à sortie contrainte.

Le LLM peut répondre ``none`` (aucune catégorie) : c'est alors un rejet (confiance 0).
Confiance = probabilité (non contrainte) du premier token de l'étiquette choisie.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request

import numpy as np

from lib import CACHE, folds, load, mean_metrics, save, evaluate
from run_nli_finetune import descriptions

PROMPT = """Tu es un moteur de classification. Classe le message suivant dans UNE catégorie.

Catégories :
{cats}
- none : aucune des catégories ci-dessus ne convient

Message : « {text} »

Réponds uniquement en JSON."""


def classify(model: str, text: str, classes: list[str], domain: str) -> tuple[str, float, float]:
    desc = descriptions(domain)
    cats = "\n".join(f"- {c} : {desc[c]}" for c in classes)
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {"temperature": 0},
        "format": {
            "type": "object",
            "properties": {"label": {"enum": classes + ["none"]}},
            "required": ["label"],
        },
        "messages": [{"role": "user", "content": PROMPT.format(cats=cats, text=text)}],
    }
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    latency = time.perf_counter() - t0
    label = json.loads(d["message"]["content"])["label"]
    # Premier token dont le texte est un préfixe non vide de l'étiquette
    conf = 1.0
    for tok in d.get("logprobs") or []:
        s = tok["token"].strip().strip('"')
        if s and label.startswith(s):
            conf = math.exp(tok["logprob"])
            break
    return label, conf, latency


def run(model: str, dataset: str) -> dict:
    path = CACHE / f"llm_{model.replace(':', '_').replace('/', '_')}_{dataset}.json"
    ds = load(dataset)
    done = json.loads(path.read_text()) if path.exists() else {}
    for i, t in enumerate(ds.texts):
        if str(i) in done:
            continue
        done[str(i)] = classify(model, t, ds.classes, dataset)
        path.write_text(json.dumps(done, ensure_ascii=False))
        if i % 25 == 0:
            print(f"{dataset} {i}/{len(ds.texts)}", flush=True)
    idx = {c: k for k, c in enumerate(ds.classes)}
    pred = np.array([idx.get(done[str(i)][0], -1) for i in range(len(ds.texts))])
    conf = np.array([done[str(i)][1] if pred[i] >= 0 else 0.0 for i in range(len(ds.texts))])
    lat = [done[str(i)][2] for i in range(len(ds.texts))]
    return {"pred": pred, "conf": conf, "latency_p50_s": float(np.median(lat))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.8:27b")
    args = p.parse_args()
    sup, rh = load("support"), load("rh")
    r = run(args.model, "support")
    per = [evaluate(sup.y[te], r["pred"][te], r["conf"][te]) for _, te in folds(sup)]
    r_rh = run(args.model, "rh")
    name = f"llm_{args.model}"
    none_rate_ood = float((r["pred"][sup.y < 0] < 0).mean())
    out = {
        "support_cv": {name: mean_metrics(per) | {"latency_p50_s": r["latency_p50_s"], "ood_said_none": none_rate_ood}},
        "rh_transfer": {name: evaluate(rh.y, r_rh["pred"], r_rh["conf"])},
    }
    save(f"llm_{args.model.replace(':', '_')}", out)
    m, t = out["support_cv"][name], out["rh_transfer"][name]
    print(f"{name} acc={m['accuracy']:.2f} auc_acc={m['auroc_accept']:.2f} auc_ood={m['auroc_ood']:.2f}"
          f" cov@90={m['coverage_at_90']:.2f} ood->none={none_rate_ood:.2f} p50={r['latency_p50_s']:.1f}s"
          f" | RH acc={t['accuracy']:.2f} auc_acc={t['auroc_accept']:.2f}")


if __name__ == "__main__":
    main()
