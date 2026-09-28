"""Contrôle qualité du jeu généré et référence LLM sur les jeux gold.

1. Doublons exacts (texte normalisé) et quasi-doublons (cosinus e5-small > seuil),
   y compris avec les jeux gold (pour éviter toute fuite vers le test).
2. Contre-annotation par un second LLM (différent du générateur), par lots.
   Chaque texte généré reçoit ``verifier_label`` ; ``agree`` = accord avec l'étiquette
   demandée au générateur.
3. Le même LLM annote les jeux gold : c'est la référence LLM de l'étude.

Sorties : data/multi/<domaine>.tsv et results/llm_gold_<modèle>.json

    uv run python verify.py --model qwen3.8:27b
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
import urllib.request

import numpy as np

from lib import CACHE, DATA, RESULTS

MULTI = DATA / "multi"
NEAR_DUP = 0.95
BATCH = 12
_ST = None


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def load_gold() -> dict[str, list[dict]]:
    gold: dict[str, list[dict]] = {}
    with open(DATA / "gold_fr.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            gold.setdefault(r["domain"], []).append(r)
    return gold


def embed(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    global _ST
    if _ST is None:
        _ST = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    return _ST.encode(["query: " + t for t in texts], normalize_embeddings=True, batch_size=64)


def dedup(rows: list[dict], gold_texts: list[str]) -> tuple[list[dict], dict]:
    stats = {"raw": len(rows)}
    seen, uniq = set(), []
    for r in rows:
        k = norm(r["text"])
        if k and k not in seen:
            seen.add(k)
            uniq.append(r)
    stats["exact_dups"] = len(rows) - len(uniq)
    E = embed([r["text"] for r in uniq])
    G = embed(gold_texts) if gold_texts else np.zeros((0, E.shape[1]))
    keep, dropped_near, dropped_gold = [], 0, 0
    kept_idx: list[int] = []
    for i in range(len(uniq)):
        if len(G) and (G @ E[i]).max() > NEAR_DUP:
            dropped_gold += 1
            continue
        if kept_idx and (E[kept_idx] @ E[i]).max() > NEAR_DUP:
            dropped_near += 1
            continue
        kept_idx.append(i)
        keep.append(uniq[i])
    stats |= {"near_dups": dropped_near, "near_gold": dropped_gold, "kept": len(keep)}
    return keep, stats


def label_batch(model: str, dom: dict, texts: list[str]) -> list[str]:
    keys = list(dom["classes"]) + ["none"]
    cats = "\n".join(f"- {k} : {d}" for k, d in dom["classes"].items())
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = f"""Classe chacun des {len(texts)} textes ci-dessous dans UNE catégorie.

Contexte : {dom["situation"]}.
Catégories :
{cats}
- none : aucune des catégories ci-dessus ne convient

Textes :
{listing}

Réponds en JSON avec la liste « labels » contenant exactement {len(texts)} étiquettes, dans l'ordre."""
    body = {
        "model": model, "stream": False, "think": False,
        "options": {"temperature": 0, "num_ctx": 4096},
        "format": {"type": "object", "properties": {"labels": {"type": "array", "items": {"enum": keys}}},
                   "required": ["labels"]},
        "messages": [{"role": "user", "content": prompt}],
    }
    for attempt in range(5):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                labels = json.loads(json.load(r)["message"]["content"])["labels"]
            break
        except Exception as exc:  # serveur redémarré, JSON tronqué…
            print(f"  lot en échec ({exc}), nouvelle tentative", flush=True)
            time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError("Ollama ne répond plus")
    if len(labels) != len(texts):
        if len(texts) == 1:
            return ["?"]
        mid = len(texts) // 2  # longueur incohérente : on redécoupe
        return label_batch(model, dom, texts[:mid]) + label_batch(model, dom, texts[mid:])
    return labels


def label_all(model: str, dom: dict, texts: list[str], cache_key: str) -> list[str]:
    path = CACHE / f"verify_{model.replace(':', '_')}_{cache_key}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [t for t in texts if t not in cache]
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        cache.update(zip(chunk, label_batch(model, dom, chunk)))
        path.write_text(json.dumps(cache, ensure_ascii=False))
    return [cache[t] for t in texts]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.8:27b")
    args = p.parse_args()
    taxo = json.loads((DATA / "taxonomy.json").read_text())
    gold = load_gold()
    MULTI.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    report, llm_gold = {}, {}
    t0 = time.perf_counter()
    for dname, dom in taxo.items():
        rows = [json.loads(l) for l in (DATA / "generated" / f"{dname}.jsonl").open()]
        rows, stats = dedup(rows, [g["text"] for g in gold.get(dname, [])])
        vlabels = label_all(args.model, dom, [r["text"] for r in rows], dname)
        for r, v in zip(rows, vlabels):
            r["verifier_label"], r["agree"] = v, int(v == r["label"])
        with open(MULTI / f"{dname}.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["label", "text", "style", "verifier_label", "agree"], delimiter="\t",
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        stats["agree_rate"] = float(np.mean([r["agree"] for r in rows]))
        stats["kept_clean"] = int(sum(r["agree"] for r in rows))
        report[dname] = stats

        g = gold[dname]
        gl = label_all(args.model, dom, [x["text"] for x in g], f"gold_{dname}")
        y, pred = [x["label"] for x in g], gl
        ind = [i for i, l in enumerate(y) if l != "none"]
        ood = [i for i, l in enumerate(y) if l == "none"]
        llm_gold[dname] = {
            "accuracy": float(np.mean([pred[i] == y[i] for i in ind])),
            "ood_said_none": float(np.mean([pred[i] == "none" for i in ood])),
            "in_domain_said_none": float(np.mean([pred[i] == "none" for i in ind])),
        }
        print(f"[{time.perf_counter() - t0:5.0f}s] {dname:14} {stats} | LLM gold {llm_gold[dname]}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "dataset_quality.json").write_text(json.dumps(report, indent=2))
    (RESULTS / f"llm_gold_{args.model.replace(':', '_')}.json").write_text(json.dumps(llm_gold, indent=2))


if __name__ == "__main__":
    main()
