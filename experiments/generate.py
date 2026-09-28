"""Génère le jeu multi-domaines avec un LLM local (Ollama).

Pour chaque domaine de data/taxonomy.json et chaque catégorie, plusieurs appels
avec des consignes de style différentes, plus des messages hors-sujet « proches »
(même contexte, aucune catégorie ne convient).

Sortie : data/generated/<domaine>.jsonl (une ligne par message ; reprise possible).

    uv run python generate.py --model gemma4:26b
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request

from lib import DATA

OUT = DATA / "generated"

STYLES = {
    "default": [
        "des messages courts et directs (une phrase), niveau de langue courant",
        "des messages longs (3 à 5 phrases) avec du contexte personnel ; le vrai sujet n'apparaît qu'au milieu ou à la fin",
        "un style SMS ou messagerie instantanée : familier, abréviations, fautes d'orthographe, peu de ponctuation",
        "des e-mails formels avec formule d'appel et formule de politesse",
        "des formulations indirectes : le sujet se devine sans reprendre les mots-clés de la description de la catégorie",
    ],
    "news": [
        "des titres d'articles courts et factuels",
        "des titres accrocheurs de site web d'information",
        "des chapeaux d'articles de deux phrases",
        "des extraits de dépêche d'agence de presse (3 à 4 phrases) avec lieux et chiffres",
        "des titres ou extraits où le thème se devine sans mot-clé évident",
    ],
    "politics": [
        "des questions de citoyens adressées à un élu local ou national",
        "des extraits de débat parlementaire ou télévisé (2 à 3 phrases)",
        "des titres d'articles sur des projets de loi ou des réformes",
        "des extraits de tribunes ou de programmes, sans nommer de parti ni de personnalité réelle",
        "des messages familiers sur les réseaux sociaux à propos de la politique menée, sans insulte",
    ],
    "ecommerce": [
        "des avis très courts (quelques mots à une phrase)",
        "des avis détaillés de 3 à 5 phrases qui racontent l'expérience",
        "des avis familiers avec fautes d'orthographe et émotions (positives ou négatives)",
        "des avis mitigés qui mentionnent aussi d'autres aspects secondaires",
        "des avis positifs enthousiastes",
    ],
}
OOD_STYLES = [
    "des questions pratiques (horaires, adresse, accès, tarifs généraux, contact)",
    "des remerciements, félicitations ou messages de politesse sans demande précise",
    "des candidatures, propositions commerciales, sollicitations ou partenariats",
    "des messages sur des sujets proches du domaine mais qui ne relèvent d'aucune catégorie listée",
]

SCHEMA = {
    "type": "object",
    "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
    "required": ["messages"],
}


def ask(model: str, prompt: str, seed: int, n_retry: int = 3) -> list[str]:
    body = {
        "model": model, "stream": False, "think": False, "format": SCHEMA,
        "options": {"temperature": 1.0, "top_p": 0.95, "seed": seed, "num_ctx": 4096},
        "messages": [{"role": "user", "content": prompt}],
    }
    for attempt in range(n_retry):
        try:
            req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            msgs = json.loads(d["message"]["content"])["messages"]
            return [m.strip() for m in msgs if isinstance(m, str) and len(m.strip()) > 3]
        except Exception as exc:  # JSON tronqué, timeout…
            print(f"  essai {attempt + 1} échoué : {exc}", flush=True)
            time.sleep(2)
    return []


def class_prompt(dom: dict, cls: str, style: str, n: int) -> str:
    others = "\n".join(f"- {d}" for c, d in dom["classes"].items() if c != cls)
    return f"""Tu génères des données d'évaluation réalistes en français pour un classifieur de textes.

Contexte : {dom["situation"]}.
Écris {n} textes différents qui relèvent clairement de la catégorie suivante :
« {dom["classes"][cls]} »

Ils ne doivent PAS relever de ces autres catégories :
{others}

Style demandé : {style}.
Varie les situations, les personnes, les détails (noms de produits, chiffres, lieux) et la longueur.
N'écris aucune numérotation ni explication. Réponds en JSON."""


def ood_prompt(dom: dict, style: str, n: int) -> str:
    cats = "\n".join(f"- {d}" for d in dom["classes"].values())
    return f"""Tu génères des données d'évaluation réalistes en français pour un classifieur de textes.

Contexte : {dom["situation"]}.
Écris {n} textes réalistes dans ce contexte, mais qui ne relèvent d'AUCUNE des catégories suivantes :
{cats}

Type de textes : {style}.
Varie la longueur et le ton. N'écris aucune numérotation ni explication. Réponds en JSON."""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma4:26b")
    p.add_argument("--per-call", type=int, default=10)
    p.add_argument("--domains", nargs="*")
    args = p.parse_args()
    taxo = json.loads((DATA / "taxonomy.json").read_text())
    OUT.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    for di, (dname, dom) in enumerate(taxo.items()):
        if args.domains and dname not in args.domains:
            continue
        path = OUT / f"{dname}.jsonl"
        done = {(r["label"], r["style"]) for r in map(json.loads, path.open())} if path.exists() else set()
        styles = STYLES.get(dname, STYLES["default"])
        jobs = [(c, s, class_prompt(dom, c, s, args.per_call)) for c in dom["classes"] for s in styles]
        jobs += [("none", s, ood_prompt(dom, s, args.per_call)) for s in OOD_STYLES]
        for ji, (label, style, prompt) in enumerate(jobs):
            if (label, style) in done:
                continue
            seed = random.Random(f"{dname}/{label}/{style}").randrange(2**31)
            msgs = ask(args.model, prompt, seed)
            with path.open("a") as f:
                for m in msgs:
                    f.write(json.dumps({"label": label, "text": m, "style": style, "model": args.model},
                                       ensure_ascii=False) + "\n")
            print(f"[{time.perf_counter() - t_start:6.0f}s] {dname} {ji + 1}/{len(jobs)} {label}: {len(msgs)}",
                  flush=True)


if __name__ == "__main__":
    main()
