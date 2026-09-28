"""Tableaux Markdown de l'étude multi-domaines, générés depuis results/*.json.

    uv run python summarize_multidomain.py > results/multidomain_tables.md
"""

from __future__ import annotations

import json

import numpy as np

from lib import RESULTS

NAMES = {
    "legal": "Juridique", "programming": "Programmation", "finance": "Compta / finance",
    "marketing": "Marketing", "automotive": "Auto / moto", "communication": "Communication",
    "news": "Actualités", "politics": "Politiques publiques", "health": "Santé (admin.)",
    "realestate": "Immobilier", "itsec": "Support IT / sécu", "ecommerce": "Avis e-commerce",
    "support": "Support client (étude 1)", "rh": "RH (étude 1)",
}
COLS = [
    ("zs_mdeberta", "mDeBERTa zero-shot"),
    ("zs_e5_base", "Embeddings e5 zero-shot"),
    ("ft_support_only", "Fine-tuné support seul"),
    ("lodo", "**Fine-tuné 11 domaines (domaine exclu)**"),
    ("ft_all", "Fine-tuné 12 domaines"),
    ("sup_e5_base", "Supervisé e5 + LR"),
]


def pct(x):
    return "—" if x is None else f"{100 * x:.0f} %"


def get(res, col, d, metric, split="gold"):
    m = res.get(col, {}).get(d, {}).get(split)
    return None if m is None else m[metric]


def table(res, llm, metric, title, split="gold", with_llm=True):
    doms = [d for d in NAMES if d in res.get("zs_mdeberta", {})]
    cols = [(c, n) for c, n in COLS if split == "gold" or c in ("zs_mdeberta", "lodo")]
    head = "| Domaine | " + " | ".join(n for _, n in cols) + (" | LLM 27B |" if with_llm else " |")
    lines = [f"### {title}", "", head, "|---|" + ":-:|" * (len(cols) + with_llm)]
    avgs = {c: [] for c, _ in cols}
    for d in doms:
        row = [NAMES[d]]
        for c, _ in cols:
            v = get(res, c, d, metric, split)
            row.append(pct(v))
            if v is not None and d not in ("support", "rh"):
                avgs[c].append(v)
        if with_llm:
            row.append(pct(llm.get(d, {}).get("accuracy")) if metric == "accuracy" else "—")
        lines.append("| " + " | ".join(row) + " |")
    avg_row = ["**Moyenne des 12 nouveaux domaines**"] + [
        f"**{pct(np.mean(v))}**" if len(v) == 12 else f"({pct(np.mean(v)) if v else '—'}, {len(v)}/12)"
        for v in avgs.values()
    ]
    if with_llm:
        vals = [llm[d]["accuracy"] for d in llm] if metric == "accuracy" else []
        avg_row.append(f"**{pct(np.mean(vals))}**" if vals else "—")
    lines.append("| " + " | ".join(avg_row) + " |")
    return "\n".join(lines)


def main():
    res = json.loads((RESULTS / "multidomain.json").read_text())
    llm = json.loads((RESULTS / "llm_gold_qwen3.8_27b.json").read_text())
    print(table(res, llm, "accuracy", "Précision sur les jeux gold (messages du domaine)"))
    print()
    print(table(res, llm, "auroc_ood", "AUROC hors-sujet sur les jeux gold", with_llm=False))
    print()
    print(table(res, llm, "coverage_at_90", "Traités automatiquement à 90 % de précision (gold)", with_llm=False))
    print()
    print(table(res, llm, "accuracy", "Précision sur les textes générés du domaine exclu (~300 par domaine)",
                split="gen", with_llm=False))
    q = json.loads((RESULTS / "dataset_quality.json").read_text())
    print("\n### Qualité du jeu généré\n")
    print("| Domaine | Générés | Doublons retirés | Trop proches du gold | Accord Gemma / Qwen | Gardés |")
    print("|---|:-:|:-:|:-:|:-:|:-:|")
    for d, s in q.items():
        print(f"| {NAMES[d]} | {s['raw']} | {s['exact_dups'] + s['near_dups']} | {s['near_gold']} "
              f"| {100 * s['agree_rate']:.1f} % | {s['kept_clean']} |")
    tot = {k: sum(s[k] for s in q.values()) for k in ("raw", "kept_clean")}
    print(f"| **Total** | **{tot['raw']}** | | | | **{tot['kept_clean']}** |")


if __name__ == "__main__":
    main()
