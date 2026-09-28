"""Fine-tuning d'un cross-encoder NLI multilingue sur les tickets de support.

Pour chaque pli, on construit des paires (texte, hypothèse) à partir du pli d'entraînement :
- texte du domaine + hypothèse de sa classe  -> entailment
- texte du domaine + hypothèse d'une autre classe -> contradiction
- texte hors-sujet + n'importe quelle hypothèse -> contradiction (apprend à dire « aucune »)
Les hypothèses sont formulées avec plusieurs gabarits pour limiter le sur-apprentissage
d'une formulation précise.

Sorties :
- métriques par pli (support) avec le scoring Krito (relatif) et absolu ;
- logits « hors pli » (chaque texte prédit par un modèle qui ne l'a pas vu), réutilisés par le juge ;
- transfert : un modèle entraîné sur tout le support, évalué sur le domaine RH jamais vu.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import features
from lib import CACHE, HYPOTHESES, SEED, evaluate, folds, load, mean_metrics, save
from run_zero_shot import nli_scores

TEMPLATES = [
    "Ce message concerne {}.",
    "Le sujet de ce message est {}.",
    "Il s'agit de {}.",
    "Cette demande porte sur {}.",
]
PREFIX = "Ce message concerne "


def descriptions(domain: str) -> dict[str, str]:
    return {c: h[len(PREFIX) : -1] for c, h in HYPOTHESES[domain].items()}


def build_pairs(texts, labels, classes, domain, rng, n_neg=2):
    desc = descriptions(domain)
    pairs = []  # (texte, hypothèse, cible) ; cible 0=entail, 1=contradiction
    for t, l in zip(texts, labels):
        if l == "none":
            for c in rng.sample(classes, n_neg):
                pairs.append((t, rng.choice(TEMPLATES).format(desc[c]), 1))
            continue
        pairs.append((t, rng.choice(TEMPLATES).format(desc[l]), 0))
        for c in rng.sample([c for c in classes if c != l], n_neg):
            pairs.append((t, rng.choice(TEMPLATES).format(desc[c]), 1))
    rng.shuffle(pairs)
    return pairs


class NLI:
    def __init__(self, name: str, device: str = "cuda", fp32: bool = False, max_length: int = 128):
        self.max_length = max_length
        self.amp_dtype = torch.float32 if fp32 else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32).to(device)
        self.order = features.label_order(self.model.config)  # [E, C, N]
        self.device = device

    def train(self, pairs, epochs=3, lr=2e-5, bs=16):
        for p in self.model.base_model.embeddings.parameters():
            p.requires_grad = False
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        total = epochs * ((len(pairs) + bs - 1) // bs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * total)) * max(0.0, 1 - s / total))
        target_idx = torch.tensor(self.order[:2], device=self.device)  # cible 0 -> idx E, 1 -> idx C
        self.model.train()
        for _ in range(epochs):
            for i in range(0, len(pairs), bs):
                b = pairs[i : i + bs]
                enc = self.tok([x[0] for x in b], [x[1] for x in b], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                y = target_idx[torch.tensor([x[2] for x in b], device=self.device)]
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                    logits = self.model(**enc).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
        self.model.eval()

    @torch.no_grad()
    def logits(self, texts, hyps, bs=32) -> np.ndarray:
        pairs = [(t, h) for t in texts for h in hyps]
        out = []
        for i in range(0, len(pairs), bs):
            b = pairs[i : i + bs]
            enc = self.tok([x[0] for x in b], [x[1] for x in b], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32):
                out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)[:, self.order].reshape(len(texts), len(hyps), 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["minilm_multi", "mdeberta"], default="minilm_multi")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--fp32", action="store_true", help="Pleine précision (DeBERTa-v3 diverge en bf16)")
    args = p.parse_args()
    name = features.NLI_MODELS[args.model]

    ds, rh = load("support"), load("rh")
    y = ds.y
    texts, labels = np.array(ds.texts), np.array(ds.labels)
    oof = np.zeros((len(texts), len(ds.classes), 3), dtype=np.float32)
    per = {"krito": [], "abs": []}

    for fold_i, (tr, te) in enumerate(folds(ds)):
        t0 = time.perf_counter()
        torch.manual_seed(SEED + fold_i)
        rng = random.Random(SEED + fold_i)
        nli = NLI(name, fp32=args.fp32)
        nli.train(build_pairs(texts[tr].tolist(), labels[tr].tolist(), ds.classes, "support", rng), epochs=args.epochs)
        oof[te] = nli.logits(texts[te].tolist(), ds.hypotheses)
        for variant, (pred, conf) in nli_scores(oof[te]).items():
            per[variant].append(evaluate(y[te], pred, conf))
        print(f"pli {fold_i}: acc={per['krito'][-1]['accuracy']:.2f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        del nli
        torch.cuda.empty_cache()

    # Modèle entraîné sur tout le support -> transfert RH
    torch.manual_seed(SEED)
    nli = NLI(name, fp32=args.fp32)
    nli.train(build_pairs(ds.texts, ds.labels, ds.classes, "support", random.Random(SEED)), epochs=args.epochs)
    rh_logits = nli.logits(rh.texts, rh.hypotheses)
    np.savez(CACHE / f"nliFT_{args.model}_support_oof.npz", logits=oof)
    np.savez(CACHE / f"nliFT_{args.model}_rh.npz", logits=rh_logits)

    results = {"support_cv": {}, "rh_transfer": {}}
    for variant in per:
        results["support_cv"][f"{variant}_ft_{args.model}"] = mean_metrics(per[variant])
    for variant, (pred, conf) in nli_scores(rh_logits).items():
        results["rh_transfer"][f"{variant}_ft_{args.model}"] = evaluate(rh.y, pred, conf)
    save(f"nli_finetune_{args.model}", results)
    for k, m in results["support_cv"].items():
        t = results["rh_transfer"][k]
        print(f"{k:24} acc={m['accuracy']:.2f}±{m['accuracy_std']:.2f} auc_acc={m['auroc_accept']:.2f}"
              f" auc_ood={m['auroc_ood']:.2f} cov@90={m['coverage_at_90']:.2f} ece={m['ece']:.2f}"
              f" | RH acc={t['accuracy']:.2f} auc_acc={t['auroc_accept']:.2f} auc_ood={t['auroc_ood']:.2f}")


if __name__ == "__main__":
    main()
