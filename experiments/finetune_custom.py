"""Fine-tuner Krito sur VOS données, puis produire un modèle ONNX prêt pour le CPU.

Entrées :
- un CSV annoté avec les colonnes ``text`` et ``label`` (``none`` pour les messages hors-sujet) ;
- une taxonomie JSON ``{"etiquette": "description en langage naturel", ...}``.

Sortie : un dossier utilisable directement par ``KritoEngine.from_onnx(dossier)``
(model.opt.onnx, tokenizer.json, labels.json) et les poids PyTorch dans ``dossier/hf``.

    uv run python experiments/finetune_custom.py \\
        --data mes_messages.csv --taxonomy mes_categories.json --out modeles/krito-maboite \\
        --eval mes_messages_test.csv

Mémoire : produire le modèle demande environ 8 Go de RAM libre (pic de la quantification
int8), une seule fois. L'utiliser ensuite n'en demande qu'environ 710 Mo.

Un GPU accélère fortement l'entraînement (quelques minutes pour quelques centaines de
messages) ; sur CPU, comptez beaucoup plus longtemps. L'inférence, elle, reste sur CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import features  # noqa: E402
from export_onnx import MODELS  # noqa: E402
from run_multidomain import TEMPLATES  # noqa: E402
from run_nli_finetune import NLI  # noqa: E402

DEFAULT_BASE = MODELS / "krito-nli-fr-multi" / "hf"


def read_csv(path: str) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        return [(r["label"].strip(), r["text"]) for r in csv.DictReader(f)]


def pairs(rows, desc: dict[str, str], rng: random.Random, n_neg: int = 2):
    keys = list(desc)
    out = []
    for label, text in rows:
        negs = keys if label == "none" else [k for k in keys if k != label]
        if label != "none":
            out.append((text, rng.choice(TEMPLATES).format(desc[label]), 0))
        for k in rng.sample(negs, min(n_neg, len(negs))):
            out.append((text, rng.choice(TEMPLATES).format(desc[k]), 1))
    rng.shuffle(out)
    return out


def accuracy(nli: NLI, rows, desc: dict[str, str], template: str) -> float:
    keys = list(desc)
    rows = [(l, t) for l, t in rows if l != "none"]
    lg = nli.logits([t for _, t in rows], [template.format(desc[k]) for k in keys])
    pred = (lg[..., 0] - lg[..., 1]).argmax(1)
    return float(np.mean([keys[p] == l for p, (l, _) in zip(pred, rows)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="CSV d'entraînement (text, label)")
    p.add_argument("--taxonomy", required=True, help="JSON {etiquette: description}")
    p.add_argument("--out", required=True, help="dossier de sortie")
    p.add_argument("--eval", help="CSV d'évaluation, distinct de l'entraînement")
    p.add_argument("--base", default=str(DEFAULT_BASE) if DEFAULT_BASE.exists() else features.NLI_MODELS["mdeberta"],
                   help="modèle de départ (dossier ou identifiant Hugging Face)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--template", default="Ce texte concerne {}.", help="gabarit utilisé pour l'évaluation")
    args = p.parse_args()

    desc = json.loads(Path(args.taxonomy).read_text(encoding="utf-8"))
    train = read_csv(args.data)
    unknown = {l for l, _ in train} - set(desc) - {"none"}
    if unknown:
        sys.exit(f"Étiquettes absentes de la taxonomie : {sorted(unknown)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(train)} exemples, {len(desc)} catégories, départ : {args.base}, appareil : {device}")

    torch.manual_seed(42)
    nli = NLI(args.base, device=device, fp32=True, max_length=256)
    test = read_csv(args.eval) if args.eval else None
    if test:
        print(f"précision avant fine-tuning : {accuracy(nli, test, desc, args.template):.1%}")
    nli.train(pairs(train, desc, random.Random(42)), epochs=args.epochs)
    if test:
        print(f"précision après fine-tuning : {accuracy(nli, test, desc, args.template):.1%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nli.model.save_pretrained(out / "hf")
    nli.tok.save_pretrained(out / "hf")
    # Export puis quantification, chacun dans son processus (pics d'environ 5,4 Go et 7,3 Go).
    # Ce processus se remplace d'abord par un orchestrateur léger (execv) : toute la mémoire
    # de l'entraînement (PyTorch, CUDA) est rendue avant ces étapes gourmandes.
    exporter = str(Path(__file__).parent / "export_onnx.py")
    orchestrator = (
        "import subprocess, sys, pathlib\n"
        f"for step in ('--export-fp32', '--finalize'):\n"
        f"    subprocess.run([sys.executable, {exporter!r}, step, {str(out)!r}], check=True)\n"
        f"pathlib.Path({str(out / 'model.onnx')!r}).unlink(missing_ok=True)\n"
        f"print('modèle prêt : KritoEngine.from_onnx({str(out)!r})')\n"
    )
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, "-c", orchestrator])


if __name__ == "__main__":
    main()
