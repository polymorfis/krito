"""Publie un modèle Krito (dossier ONNX) sur le Hugging Face Hub, avec sa fiche de modèle.

Seuls les fichiers nécessaires à ``KritoEngine.from_onnx`` sont envoyés : la meilleure
variante du modèle (``model.opt.onnx`` > ``model.int8.onnx`` > ``model.onnx``),
``tokenizer.json`` et ``labels.json``, plus une fiche ``README.md`` générée.

    # 1. Préparer et relire la fiche, sans rien envoyer
    uv run python experiments/publish_hub.py experiments/models/krito-nli-fr-multi \\
        --repo polymorfis/krito-nli-fr-multi --data-card donnees.md --metrics resultats.json --dry-run
    # 2. Publier (jeton : `hf auth login` ou variable HF_TOKEN)
    uv run python experiments/publish_hub.py ... sans --dry-run

``--data-card`` : description Markdown des données d'entraînement (origine, réelles ou
synthétiques, volume, anonymisation, licence). Obligatoire : un modèle publié sans
description de ses données n'est pas évaluable.
``--metrics`` : JSON ``{"nom de la mesure": valeur, ...}`` affiché dans la fiche.

Avant de publier, vérifiez :
- que les données d'entraînement ne contiennent aucune donnée personnelle
  (``experiments/check_annotations.py``) ;
- les conditions d'utilisation des LLM ayant généré ou filtré des données : certaines encadrent
  l'usage de leurs sorties pour entraîner d'autres modèles.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from krito.krito import ONNX_MODEL_FILES

BASE_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def model_card(repo: str, model_file: str, data_card: str, metrics: dict, template: str, base_model: str) -> str:
    rows = "\n".join(f"| {k} | {v:.1%} |" if isinstance(v, float) and v <= 1 else f"| {k} | {v} |"
                     for k, v in metrics.items())
    metrics_md = f"| Mesure | Valeur |\n|---|---|\n{rows}" if metrics else "_Aucune mesure fournie._"
    return f"""---
license: mit
language:
- fr
library_name: onnx
pipeline_tag: zero-shot-classification
base_model: {base_model}
tags:
- krito
- nli
- onnx
- cpu
---

# {repo.split("/")[-1]}

Modèle d'inférence en langage naturel (NLI) pour [Krito](https://github.com/polymorfis/krito) :
des décisions typées avec une confiance chiffrée, sur CPU, sans GPU ni API externe.
Export ONNX (`{model_file}`), utilisable sans PyTorch.

## Utilisation

```python
from krito import KritoEngine

engine = KritoEngine.from_onnx("{repo}", threads=2, hypothesis_template="{template}")
result = engine.classify(
    "Le colis indiqué comme livré n'est jamais arrivé.",
    {{"facturation": "un problème de facturation ou de paiement",
      "livraison": "la livraison ou le suivi d'un colis"}},
    min_entailment=0.5,
)
```

Calibrez les garde-fous sur vos propres données annotées : `engine.calibrate(exemples, options)`.

## Données d'entraînement

{data_card.strip()}

## Résultats

{metrics_md}

## Limites

- Entraîné et évalué en français ; les autres langues n'ont pas été mesurées.
- Au-delà de 512 tokens, la fin du texte est ignorée.
- L'échelle de `entailment_prob` dépend du modèle et du gabarit : les seuils se calibrent
  sur des données annotées du domaine visé.
- Une décision acceptée peut être fausse : gardez une revue humaine sur une partie du flux.

## Origine et licence

Fine-tuné à partir de [`{base_model}`](https://huggingface.co/{base_model}). Licence MIT.
Publié par [POLYMORFIS](https://www.polymorfis.com/).
"""


def stage(model_dir: Path, card: str) -> Path:
    model_file = next((n for n in ONNX_MODEL_FILES if (model_dir / n).exists()), None)
    missing = [n for n in ("tokenizer.json", "labels.json") if not (model_dir / n).exists()]
    if model_file is None or missing:
        raise SystemExit(f"Dossier incomplet : il faut un modèle ({', '.join(ONNX_MODEL_FILES)}), "
                         f"tokenizer.json et labels.json. Manquant : {missing or 'modèle ONNX'}")
    out = Path(tempfile.mkdtemp(prefix="krito-hub-"))
    for name in (model_file, "tokenizer.json", "labels.json"):
        # Lien physique si possible : pas de copie de 400 Mo
        try:
            (out / name).hardlink_to(model_dir / name)
        except OSError:
            shutil.copy2(model_dir / name, out / name)
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", help="dossier produit par export_onnx.py ou finetune_custom.py")
    p.add_argument("--repo", required=True, help="dépôt du Hub, ex. polymorfis/krito-nli-fr-multi")
    p.add_argument("--data-card", required=True, help="Markdown décrivant les données d'entraînement")
    p.add_argument("--metrics", help="JSON {mesure: valeur} à afficher dans la fiche")
    p.add_argument("--template", default="Ce texte concerne {}.", help="gabarit d'hypothèse recommandé")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--private", action="store_true", help="crée un dépôt privé")
    p.add_argument("--dry-run", action="store_true", help="prépare les fichiers sans rien envoyer")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    model_file = next((n for n in ONNX_MODEL_FILES if (model_dir / n).exists()), "model.onnx")
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8")) if args.metrics else {}
    card = model_card(args.repo, model_file, Path(args.data_card).read_text(encoding="utf-8"),
                      metrics, args.template, args.base_model)
    staged = stage(model_dir, card)
    size = sum(f.stat().st_size for f in staged.iterdir()) / 1e6
    print(f"Fichiers prêts dans {staged} ({size:.0f} Mo) : {sorted(f.name for f in staged.iterdir())}")
    if args.dry_run:
        print(f"--dry-run : rien n'a été envoyé. Relisez {staged / 'README.md'}.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(staged), repo_id=args.repo,
                      commit_message="Publication du modèle Krito")
    shutil.rmtree(staged)
    print(f"Publié : https://huggingface.co/{args.repo}\n"
          f"Utilisation : KritoEngine.from_onnx({args.repo!r})")


if __name__ == "__main__":
    main()
