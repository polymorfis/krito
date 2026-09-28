# Entraîner et exporter un modèle

Krito fonctionne avec n'importe quel modèle NLI à 3 classes, mais un modèle **fine-tuné sur votre domaine** est le levier de qualité le plus puissant. Ce document explique comment spécialiser un modèle sur vos données, puis l'exporter pour une exécution sur CPU.

## 1. Pourquoi fine-tuner ?

Chiffres issus des études du dépôt ([RESULTS.md](../experiments/RESULTS.md), [RESULTS_MULTIDOMAIN.md](../experiments/RESULTS_MULTIDOMAIN.md)) :

| Situation | Précision |
|---|:-:|
| Modèle multilingue de base, sans fine-tuning (moyenne sur 12 domaines) | 58 % |
| Fine-tuné sur 11 autres domaines, testé sur un domaine **jamais vu** | 88 % |
| Fine-tuné en incluant le domaine | 95 % |
| `krito-nli-fr-multi`, puis fine-tuné sur **202 tickets de support annotés** (test sur 72 autres tickets) | 84,7 % → **91,7 %** |

Quelques centaines d'exemples annotés de *votre* domaine suffisent à gagner plusieurs points. Le modèle reste utilisable en zero-shot sur d'autres catégories : fine-tuner sur un domaine a amélioré les résultats sur les domaines non vus au lieu de les dégrader.

## 2. Fine-tuner sur vos données

### Ce qu'il vous faut

1. **Un CSV annoté** avec les colonnes `text` et `label`. Utilisez `none` pour les messages qui ne relèvent d'aucune catégorie : c'est ce qui apprend au modèle à les rejeter. Visez 30 à 50 exemples par catégorie, plus 10 à 15 % de hors-sujet.
2. **Une taxonomie JSON** qui décrit chaque catégorie en langage naturel :

```json
{
  "facturation": "un problème de facturation, de paiement ou de remboursement",
  "livraison": "la livraison, le suivi ou le retour d'un colis",
  "compte": "la connexion, le mot de passe ou l'accès au compte"
}
```

3. **Un second CSV** pour l'évaluation, distinct du premier.
4. **Du matériel** :
   - un GPU grand public accélère fortement l'entraînement (environ 1 min pour 200 exemples sur une RTX 3060) ; sur CPU seul, c'est possible mais beaucoup plus long ;
   - **environ 8 Go de RAM libre pour l'export** (voir § 4). C'est une étape unique : l'utilisation du modèle ne demande ensuite qu'environ 710 Mo.

### Lancer

```bash
uv sync --all-groups
uv run python experiments/finetune_custom.py \
    --data mes_messages.csv --taxonomy mes_categories.json \
    --eval mes_messages_test.csv --out modeles/krito-maboite
```

Le script :
1. part de `krito-nli-fr-multi` s'il est présent (sinon du modèle multilingue de base) ;
2. affiche la précision avant et après fine-tuning sur votre jeu d'évaluation ;
3. exporte un dossier directement utilisable :

```python
engine = KritoEngine.from_onnx("modeles/krito-maboite", hypothesis_template="Ce texte concerne {}.")
```

Recalibrez ensuite les garde-fous avec [06_calibration_seuil.py](../examples/06_calibration_seuil.py) : un nouveau modèle a une nouvelle échelle de scores.

## 3. Comment l'entraînement fonctionne

Chaque exemple annoté devient plusieurs **paires** (texte, hypothèse) :

| Exemple | Paires produites |
|---|---|
| Texte de la catégorie *A* | (texte, hypothèse *A*) → *entailment* ; (texte, hypothèse *B*) → *contradiction* ; (texte, hypothèse *C*) → *contradiction* |
| Texte hors-sujet (`none`) | (texte, deux hypothèses tirées au hasard) → *contradiction* |

Les hypothèses sont formulées avec 5 gabarits de phrase différents (« Ce texte concerne… », « Il s'agit de… »…), pour que le modèle ne dépende pas d'une formulation exacte.

Réglages utilisés dans les études :
- 2 à 3 époques ;
- taux d'apprentissage 2e-5 ;
- lots de 16 ;
- 256 tokens maximum ;
- table d'embeddings **gelée**, ce qui préserve le multilinguisme et réduit la mémoire nécessaire ;
- entraînement en **fp32** : DeBERTa-v3 diverge en bf16 et produit des NaN.

## 4. Export pour le CPU

L'export produit trois variantes ; `KritoEngine.from_onnx` choisit automatiquement la meilleure disponible.

| Fichier | Taille | Rôle |
|---|:-:|---|
| `model.onnx` | 1,1 Go | Export fp32, identique au modèle PyTorch (supprimé après quantification). |
| `model.int8.onnx` | 425 Mo | Quantification int8 **sélective**. |
| `model.opt.onnx` | 404 Mo | int8 + optimisation hors ligne : même précision et même vitesse, pic de RAM au chargement 710 Mo au lieu de 869 Mo. |

### Pièges rencontrés et contournés

- **La quantification int8 par défaut détruit DeBERTa** (précision d'environ 30 %). Nos mesures couche par couche montrent que la couche `output.dense` du bloc feed-forward et les MatMul d'attention ne supportent pas l'int8. `export_onnx.quantize()` les exclut et quantifie le reste, embeddings compris. Résultat : même précision moyenne que le fp32 sur 12 domaines (95,9 % contre 94,8 %, dans le bruit de mesure).
- **L'optimisation à chaque chargement coûte de la RAM.** `export_onnx.optimize_offline()` l'applique une fois pour toutes (niveau *basic*, indépendant du processeur).
- **Mémoire de l'export.** L'export ONNX culmine à environ 5,4 Go de RAM et la quantification à environ 7,3 Go. `finetune_custom.py` lance ces étapes dans des processus séparés, après avoir rendu toute la mémoire de l'entraînement. Sur une machine de 16 Go, fermez les applications gourmandes pendant cette étape, ou lancez-la dans un cadre limité :

```bash
systemd-run --user --scope -p MemoryMax=9G uv run python experiments/finetune_custom.py ...
```

- **Ne pas écrire les modèles dans un `/tmp` monté en RAM** (tmpfs, fréquent sur les distributions récentes) : chaque Go écrit y consomme un Go de mémoire vive.

## 5. Reproduire les modèles publiés

```bash
uv sync --all-groups
cd experiments

# Étude 1 : modèle spécialisé support client (≈ 2 min sur GPU)
uv run python export_onnx.py --model mdeberta

# Étude 2 : modèle 12 domaines
uv run python generate.py --model gemma4:26b     # données, via Ollama (≈ 45 min)
uv run python verify.py --model qwen3.8:27b      # contrôle qualité (≈ 30 min)
uv run python run_multidomain.py                 # entraînements et évaluations (≈ 1 h 30 sur un GPU)
uv run python export_onnx.py --model multi       # export ONNX du modèle 12 domaines
```

Les données générées et vérifiées sont versionnées dans `experiments/data/` : vous pouvez sauter les deux premières étapes.

## 6. Publier un modèle sur le Hugging Face Hub

Les modèles (plus de 400 Mo) ne vont pas dans git. Publiez le dossier ONNX sur le Hub :

```bash
hf upload polymorfis/krito-nli-fr-multi experiments/models/krito-nli-fr-multi \
    --include "model.opt.onnx" "tokenizer.json" "labels.json"
```

Ajoutez une fiche de modèle (`README.md`) : origine (`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, licence MIT), données d'entraînement, résultats et limites. **Avant de publier**, vérifiez les conditions d'utilisation des LLM qui ont généré ou filtré les données (Gemma, Qwen) : certaines encadrent l'usage de leurs sorties pour entraîner d'autres modèles.

Les utilisateurs chargent ensuite le modèle directement :

```python
KritoEngine.from_onnx("polymorfis/krito-nli-fr-multi")
```
