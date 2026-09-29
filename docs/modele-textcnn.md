# Modèle léger TextCNN (`krito-textcnn-fr`)

Krito livre, **dans le paquet lui-même**, un petit modèle entraîné de zéro sur les données du dépôt. Il n'utilise aucun poids pré-entraîné (ni BERT ni DeBERTa) et ne télécharge rien : il fonctionne hors ligne dès l'installation de `krito[onnx]`.

```python
from krito import KritoEngine

engine = KritoEngine.from_textcnn()          # modèle livré, gabarit d'hypothèse de l'entraînement
engine.classify("Mon colis n'est jamais arrivé.", options, min_entailment=0.5)
```

Toutes les primitives de Krito fonctionnent avec ce modèle : `classify`, `yes_no`, `scale`, les versions par lots, `calibrate` et le juge de confiance.

## Quand l'utiliser ?

| | `krito-textcnn-fr` | `krito-nli-fr-multi` (mDeBERTa fine-tuné) | mDeBERTa zero-shot (défaut PyTorch) |
|---|:-:|:-:|:-:|
| Taille du modèle | **{SIZE} Mo** | 404 Mo | 1,1 Go |
| Paramètres | {PARAMS} M | 278 M | 278 M |
| Téléchargement | **aucun** (dans le paquet) | Hugging Face Hub | Hugging Face Hub |
| Latence CPU, 6 options (p50) | **{LAT6} ms** | ~230 ms | ~0,2–1,3 s |
| Pic de RAM du processus | **{RAM} Mo** | 710 Mo | plusieurs Go |
| Précision moyenne, 12 domaines vus à l'entraînement (gold) | {FTALL} % | 95 % | — |
| Précision moyenne, domaine **jamais vu** (un domaine exclu, gold) | {LODO} % | 88 % | 58 % |
| AUROC hors-sujet, domaine jamais vu | {LODOAUC} % | 95 % | 64 % |
| Entraînement complet sur CPU (4 cœurs) | **~{TRAIN} min** | ~1 h 30 sur GPU | — |

Mesures : jeux gold de [RESULTS_MULTIDOMAIN.md](../experiments/RESULTS_MULTIDOMAIN.md), même protocole ; résultats bruts dans [experiments/results/textcnn.json](../experiments/results/textcnn.json). Latence et mémoire : [benchmarks/bench_textcnn.py](../benchmarks/bench_textcnn.py), 1 thread, conteneur Linux 4 cœurs.

**En résumé** : le TextCNN est plus de 50 fois plus petit et environ 100 fois plus rapide. Il convient aux taxonomies **proches des 12 domaines d'entraînement** ou à un modèle **réentraîné sur vos données** en quelques minutes sur CPU. Pour des catégories vraiment nouvelles, sans exemples annotés, `krito-nli-fr-multi` reste nettement meilleur : un modèle sans pré-entraînement ne connaît que le vocabulaire qu'il a vu.

## Architecture

```
contexte ──► tokenizer WordPiece ──► TextCNN ─┬─► tokens [L, H] ──────────────┐
                                              └─► bi-encodeur ──► vecteur c ──┤
option i ──► tokenizer WordPiece ──► TextCNN ─┬─► tokens [L, H] ──────────────┤
                                              └─► bi-encodeur ──► vecteur oᵢ ─┤
                                                                              ▼
                  n options ≤ threshold (10) : toutes les options ──► cross-encodeur ──► logits (E, C, N)
                  n options > threshold      : top_k (5) de c·oᵢ ──► cross-encodeur ──► logits (E, C, N)
                                               autres options ──► (−10, 10, 0) : probabilité ≈ 0
```

- **Backbone TextCNN** (partagé) : embeddings de sous-mots (vocabulaire WordPiece de 8 000, minuscules, sans accents), convolutions 1D de noyaux 1, 3 et 5, puis une convolution dilatée résiduelle. Les convolutions conservent la longueur, ce qui fournit une représentation **par token**. Le padding est masqué à chaque couche et dans le pooling.
- **Bi-encodeur** : pooling max + moyenne, puis projection normalisée. Contexte et options sont encodés **indépendamment**, et le score est un produit scalaire.
- **Cross-encodeur** : chaque token du contexte est aligné par attention sur ceux de l'option, et inversement (comme dans ESIM). Il est comparé à son alignement ([x, x̃, x − x̃, x ⊙ x̃]), puis le modèle produit **3 logits NLI** (entailment, contradiction, neutral). Ce sont les mêmes sorties que les cross-encoders NLI : les garde-fous, la calibration et le juge fonctionnent sans changement.
- **`AdaptiveSystem1Classifier`** : routage « System 1 ». Chaque texte distinct n'est encodé qu'une fois par appel ; le cross-encodeur ne travaille que sur des caractéristiques déjà calculées.

Réglage du routage : `KritoEngine.from_textcnn(threshold=20, top_k=8)`. Les options écartées apparaissent dans `result.scores` avec `decision_score = -20` et une probabilité quasi nulle.

## Réentraîner sur vos données

L'entraînement part de zéro (tokenizer compris) : il demande `krito[torch]` et `onnxscript` (groupe `experiments`), mais **pas de GPU**.

```bash
uv sync --all-groups
uv run krito-textcnn \
    --domain mes_messages.csv mes_categories.json \
    --eval mes_messages_test.csv mes_categories.json \
    --out modeles/krito-textcnn-maboite
```

- CSV : colonnes `text` et `label` (`none` pour les hors-sujet), taxonomie JSON `{"etiquette": "description"}`, comme pour [le fine-tuning](entrainement.md).
- `--domain` se répète : ajoutez les 12 domaines du dépôt pour garder un modèle généraliste (chaque domaine apporte aussi des négatifs « sans rapport », étiquetés *neutral*).
- `--threshold` / `--top-k` règlent le routage enregistré dans `config.json` ; `--quantize` produit une variante int8.

```python
engine = KritoEngine.from_textcnn("modeles/krito-textcnn-maboite")
```

Depuis Python : `krito.textcnn.train.train([...])`, `evaluate(...)` et `export_onnx(...)`. Le dossier `torch/` produit par la commande contient les poids PyTorch (`TorchTextCNNBackend.load`).

### Objectifs d'entraînement

Chaque texte annoté produit des paires : sa catégorie → *entailment* ; une autre catégorie du domaine → *contradiction* ; une catégorie d'un autre domaine → *neutral* ; un texte `none` face aux catégories de son domaine → *contradiction*. Les hypothèses varient entre 5 gabarits. S'y ajoute un objectif contrastif du bi-encodeur : le texte doit être plus proche de sa catégorie que des autres catégories de son domaine. Réglages par défaut : 20 époques, AdamW (lr 5e-3, décroissance cosinus), 10 % de mots remplacés par `[UNK]`, dropout 0,2.

Réglages comparés (précision moyenne gold, 12 domaines vus, une graine) :

| Variante | Précision |
|---|:-:|
| Poids contrastif 0,5, lr 3e-3 | 73 % |
| + régularisation (dropout 0,3, 25 % de mots masqués) | 70 % |
| + négatifs (4 contradictions, 2 neutres par texte) | 72 % |
| Modèle plus large (embeddings 192, 96 filtres) | 76 % |
| **Poids contrastif 1,0, lr 5e-3 (retenu)** | **80 %** |
| Poids contrastif 2,0, lr 5e-3 | 75 % |

### Export

`export_onnx` écrit deux graphes, pour que chaque texte ne soit encodé qu'une fois :

| Fichier | Entrées → sorties |
|---|---|
| `encoder.onnx` | `input_ids`, `attention_mask` → `token_features [B, L, H]`, `sentence_embedding [B, D]` |
| `cross.onnx` | `p_tokens`, `p_mask`, `q_tokens`, `q_mask` → `nli_logits [B, 3]` (E, C, N) |
| `tokenizer.json`, `config.json` | tokenizer WordPiece ; hyperparamètres, routage, gabarit d'hypothèse |

L'export vérifie que le backend ONNX reproduit les logits PyTorch. Le backend (`krito.textcnn.TextCNNBackend`) ne dépend que d'ONNX Runtime et de tokenizers. Il est utilisable depuis plusieurs threads.

Reproduire le modèle livré et les mesures : `cd experiments && uv run python run_textcnn.py` (environ 1 h 30 sur CPU, dont 12 entraînements « un domaine exclu » ; `--skip-lodo` pour le seul modèle final).

## Limites

- **Données synthétiques** : comme pour les autres modèles du dépôt, les textes d'entraînement sont générés par un LLM. Validez sur vos données.
- **Vocabulaire fermé** : les mots jamais vus deviennent des sous-mots ou `[UNK]`. Le jargon d'un nouveau domaine est mal compris tant qu'il n'est pas dans les données d'entraînement.
- **Français uniquement.**
- **Une seule graine mesurée** : sur 24 à 28 textes gold par domaine, un texte vaut environ 4 points de précision.
- **Recalibrez les garde-fous** (`engine.calibrate`) : l'échelle de `entailment_prob` n'est pas celle des modèles NLI.
