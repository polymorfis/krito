# Krito : quelle approche pour de meilleures décisions ?

Rapport d'expérience du 28 septembre 2026. Tout est reproductible avec les scripts de ce dossier.

> **Suite :** l'étude multi-domaines ([RESULTS_MULTIDOMAIN.md](RESULTS_MULTIDOMAIN.md)) étend ces résultats à 12 domaines. Elle confirme le fine-tuning NLI et montre que la diversité des domaines est le principal levier du zero-shot (88 % sur un domaine jamais vu).

## Résumé

| Question | Réponse courte |
|---|---|
| Entraîner notre propre modèle améliore-t-il les résultats ? | **Oui, fortement.** Fine-tuner un NLI multilingue existant fait passer la précision de 59 % à **94 %** sur le domaine d'entraînement. |
| Perd-on le zero-shot (options libres à chaque appel) ? | **Non, au contraire.** Sur un domaine jamais vu (RH), la précision passe de 73 % à **87 %**. |
| scikit-learn est-il utile ? | Oui, mais pas comme classifieur TF-IDF (82 % au mieux). Il est très utile **au-dessus des embeddings** (95 % avec des catégories fixes) et comme **juge de confiance**. |
| Faut-il un système expert ou de la logique floue ? | Pas nécessaire. Un **juge appris** (régression logistique sur plusieurs signaux) fait mieux. Des règles métier restent utiles pour les contraintes impératives, sans être évaluables ici. |
| Peut-on faire du serverless ? | **Oui.** Export ONNX + quantification int8 : 425 Mo de modèle, 154 Mo de dépendances, démarrage à froid d'environ 3 s, sans PyTorch. |
| Où se situe un LLM ? | Référence haute : 96 % de précision, mais 3,8 s par décision sur GPU et 17 Go de modèle. |

**Recommandation :** faire du modèle NLI multilingue fine-tuné en français le cœur de Krito, servi en ONNX, avec le garde-fou `min_entailment`. Pour les clients à taxonomie fixe, ajouter un mode « quelques exemples » : embeddings, régression logistique et juge.

## Protocole

- **Données** ([data/](data/)) : 240 tickets de support en français répartis en 6 catégories (facturation, livraison, compte, résiliation, bug, commercial), plus 50 messages hors-sujet. Le jeu contient des fautes, du langage familier et des cas ambigus.
- **Domaine de transfert** : 75 demandes RH en 5 catégories (congés, paie, matériel, formation, notes de frais), plus 10 hors-sujet. **Il n'est jamais utilisé à l'entraînement**, ni pour le modèle ni pour le juge. Il mesure le zero-shot sur des catégories inconnues.
- **Évaluation** : validation croisée stratifiée en 5 plis (graine 42), avec les mêmes plis pour toutes les méthodes. Les hors-sujet ne sont jamais une classe à prédire : ils mesurent la capacité de rejet.
- **Métriques** ([lib.py](lib.py)) :
  - *Précision* : sur les messages du domaine.
  - *AUROC accept.* : capacité de la confiance à séparer une bonne décision d'une erreur ou d'un hors-sujet (0,5 = hasard, 1 = parfait).
  - *AUROC hors-sujet* : capacité de la confiance à séparer « dans le domaine » de « hors-sujet ».
  - *Auto @90 %* : part maximale des messages traités automatiquement en gardant au moins 90 % de précision parmi les décisions acceptées, hors-sujet acceptés comptés comme erreurs. Le seuil est fixé a posteriori : cette métrique compare les méthodes, elle ne prédit pas la production.

## Résultats

### 1. Zero-shot, sans entraînement

| Méthode | Précision | AUROC accept. | AUROC hors-sujet | Auto @90 % | RH précision |
|---|:-:|:-:|:-:|:-:|:-:|
| Krito v0.1 (`nli-deberta-v3-xsmall`, anglais) | 30 % | 0,69 | 0,51 | 4 % | 48 % |
| `nli-deberta-v3-base` (anglais) | 48 % | 0,73 | 0,55 | 8 % | 68 % |
| `multilingual-MiniLMv2-L6` | 30 % | 0,58 | 0,40 | 3 % | 60 % |
| `mDeBERTa-v3-base-xnli` (multilingue) | 59 % | 0,66 | 0,52 | 4 % | 73 % |
| Similarité d'embeddings `e5-small` | 64 % | 0,65 | 0,80 | 13 % | 79 % |
| Similarité d'embeddings `e5-base` | **71 %** | 0,72 | **0,84** | 18 % | **81 %** |

En zero-shot pur sur du français réaliste, la **simple similarité d'embeddings bat tous les modèles NLI**, et détecte bien mieux les hors-sujet.

### 2. Supervisé (catégories fixes), selon le nombre d'exemples par classe

| Méthode | 4 ex. | 8 ex. | 16 ex. | ~32 ex. | Auto @90 % (~32 ex.) |
|---|:-:|:-:|:-:|:-:|:-:|
| TF-IDF + régression logistique | 55 % | 62 % | 69 % | 82 % | 48 % |
| `e5-small` figé + régression logistique | 75 % | 81 % | 86 % | 90 % | 50 % ¹ |
| `e5-base` figé + régression logistique | **77 %** | **88 %** | **88 %** | **95 %** | 78 % ¹ |
| SetFit (`e5-small` fine-tuné, contrastif) | 78 % | 83 % | 88 % | 89 % | 77 % ¹ |

¹ Avec la confiance « plus proche voisin » : similarité à l'exemple d'entraînement le plus proche de la classe prédite. Elle détecte beaucoup mieux les hors-sujet (AUROC 0,86 contre 0,71 pour la probabilité de la régression logistique).

- Avec **8 exemples par classe**, `e5-base` + régression logistique atteint déjà 88 %.
- SetFit n'apporte rien de décisif par rapport à `e5-base` figé, qui ne demande aucun entraînement de réseau.
- TF-IDF est rapide et léger, mais faible sur les paraphrases.

### 3. NLI fine-tuné (garde les options libres)

Entraînement sur des paires (ticket, hypothèse) : *entailment* pour la bonne catégorie, *contradiction* pour 2 autres catégories tirées au hasard et pour les hors-sujet. Quatre gabarits de phrase, 3 époques, embeddings gelés ([run_nli_finetune.py](run_nli_finetune.py)). Environ 25 s par pli sur une RTX 3060.

| Modèle | Précision | AUROC accept. | AUROC hors-sujet | Auto @90 % | RH précision | RH AUROC hors-sujet |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `MiniLMv2-L6` fine-tuné | 66 % | 0,81 | 0,88 | 33 % | 71 % | 0,85 |
| **`mDeBERTa` fine-tuné** (conf. relative) | **94 %** | 0,90 | 0,90 | 84 % | **87 %** | 0,86 |
| **`mDeBERTa` fine-tuné** (conf. absolue `P(entailment)`) | **94 %** | **0,91** | **0,96** | **90 %** | **87 %** | **0,90** |

- **Aucun oubli catastrophique.** Le modèle progresse aussi sur le domaine RH, qu'il n'a jamais vu (73 % → 87 %). Il a appris la tâche « ce message concerne X » en français, pas seulement nos 6 catégories.
- La **confiance absolue** (probabilité d'implication de l'option retenue) devient le meilleur signal de rejet une fois le modèle fine-tuné. Avant fine-tuning, c'était l'inverse (0,43). C'est ce qui justifie le garde-fou `min_entailment`.
- `MiniLM` est trop petit pour cette tâche.

### 4. Juge de confiance (scikit-learn)

Une régression logistique prédit « cette décision est bonne » à partir d'une dizaine de signaux ([run_judge.py](run_judge.py)) : confiances et marges NLI et embeddings, implication absolue, accord entre les deux modèles, entropie, similarité au plus proche voisin.

| Pipeline | AUROC accept. sans → avec juge | Auto @90 % sans → avec juge | RH AUROC accept. sans → avec juge |
|---|:-:|:-:|:-:|
| Zero-shot `mDeBERTa` + `e5-base` | 0,76 → **0,91** | 35 % → 48 % | 0,86 → **0,92** |
| `mDeBERTa` fine-tuné + `e5-base` | 0,88 → 0,91 | 85 % → 90 % | 0,89 → 0,90 |
| Supervisé `e5-base` + régression logistique | 0,74 → **0,88** | 52 % → **87 %** | — |

- Le juge **transfère** : entraîné sur le support, il améliore aussi la confiance sur le domaine RH.
- Il est très utile quand le modèle de base est moyen (zero-shot, supervisé), et marginal avec le NLI fine-tuné, qui est déjà bien calibré. Comme il impose de charger un second modèle (`e5-base`), **il n'est pas intégré au cœur de Krito** pour l'instant.

**Prédiction conforme** (pipeline supervisé, α = 0,1) : la couverture réelle est de 94,6 % pour 90 % garantis, avec des ensembles de 1,06 option en moyenne. 89 % des tickets obtiennent un ensemble à une seule option, correcte dans 96,7 % des cas. En revanche, **66 % des hors-sujet reçoivent aussi un ensemble à une seule option** : la prédiction conforme ne protège pas contre les hors-sujet, il faut la combiner avec un juge ou `min_entailment`.

### 5. Référence LLM local (`qwen3.8:27b` via Ollama)

Sortie JSON contrainte à la liste des catégories, avec une option `none` ([run_llm.py](run_llm.py)).

| | Précision | AUROC accept. | Auto @90 % | Hors-sujet → `none` | RH précision | Latence (p50) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| LLM 27B | **96 %** | **0,93** | **96 %** | 94 % | **97 %** | **3,8 s** (GPU) |
| `mDeBERTa` fine-tuné | 94 % | 0,91 | 90 % | — | 87 % | 0,2 s (CPU) |

Le LLM reste meilleur, surtout en transfert (97 % contre 87 %). Mais il est environ 20 fois plus lent sur GPU, et son modèle pèse 17 Go contre 425 Mo. C'est exactement le compromis que vise Jev.

### 6. Serverless : export ONNX

Mesures sur un Intel i7-4770K (2013, AVX2, sans VNNI), 5 options par décision ([export_onnx.py](export_onnx.py)).

| Variante | Taille | Précision RH | Accord avec PyTorch | Latence 1 thread | Latence 2 threads |
|---|:-:|:-:|:-:|:-:|:-:|
| ONNX fp32 | 1 116 Mo | 86,7 % | 100 % | 405 ms | 228 ms |
| ONNX int8, quantification naïve | 338 Mo | **31 %** | 29 % | — | — |
| **ONNX int8, quantification sélective** | **425 Mo** | **86,7 %** | 93 % | 416 ms | 234 ms |

- **Piège rencontré :** la quantification dynamique par défaut détruit DeBERTa. Une mesure couche par couche montre que c'est la sortie du bloc feed-forward (`output.dense`, valeurs d'activation extrêmes) qui ne supporte pas l'int8. On la laisse en fp32 et on quantifie le reste, y compris la table d'embeddings.
- L'int8 ne fait pas gagner de vitesse sur ce processeur ancien, qui n'a pas les instructions VNNI. Un gain est probable sur un CPU récent, mais il n'a pas été mesuré.
- **Installation `krito[onnx]` seule** (environnement vierge, Python 3.12) :
  - dépendances : 154 Mo, contre 5,6 Go avec PyTorch ;
  - démarrage à froid (import, chargement du modèle, 1re décision) : 2,9 à 3,2 s ;
  - mémoire maximale : 0,9 Go ;
  - PyTorch n'est jamais importé.
- Dépendances et modèle (environ 580 Mo) dépassent la limite de 250 Mo d'une archive AWS Lambda. Il faut donc une **image conteneur** (limite de 10 Go), ou un modèle chargé depuis un stockage attaché.

## Limites de cette étude

1. **Les données sont synthétiques.** J'ai (Claude) rédigé les 375 exemples. Leur style est plus homogène que celui de vrais tickets, et **les chiffres du fine-tuning sont probablement optimistes**. La validation sur de vrais tickets clients annotés est la prochaine étape indispensable.
2. **Les échantillons sont petits.** 48 exemples par pli de test et 85 exemples RH : les écarts de moins de 3 à 5 points ne sont pas significatifs. Les écarts-types entre plis sont dans [results/](results/).
3. **Le seuil « Auto @90 % » est choisi a posteriori** sur les données de test. En production, il faudra le fixer sur un jeu de validation séparé.
4. **Un seul domaine d'entraînement.** Le transfert est mesuré sur un seul autre domaine (RH), assez proche : des demandes internes en français.
5. **Règles métier et logique floue non évaluées.** Ayant écrit les données, toute règle que j'écrirais serait biaisée.
6. **Cas ambigus.** Exemple : « erreur 500 lors du paiement » est annoté *bug*, mais le modèle fine-tuné le classe *facturation* quand seules 4 options sont proposées. Ce genre de cas relève d'une décision métier (priorités entre catégories), pas du modèle.

## Prochaines étapes proposées

1. Constituer 300 à 1 000 **vrais** tickets annotés, sans oublier les hors-sujet. C'est l'actif le plus important.
2. Enrichir l'entraînement NLI avec plusieurs domaines, par exemple des données synthétiques générées par un LLM sur 10 à 20 domaines puis relues, et évaluer sur un domaine vraiment éloigné.
3. Publier le modèle fine-tuné sur le Hugging Face Hub et en faire le modèle par défaut de Krito.
4. Ajouter un mode supervisé (`KritoClassifier.fit(exemples)` : `e5-base` + régression logistique + juge) pour les taxonomies fixes.
5. Mesurer la latence int8 sur un CPU récent et sur une vraie fonction serverless.

## Reproduire

```bash
uv sync --all-groups
cd experiments
uv run python features.py                               # cache des sorties zero-shot
uv run python run_zero_shot.py
uv run python run_supervised.py                         # dont SetFit (GPU)
uv run python run_nli_finetune.py --model mdeberta --fp32
uv run python run_nli_finetune.py --model minilm_multi
uv run python run_judge.py
uv run python run_llm.py --model qwen3.8:27b            # nécessite Ollama
uv run python export_onnx.py --model mdeberta           # modèle final + ONNX + mesures
```

Durée totale : environ 1 h 30 sur une RTX 3060, dont environ 1 h pour le LLM.
