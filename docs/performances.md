# Performances

Tous les chiffres de ce document ont été **mesurés**, jamais estimés. Chacun renvoie à son script et à ses données brutes. Lisez aussi la section [Limites](#limites-des-mesures) : elle dit ce que ces chiffres ne prouvent pas.

## En bref

| | Valeur | Conditions |
|---|:-:|---|
| Précision, domaine **jamais vu** | **88 %** | Moyenne sur 12 domaines ; entraîné sur les 11 autres. |
| Précision, domaine vu à l'entraînement | **95 %** | Moyenne sur 12 domaines ; textes de test écrits par un autre auteur. |
| Référence : LLM local de 27 milliards de paramètres | 97 % | Mêmes textes. |
| Taille du modèle | **404 Mo** | ONNX int8 optimisé (`model.opt.onnx`). |
| RAM maximale | **710 Mo** | Processus Python complet (bibliothèque et modèle). |
| Dépendances | **154 Mo** | `krito[onnx]`, sans PyTorch. |
| Latence par décision (CPU de 2013) | **154 à 448 ms** | 6 options ; 4 à 1 thread. |
| GPU | **aucun** | |

## 1. Qualité

### Protocole

- **12 domaines** (juridique, programmation, compta / finance, marketing, auto / moto, communication, actualités, politiques publiques, santé administrative, immobilier, support informatique, avis e-commerce), soit 73 catégories.
- **Entraînement** : 4 053 textes générés par `gemma4:26b`, contre-annotés par `qwen3.8:27b`.
- **Test** : jeux « gold » écrits par un autre auteur (Claude), environ 24 messages du domaine et 5 hors-sujet par domaine, plus 290 tickets de support et 85 demandes RH jamais utilisés à l'entraînement.
- **Métriques** : précision sur les messages du domaine ; AUROC hors-sujet (capacité de la confiance à distinguer un message du domaine d'un hors-sujet, 1 = parfait) ; « traité automatiquement » = part maximale des messages acceptés en gardant 90 % de précision.

Rapport complet : [RESULTS_MULTIDOMAIN.md](../experiments/RESULTS_MULTIDOMAIN.md).

### Résultats (moyenne sur les 12 domaines)

| Approche | Options libres | Précision | AUROC hors-sujet | Traité automatiquement à 90 % |
|---|:-:|:-:|:-:|:-:|
| NLI multilingue de base (mDeBERTa, sans fine-tuning) | oui | 58 % | 64 % | 16 % |
| Similarité d'embeddings (e5-base) | oui | 80 % | 80 % | 42 % |
| Fine-tuné sur un seul domaine (support) | oui | 79 % | 94 % | 57 % |
| **Krito, domaine jamais vu** (11 domaines → 12e) | oui | **88 %** | **95 %** | **75 %** |
| **Krito `krito-nli-fr-multi`** (12 domaines) | oui | **95 %** | **96 %** | **91 %** |
| Classifieur supervisé par domaine (e5-base + régression logistique) | non | 94 % | 91 % | 85 % |
| LLM `qwen3.8:27b` (17 Go, GPU) | oui | 97 % | — | — |

### Par domaine (Krito 12 domaines, jeux gold)

| Domaine | Précision | Domaine | Précision |
|---|:-:|---|:-:|
| Juridique | 100 % | Actualités | 100 % |
| Programmation | 88 % | Politiques publiques | 96 % |
| Compta / finance | 88 % | Santé (administratif) | 92 % |
| Marketing | 96 % | Immobilier | 92 % |
| Auto / moto | 96 % | Support IT / sécurité | 96 % |
| Communication | 96 % | Avis e-commerce | 100 % |

Sur les domaines jamais vus, les plus difficiles sont ceux à jargon : programmation (67 %), support informatique et communication (79 %).

### Fine-tuning sur vos données

Partant de `krito-nli-fr-multi`, un fine-tuning sur **202 tickets de support annotés** fait passer la précision sur 72 autres tickets de **84,7 % à 91,7 %** ([finetune_custom.py](../experiments/finetune_custom.py)).

## 2. Ressources CPU et mémoire

### Bibliothèque, installation minimale

Environnement vierge Python 3.12 avec `krito[onnx]` uniquement ; modèle `krito-nli-fr-multi` (`model.opt.onnx`) ; 6 options par décision ; Intel i7-4770K (2013, 4 cœurs / 8 threads, AVX2) ; 40 décisions sur 5 messages de longueurs variées.

| Threads | Latence médiane | p95 | RAM max du processus |
|:-:|:-:|:-:|:-:|
| 1 | 448 ms | 708 ms | 710 Mo |
| 2 | 246 ms | 380 ms | 710 Mo |
| 4 | 154 ms | 236 ms | 708 Mo |

- Import et chargement : 1,4 à 2,6 s.
- Dépendances installées : 154 Mo (contre 5,6 Go pour l'environnement PyTorch avec CUDA).

### Effet des optimisations mémoire

| Variante | Fichier | RAM max | Latence (2 threads) | Précision (12 domaines) |
|---|:-:|:-:|:-:|:-:|
| ONNX fp32 | 1 116 Mo | — | — | 94,8 % |
| ONNX int8, quantification par défaut | 338 Mo | — | — | ≈ 30 % (inutilisable) |
| ONNX int8, quantification sélective | 425 Mo | 869 Mo | 295 ms | 95,9 % |
| **+ optimisation hors ligne** (`model.opt.onnx`) | **404 Mo** | **710 Mo** | 296 ms | **95,9 %** (prédictions identiques) |

### Conteneurs

| Déploiement | Limites | Démarrage | RAM | Latence |
|---|---|:-:|:-:|:-:|
| API FastAPI (Docker) | 1 Go, 2 CPU | 2,9 s | 651 à 701 Mio | p50 278 ms (6 options), 160 ms (3 options) |
| AWS Lambda (émulateur officiel) | 1 Go, 2 CPU | 1,8 s à froid | 655 Mio | p50 167 ms à chaud (3 options) |

Images : 735 Mo (API) ; image Lambda plus lourde (base AWS). Mesures de latence HTTP de bout en bout, sur 40 requêtes.

### Production du modèle (étape unique)

| Étape | Pic de RAM | Durée |
|---|:-:|:-:|
| Fine-tuning d'environ 3 300 textes (11 domaines) | — | ≈ 6 min 30 sur RTX 3060 |
| Export ONNX fp32 | ≈ 5,4 Go | ≈ 8 s |
| Quantification + optimisation | ≈ 7,3 Go | ≈ 13 s |

## Limites des mesures

1. **Aucun texte n'est écrit par un humain.** L'entraînement vient d'un LLM (Gemma) et les tests d'un autre (Claude). Les messages réels sont plus bruités et plus ambigus : **attendez-vous à des chiffres plus bas sur vos données**. Les comparaisons entre méthodes, faites sur les mêmes textes, restent valables.
2. **Les jeux gold sont petits.** Environ 24 messages du domaine par domaine : un chiffre par domaine a une marge d'environ ±8 à 10 points. Les moyennes sur 12 domaines sont fiables à environ ±3 points.
3. **Un seul processeur mesuré**, ancien (2013, sans instructions VNNI). Un serveur récent devrait être plus rapide, notamment en int8 ; ce n'est pas mesuré.
4. **AWS Lambda n'a pas été testé sur AWS**, seulement dans l'émulateur officiel.
5. **Les seuils ne se transposent pas.** L'échelle de `entailment_prob` dépend du modèle et du gabarit : calibrez les garde-fous sur vos données ([guide](guide-utilisateur.md#calibrer-sur-vos-données)).

## Reproduire

| Mesure | Commande |
|---|---|
| Qualité 12 domaines | `cd experiments && uv run python run_multidomain.py && uv run python summarize_multidomain.py` |
| Latence et RAM | [benchmarks/bench_cpu.py](../benchmarks/bench_cpu.py), dans un environnement `krito[onnx]` sans PyTorch |
| Conteneur API | voir [integration.md](integration.md#image-docker) |
| Captures d'écran | `uv run --group site python site/tools/capture.py` |
