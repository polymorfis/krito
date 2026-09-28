# Krito : étude multi-domaines

Rapport d'expérience du 28 septembre 2026, suite de [RESULTS.md](RESULTS.md). Tout est reproductible avec les scripts de ce dossier.

## Résumé

La première étude reposait sur un seul domaine d'entraînement (support client). Cette étude ajoute **12 domaines** et **4 053 textes d'entraînement**, et mesure le vrai zero-shot : **on entraîne sur 11 domaines et on teste sur le 12e, jamais vu**.

| Question | Réponse |
|---|---|
| Entraîner sur plusieurs domaines améliore-t-il le zero-shot ? | **Oui, nettement.** Sur un domaine jamais vu, la précision moyenne passe de 58 % (modèle de base) et 79 % (fine-tuné sur le support seul) à **88 %**. |
| Et si le domaine fait partie de l'entraînement ? | **95 %**, contre 97 % pour un LLM de 27 milliards de paramètres. C'est au-dessus du classifieur supervisé à catégories fixes (94 %), avec un seul modèle pour tous les domaines et des options libres. |
| Le modèle rejette-t-il les hors-sujet ? | Oui : AUROC hors-sujet de 95 à 96 % en moyenne. |
| Quels domaines restent difficiles ? | Ceux à jargon : **programmation (67 %)**, support informatique et communication (79 %) quand ils sont exclus de l'entraînement. Il faut des données de ces domaines. |
| La version serverless (ONNX int8) tient-elle ? | Oui : 95,9 % en moyenne, contre 94,8 % en fp32 (écart dans le bruit). |

**Recommandation :** faire du modèle entraîné sur les 12 domaines (`krito-nli-fr-multi`) le modèle de référence de Krito, et le valider sur de vraies données avant de le publier.

## Jeu de données

### Taxonomie

12 nouveaux domaines de 6 à 7 catégories chacun, avec volontairement des catégories voisines ([data/taxonomy.json](data/taxonomy.json)) :

| Domaine | Catégories |
|---|---|
| Juridique | droit du travail, famille, succession, consommation, pénal, sociétés |
| Programmation | erreur d'exécution, dépendances, performance, base de données, déploiement, conception |
| Compta / finance | fiscalité, clôture, trésorerie, recouvrement, placements, audit |
| Marketing | SEO, publicité payante, réseaux sociaux, e-mailing, étude de marché, image de marque |
| Auto / moto | moteur, freinage, électricité, pneus / suspension, entretien, carrosserie |
| Communication | presse, interne, crise, événementiel, prise de parole, contenus |
| Actualités | sport, économie, sciences / tech, culture, faits divers, environnement |
| Politiques publiques | santé publique, budget / impôts, éducation, sécurité / justice, énergie / climat, travail / retraites, international / défense |
| Santé (administratif) | rendez-vous, remboursement, ordonnance, résultats, téléconsultation, formalités |
| Immobilier | achat, location, travaux, estimation, copropriété, crédit |
| Support IT / sécurité | hameçonnage, logiciel malveillant, accès, sauvegarde, réseau, perte / vol d'appareil |
| Avis e-commerce | qualité produit, livraison, service client, prix, conformité, site web |

La politique est classée **par thème** (santé, fiscalité…), jamais par orientation, et sans personnalité réelle. La santé se limite à l'**administratif**, sans aucune demande de diagnostic.

### Trois auteurs différents

| Rôle | Auteur | Volume |
|---|---|---|
| **Entraînement** | `gemma4:26b` (local, Ollama) | 4 120 générés, 4 053 gardés |
| **Contrôle des étiquettes** | `qwen3.8:27b` (local, Ollama) | contre-annotation de tous les textes |
| **Test « gold »** | Claude, écrits **avant** la génération | 29 à 33 par domaine (≈ 24 du domaine + 5 hors-sujet proches) |

L'auteur des données de test n'est jamais celui des données d'entraînement. Les jeux support et RH de la première étude (également écrits par Claude) servent uniquement au test.

**Génération** ([generate.py](generate.py)) : pour chaque catégorie, 5 lots de 10 textes, chacun avec un style imposé : court, long avec le sujet à la fin, SMS avec fautes, e-mail formel, formulation indirecte. Actualités, politique et avis e-commerce ont leurs propres styles (titres, dépêches, questions de citoyens, avis mitigés…). S'y ajoutent 40 hors-sujet « proches » par domaine : même contexte, aucune catégorie ne convient. Médiane : 27 mots par texte, maximum 88.

**Contrôle qualité** ([verify.py](verify.py)) :
- suppression des doublons exacts et des quasi-doublons (cosinus > 0,95) ;
- suppression des textes générés trop proches d'un texte gold, pour éviter toute fuite vers le test ;
- contre-annotation par Qwen : les textes en désaccord avec l'étiquette demandée sont écartés.

L'accord Gemma / Qwen est de **97 à 100 %**. Les 27 désaccords sont surtout de vraies ambiguïtés : « le volant tremble au freinage » (freins ou suspension ?), ou une sollicitation commerciale reçue au support informatique (classée *hameçonnage* par Qwen).

## Protocole

- **Modèle** : `mDeBERTa-v3-base-xnli` fine-tuné comme dans la première étude. Paires (texte, hypothèse) : *entailment* pour la bonne catégorie, *contradiction* pour 2 autres catégories du domaine et pour les hors-sujet. 5 gabarits de phrase, 2 époques, embeddings gelés, 256 tokens maximum. Environ 6 min 30 par entraînement sur une RTX 3060 (deux entraînements en parallèle, un par GPU).
- **Un domaine exclu** : 12 entraînements, chacun sur 11 domaines générés ; évaluation sur le 12e (gold et générés).
- **Tous domaines** : un entraînement sur les 12 domaines générés, évalué sur les gold (même domaine, autre auteur) et sur support / RH (jamais vus).
- **Références** : `mDeBERTa` sans entraînement ; similarité d'embeddings `e5-base` ; modèle de la première étude (support seul) ; `e5-base` + régression logistique entraînée par domaine sur les textes générés (catégories fixes) ; Qwen 27B.
- **Métriques** : celles de la première étude ([lib.py](lib.py)). Confiance = probabilité d'implication absolue de l'option retenue.

## Résultats

### Vue d'ensemble (moyenne sur les 12 nouveaux domaines, jeux gold)

| Approche | Options libres ? | Précision | AUROC hors-sujet | Auto @90 % |
|---|:-:|:-:|:-:|:-:|
| `mDeBERTa` sans entraînement | oui | 58 % | 64 % | 16 % |
| Similarité d'embeddings `e5-base` | oui | 80 % | 80 % | 42 % |
| Fine-tuné sur le support seul (étude 1) | oui | 79 % | 94 % | 57 % |
| **Fine-tuné sur 11 domaines, testé sur le 12e** | oui | **88 %** | **95 %** | **75 %** |
| **Fine-tuné sur les 12 domaines** | oui | **95 %** | **96 %** | **91 %** |
| `e5-base` + régression logistique, par domaine | non | 94 % | 91 % | 85 % |
| LLM `qwen3.8:27b` | oui | 97 % | — | — |

Et sur les jeux de la première étude, **jamais utilisés à l'entraînement** :

| | Support client | RH |
|---|:-:|:-:|
| `mDeBERTa` sans entraînement | 56 % | 75 % |
| Fine-tuné sur le support (étude 1) | 94 % ¹ | 87 % |
| **Fine-tuné sur les 12 domaines** | **87 %** | **92 %** |

¹ En validation croisée, le support étant son domaine d'entraînement.

### Ce qu'il faut retenir

1. **La diversité des domaines est le levier principal du zero-shot.** À domaine inconnu, passer d'un domaine d'entraînement à 11 fait gagner 9 points de précision (79 → 88 %) et 18 points de traitement automatique (57 → 75 %). Sur les textes générés du domaine exclu (environ 300 par domaine, donc plus stables), la précision passe de 57 % à **91 %**.
2. **Des données synthétiques du bon domaine suffisent à s'approcher d'un LLM.** Le modèle 12 domaines (425 Mo en int8, environ 0,2 s par décision sur CPU) atteint 95 % sur des textes d'un autre auteur, contre 97 % pour le LLM 27B (17 Go, 3,8 s sur GPU).
3. **Un seul modèle bat 12 classifieurs spécialisés.** Le NLI 12 domaines fait mieux que le supervisé `e5-base` + régression logistique entraîné domaine par domaine (95 % contre 94 %, 91 % contre 85 % de traitement automatique), tout en acceptant de nouvelles catégories sans réentraînement.
4. **Le jargon reste le point faible du zero-shot.** Exclue de l'entraînement, la programmation plafonne à 67 % sur le gold, sous la simple similarité d'embeddings (71 %). Une fois incluse, elle monte à 88 %. Le support informatique et la communication (79 %) suivent le même schéma.
5. **Le seuil `min_entailment` doit être calibré par modèle, sur de vraies données.**
   - Le modèle 12 domaines est plus prudent que celui de la première étude : ses probabilités absolues sont plus basses et dépendent du gabarit. Exemple : 0,53 avec « Ce texte concerne », 0,02 avec « Ce message concerne », pour un message pourtant bien classé.
   - Calibrer sur les données d'entraînement ne marche pas : le modèle y est sûr de lui. Le seuil obtenu (0,0035) ne rejette que 70 % des hors-sujet sur le gold.
   - Sur le gold, `min_entailment=0.5` donne 90 % de précision, 85 % de traitement automatique, et rejette 83 % des hors-sujet.
6. **La quantification int8 ne coûte rien en moyenne** : 95,9 % contre 94,8 % en fp32 sur les 12 domaines. Les écarts par domaine (−4 à +8 points) sont du même ordre qu'une ou deux erreurs sur 24 exemples.

## Tableaux détaillés

Générés par [summarize_multidomain.py](summarize_multidomain.py) depuis [results/multidomain.json](results/multidomain.json).

#### Précision sur les jeux gold (messages du domaine)

| Domaine | mDeBERTa zero-shot | Embeddings e5 zero-shot | Fine-tuné support seul | **Fine-tuné 11 domaines (domaine exclu)** | Fine-tuné 12 domaines | Supervisé e5 + LR | LLM 27B |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Juridique | 62 % | 88 % | 88 % | 96 % | 100 % | 96 % | 100 % |
| Programmation | 42 % | 71 % | 54 % | 67 % | 88 % | 92 % | 100 % |
| Compta / finance | 62 % | 71 % | 83 % | 92 % | 88 % | 92 % | 92 % |
| Marketing | 54 % | 88 % | 96 % | 88 % | 96 % | 96 % | 100 % |
| Auto / moto | 42 % | 71 % | 88 % | 92 % | 96 % | 96 % | 96 % |
| Communication | 46 % | 75 % | 62 % | 79 % | 96 % | 88 % | 92 % |
| Actualités | 79 % | 88 % | 83 % | 96 % | 100 % | 96 % | 100 % |
| Politiques publiques | 89 % | 93 % | 93 % | 89 % | 96 % | 96 % | 100 % |
| Santé (admin.) | 46 % | 75 % | 71 % | 92 % | 92 % | 88 % | 92 % |
| Immobilier | 67 % | 83 % | 79 % | 92 % | 92 % | 96 % | 96 % |
| Support IT / sécu | 46 % | 79 % | 62 % | 79 % | 96 % | 96 % | 100 % |
| Avis e-commerce | 58 % | 79 % | 88 % | 96 % | 100 % | 100 % | 100 % |
| Support client (étude 1) | 56 % | 72 % | — | — | 87 % | — | — |
| RH (étude 1) | 75 % | 84 % | 87 % | — | 92 % | — | — |
| **Moyenne des 12 nouveaux domaines** | **58 %** | **80 %** | **79 %** | **88 %** | **95 %** | **94 %** | **97 %** |

#### AUROC hors-sujet sur les jeux gold

| Domaine | mDeBERTa zero-shot | Embeddings e5 zero-shot | Fine-tuné support seul | **Fine-tuné 11 domaines (domaine exclu)** | Fine-tuné 12 domaines | Supervisé e5 + LR |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Juridique | 63 % | 69 % | 98 % | 96 % | 98 % | 93 % |
| Programmation | 61 % | 82 % | 84 % | 99 % | 74 % | 100 % |
| Compta / finance | 82 % | 92 % | 100 % | 100 % | 99 % | 94 % |
| Marketing | 49 % | 92 % | 85 % | 93 % | 99 % | 85 % |
| Auto / moto | 48 % | 88 % | 95 % | 97 % | 98 % | 91 % |
| Communication | 73 % | 85 % | 99 % | 98 % | 100 % | 80 % |
| Actualités | 79 % | 47 % | 94 % | 98 % | 100 % | 99 % |
| Politiques publiques | 55 % | 72 % | 99 % | 98 % | 100 % | 98 % |
| Santé (admin.) | 88 % | 95 % | 98 % | 100 % | 98 % | 88 % |
| Immobilier | 80 % | 80 % | 98 % | 98 % | 98 % | 82 % |
| Support IT / sécu | 43 % | 97 % | 99 % | 92 % | 100 % | 100 % |
| Avis e-commerce | 47 % | 56 % | 77 % | 77 % | 86 % | 85 % |
| Support client (étude 1) | 50 % | 84 % | — | — | 94 % | — |
| RH (étude 1) | 57 % | 87 % | 90 % | — | 94 % | — |
| **Moyenne des 12 nouveaux domaines** | **64 %** | **80 %** | **94 %** | **95 %** | **96 %** | **91 %** |

#### Traités automatiquement à 90 % de précision (gold)

| Domaine | mDeBERTa zero-shot | Embeddings e5 zero-shot | Fine-tuné support seul | **Fine-tuné 11 domaines (domaine exclu)** | Fine-tuné 12 domaines | Supervisé e5 + LR |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Juridique | 4 % | 29 % | 79 % | 92 % | 100 % | 88 % |
| Programmation | 8 % | 0 % | 25 % | 46 % | 58 % | 92 % |
| Compta / finance | 38 % | 38 % | 71 % | 92 % | 88 % | 79 % |
| Marketing | 17 % | 75 % | 79 % | 21 % | 96 % | 67 % |
| Auto / moto | 0 % | 38 % | 88 % | 83 % | 92 % | 79 % |
| Communication | 8 % | 50 % | 21 % | 67 % | 96 % | 79 % |
| Actualités | 42 % | 21 % | 79 % | 96 % | 100 % | 96 % |
| Politiques publiques | 46 % | 46 % | 89 % | 86 % | 96 % | 96 % |
| Santé (admin.) | 12 % | 58 % | 54 % | 92 % | 88 % | 75 % |
| Immobilier | 8 % | 54 % | 25 % | 88 % | 92 % | 79 % |
| Support IT / sécu | 4 % | 46 % | 21 % | 50 % | 96 % | 96 % |
| Avis e-commerce | 8 % | 50 % | 54 % | 83 % | 96 % | 92 % |
| Support client (étude 1) | 11 % | 5 % | — | — | 75 % | — |
| RH (étude 1) | 16 % | 44 % | 73 % | — | 89 % | — |
| **Moyenne des 12 nouveaux domaines** | **16 %** | **42 %** | **57 %** | **75 %** | **91 %** | **85 %** |

#### Précision sur les textes générés du domaine exclu (~300 par domaine)

| Domaine | mDeBERTa zero-shot | **Fine-tuné 11 domaines (domaine exclu)** |
|---|:-:|:-:|
| Juridique | 63 % | 93 % |
| Programmation | 39 % | 80 % |
| Compta / finance | 63 % | 90 % |
| Marketing | 56 % | 94 % |
| Auto / moto | 48 % | 87 % |
| Communication | 43 % | 90 % |
| Actualités | 91 % | 98 % |
| Politiques publiques | 74 % | 97 % |
| Santé (admin.) | 50 % | 95 % |
| Immobilier | 58 % | 91 % |
| Support IT / sécu | 50 % | 86 % |
| Avis e-commerce | 46 % | 90 % |
| Support client (étude 1) | — | — |
| RH (étude 1) | — | — |
| **Moyenne des 12 nouveaux domaines** | **57 %** | **91 %** |

#### Qualité du jeu généré

| Domaine | Générés | Doublons retirés | Trop proches du gold | Accord Gemma / Qwen | Gardés |
|---|:-:|:-:|:-:|:-:|:-:|
| Juridique | 340 | 1 | 0 | 99.4 % | 337 |
| Programmation | 340 | 2 | 0 | 100.0 % | 338 |
| Compta / finance | 340 | 0 | 0 | 99.1 % | 337 |
| Marketing | 340 | 1 | 0 | 100.0 % | 339 |
| Auto / moto | 340 | 5 | 1 | 99.7 % | 333 |
| Communication | 331 | 2 | 0 | 99.7 % | 328 |
| Actualités | 340 | 2 | 1 | 100.0 % | 337 |
| Politiques publiques | 389 | 0 | 0 | 100.0 % | 389 |
| Santé (admin.) | 340 | 9 | 5 | 98.8 % | 322 |
| Immobilier | 340 | 5 | 0 | 100.0 % | 335 |
| Support IT / sécu | 340 | 3 | 0 | 97.9 % | 330 |
| Avis e-commerce | 340 | 2 | 1 | 97.3 % | 328 |
| **Total** | **4120** | | | | **4053** |

## Limites

1. **Aucun texte n'est écrit par un humain.** L'entraînement vient de Gemma, les tests de Claude. Les deux sont des LLM : leurs textes sont plus propres et plus explicites que de vrais messages. Les chiffres absolus sont donc probablement optimistes. En revanche, les **comparaisons** entre méthodes restent valables, car toutes sont évaluées sur les mêmes textes.
2. **Les jeux gold sont petits** : environ 24 messages du domaine et 5 hors-sujet par domaine. Une erreur vaut 4 points ; un chiffre par domaine a une marge d'environ ±8 à 10 points. Les moyennes sur 12 domaines (environ 290 messages) sont fiables à ±3 points environ.
3. **Le filtre Qwen favorise les textes non ambigus** (99 % d'accord). Les vrais messages le sont bien plus souvent, avec plusieurs sujets ou des catégories qui se chevauchent.
4. **Les hors-sujet « actualités » sont parfois hors genre** (des candidatures au lieu d'articles), donc faciles à rejeter.
5. **Qwen sert deux fois** : filtre des données d'entraînement, puis référence LLM sur le gold. Le gold n'étant pas filtré, la référence n'est pas biaisée, mais elle est évaluée sur des textes d'un autre LLM.
6. **Aucun réglage d'hyperparamètres** (2 époques, taux d'apprentissage 2e-5, 2 négatifs par exemple). Il y a probablement de la marge.

## Prochaines étapes

1. **Valider sur de vrais messages**, en priorité dans un ou deux domaines métier réels de POLYMORFIS. C'est aussi la seule façon de calibrer `min_entailment`.
2. Ajouter des domaines techniques (programmation, informatique, industrie) et des messages multi-sujets.
3. Publier `krito-nli-fr-multi` sur le Hugging Face Hub après cette validation, et en faire le défaut de Krito.
4. Ajouter à Krito une calibration du seuil (`engine.calibrate(exemples_annotés)`), puisque l'échelle absolue dépend du modèle.

## Reproduire

```bash
uv sync --all-groups
cd experiments
uv run python generate.py --model gemma4:26b       # ~45 min, Ollama
uv run python verify.py --model qwen3.8:27b        # ~30 min, Ollama
ollama stop qwen3.8:27b                             # libérer les GPU
uv run python run_multidomain.py                    # ~1 h 30 sur un GPU
# ou, sur deux GPU : ajouter en parallèle
#   CUDA_VISIBLE_DEVICES=0 uv run python run_multidomain.py --lodo-only
uv run python export_onnx.py --model multi          # ONNX int8 du modèle 12 domaines
uv run python summarize_multidomain.py > results/multidomain_tables.md
```

Sur une machine de 16 Go de RAM, le contexte d'Ollama doit rester à 4 096 tokens pour Qwen 27B (voir `verify.py`) : à 8 192, le service a été tué par le noyau par manque de mémoire.
