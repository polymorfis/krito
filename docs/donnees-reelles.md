# Données réelles : annoter, contrôler, publier

Les modèles et les chiffres actuels de Krito viennent de **données synthétiques** écrites par des LLM. C'est la principale limite du projet : les chiffres absolus sont probablement optimistes. Ce document décrit le protocole pour constituer un jeu de **vraies données annotées**, s'en servir pour calibrer, entraîner et évaluer, puis publier un modèle.

Il sert deux publics :
- une **entreprise** qui veut régler Krito sur ses propres messages (les données restent chez elle) ;
- les **mainteneurs** qui veulent publier un modèle entraîné et évalué sur des données réelles.

## 1. Collecter

| Règle | Pourquoi |
|---|---|
| Tirez les messages **au hasard** dans le flux réel, sur plusieurs semaines. | Un échantillon choisi à la main est plus facile que la réalité. |
| Gardez les messages **tels quels** (fautes, abréviations, messages très courts). | Le modèle doit être évalué sur ce qu'il verra en production. |
| Gardez les **hors-sujet** (10 à 15 % visés). | Sans eux, impossible de régler le rejet (`min_entailment`, juge). |
| Visez **30 à 50 exemples par catégorie**, 200 à 500 au total. | En dessous, les seuils calibrés sont instables. |

## 2. Anonymiser

**Avant toute annotation partagée, tout fine-tuning publié et toute publication**, retirez les données personnelles : noms, e-mails, téléphones, adresses, numéros de client, de commande, IBAN, cartes bancaires, numéros de sécurité sociale. Remplacez-les par des marqueurs (`[NOM]`, `[EMAIL]`, `[N_COMMANDE]`…) plutôt que de les supprimer : le texte reste naturel.

`experiments/check_annotations.py` détecte les motifs les plus courants (e-mails, téléphones, IBAN, cartes, numéros de sécurité sociale) et refuse le fichier s'il en trouve. Cette détection automatique **ne remplace pas une relecture** : les noms propres, par exemple, ne sont pas détectés.

Vérifiez aussi la base légale du traitement (RGPD) : réutiliser des messages clients pour entraîner un modèle est un traitement à part entière, à inscrire au registre.

## 3. Annoter

Format : un fichier `.csv`, `.tsv` ou `.jsonl` avec les colonnes `text` et `label`. Étiquette vide ou `none` : hors-sujet.

```csv
text,label
"Toujours pas reçu mon colis, le suivi n'a pas bougé depuis mardi.",shipping
"Quels sont vos horaires d'ouverture le samedi ?",none
```

Bonnes pratiques :
1. **Un guide d'annotation écrit** : une définition par catégorie, deux ou trois exemples limites et la règle pour trancher (« un remboursement suite à un colis perdu : *shipping* »).
2. **Double annotation** d'au moins 100 messages par deux personnes, sans concertation. Mesurez l'accord :

   ```bash
   uv run python experiments/check_annotations.py annotateur_a.csv --second annotateur_b.csv \
       --taxonomy categories.json
   ```

   | Kappa de Cohen | Interprétation |
   |:-:|---|
   | ≥ 0,8 | Bon : les catégories sont claires. |
   | 0,6 – 0,8 | Moyen : précisez le guide sur les désaccords listés. |
   | < 0,6 | Faible : la taxonomie elle-même est ambiguë ; fusionnez ou redéfinissez des catégories. |

   Aucun modèle ne fera durablement mieux que l'accord entre humains.
3. **Arbitrez** les désaccords listés par le script, puis corrigez le guide.

## 4. Découper

Un même exemple ne doit servir qu'à **une** chose. Avec un seul fichier annoté :

```python
from krito import load_examples, split_examples

exemples = load_examples("messages_annotes.csv")
entrainement, reste = split_examples(exemples, test_size=0.4)   # fine-tuning, ou fit() du mode supervisé
calibration, test = split_examples(reste, test_size=0.5)       # seuils / juge, puis mesure finale
```

Le découpage est **stratifié** (chaque catégorie et les hors-sujet sont répartis proportionnellement) et reproductible (`seed`). Si les messages ont une date, préférez un découpage **temporel** (les plus récents en test) : il reproduit la production.

## 5. Utiliser

| Objectif | Outil | Données |
|---|---|---|
| Régler les garde-fous | `engine.calibrate(calibration, options)` | calibration |
| Apprendre un juge de confiance | `engine.fit_judge(juge, options)` | un jeu distinct de la calibration |
| Taxonomie fixe | `KritoClassifier(...).fit(entrainement)` | entraînement |
| Spécialiser le modèle NLI | `experiments/finetune_custom.py` | entraînement |
| Mesurer | `evaluate_guardrails(résultats, étiquettes, **seuils)` | test, **une seule fois** |

Voir le [guide utilisateur](guide-utilisateur.md#6-régler-les-garde-fous) et [entrainement.md](entrainement.md).

## 6. Publier un modèle

1. Contrôlez le jeu d'entraînement : `check_annotations.py` ne doit signaler **aucune erreur**.
2. Fine-tunez et exportez (`finetune_custom.py`), puis mesurez sur le jeu de test.
3. Décrivez les données dans un fichier Markdown : origine (réelles, synthétiques ou les deux), période, volume par catégorie, anonymisation, accord inter-annotateurs, licence.
4. Préparez la publication sans rien envoyer, relisez la fiche générée :

   ```bash
   uv run python experiments/publish_hub.py modeles/krito-maboite \
       --repo mon-organisation/krito-maboite --data-card donnees.md --metrics resultats.json --dry-run
   ```

5. Publiez (jeton via `hf auth login` ou `HF_TOKEN`) en relançant sans `--dry-run`. Ajoutez `--private` pour un modèle interne.

Le script n'envoie que les fichiers utiles à `KritoEngine.from_onnx` (meilleure variante du modèle, `tokenizer.json`, `labels.json`) et une fiche de modèle complète (usage, données, résultats, limites, licence).

> **Ne publiez jamais** un modèle entraîné sur des données non anonymisées : un modèle peut restituer des fragments de ses données d'entraînement.
