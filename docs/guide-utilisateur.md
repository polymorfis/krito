# Guide utilisateur

Ce guide explique comment utiliser Krito au quotidien : installer, prendre une première décision, bien écrire ses options, régler les garde-fous, traiter des lots, poser des questions oui/non ou sur une échelle, et apprendre une taxonomie fixe. Pour le fonctionnement interne, voir [architecture.md](architecture.md) ; pour la mise en production, voir [integration.md](integration.md).

## 1. À quoi sert Krito

Krito prend une **décision fermée** sur un texte : parmi les options que vous fournissez, laquelle correspond ? Il renvoie aussi **à quel point il en est sûr**.

Exemples d'usages :
- router un e-mail vers le bon service ;
- trier des tickets de support ;
- classer des avis clients par sujet ;
- détecter un e-mail d'hameçonnage ;
- étiqueter des documents.

Il ne rédige pas de réponse et n'invente jamais une catégorie : la réponse est toujours l'une de vos clés.

Krito tourne **sur votre infrastructure, sur CPU**, sans GPU ni appel à un service externe. Vos textes ne sortent pas de chez vous.

## 2. Installation

Python 3.11 ou plus récent.

```bash
# production : CPU, sans PyTorch (≈ 154 Mo de dépendances)
pip install "krito[onnx] @ git+https://github.com/polymorfis/krito"
# développement : PyTorch, GPU possible (plusieurs Go)
pip install "krito[torch] @ git+https://github.com/polymorfis/krito"
# en plus, pour le juge de confiance et le mode supervisé (scikit-learn, ≈ 160 Mo)
pip install "krito[onnx,learn] @ git+https://github.com/polymorfis/krito"
```

Une fois Krito publié sur PyPI, `pip install "krito[onnx]"` suffira.

Depuis les sources :

```bash
git clone https://github.com/polymorfis/krito && cd krito
uv sync --all-groups
```

### Obtenir un modèle

| Modèle | Usage | Comment l'obtenir |
|---|---|---|
| `krito-nli-fr-multi` (**recommandé**) | Français, 12 domaines, ONNX int8 optimisé (404 Mo). | Hugging Face Hub `polymorfis/krito-nli-fr-multi` *(publication prévue)*, ou reproduction : voir [entrainement.md](entrainement.md). |
| `krito-nli-fr-mdeberta` | Français, spécialisé support client (étude 1). | Reproduction : `experiments/export_onnx.py --model mdeberta`. |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | Multilingue, sans fine-tuning ; défaut du backend PyTorch. | Téléchargé automatiquement avec `krito[torch]`. |

Le modèle par défaut du backend PyTorch n'est **pas** spécialisé : il obtient environ 58 % de précision moyenne sur nos 12 domaines français, contre 95 % pour `krito-nli-fr-multi`. Utilisez un modèle fine-tuné dès que possible.

## 3. Première décision

```python
from krito import KritoEngine

engine = KritoEngine.from_onnx("polymorfis/krito-nli-fr-multi",   # ou un dossier local
                               threads=2,
                               hypothesis_template="Ce texte concerne {}.")

result = engine.classify(
    "Bonjour, depuis la mise à jour de ce matin l'application plante dès que j'ouvre mes factures.",
    {
        "facturation": "un problème de facturation ou de paiement",
        "livraison": "la livraison ou le suivi d'un colis",
        "compte": "la connexion ou l'accès au compte",
        "bug": "une panne technique du site ou de l'application",
    },
    min_entailment=0.5,
)

print(result.selected_key)   # 'bug'
print(result.confidence)     # 0.774
print(result.accepted)       # True
```

Créez le moteur **une seule fois** (le chargement prend environ 2 s), puis réutilisez-le pour toutes les décisions.

## 4. Bien écrire ses options

C'est le réglage qui a **le plus d'effet** sur la qualité, avant tout paramètre.

### Des descriptions concrètes, pas des étiquettes

Dans [l'exemple de routage](../examples/03_multi_domaines.py), des intitulés vagues ont donné **2 bons routages sur 4** ; des descriptions concrètes, **4 sur 4**, avec le même modèle et les mêmes messages.

| À éviter | À préférer |
|---|---|
| `"informatique"` | `"l'informatique ou la cybersécurité : e-mail suspect, arnaque, mot de passe, virus, réseau"` |
| `"rh"` | `"les ressources humaines : congés, RTT, salaire, fiche de paie, formation"` |
| `"compta"` | `"la comptabilité : factures clients ou fournisseurs, paiements, relances, TVA"` |

### Un gabarit d'hypothèse

La description est insérée dans une phrase via `hypothesis_template`. Le modèle `krito-nli-fr-multi` a été entraîné avec des gabarits de cette forme : utilisez `"Ce texte concerne {}."` ou `"Ce message concerne {}."`. Rédigez donc vos descriptions pour qu'elles complètent naturellement la phrase : « Ce texte concerne *la livraison d'un colis* ».

### Des options qui ne se recouvrent pas

Si deux options peuvent décrire le même message (« remboursement » et « facturation »), le modèle hésitera, et il aura raison. Fusionnez-les, ou précisez la frontière dans les descriptions.

### Combien d'options ?

Il n'y a pas de limite technique, mais le temps de calcul est proportionnel au nombre d'options : chaque option est une paire à évaluer. Au-delà d'une dizaine, préférez un **routage en deux étapes** (domaine, puis catégorie), comme dans l'exemple 03.

## 5. Lire le résultat

`classify` renvoie un `DecisionResult` immuable :

| Champ | Signification |
|---|---|
| `selected_key` / `selected_label` | Option retenue (clé et description). |
| `confidence` | Probabilité **relative** de l'option retenue face aux autres options (la somme vaut 1). |
| `margin` | Écart de probabilité avec la deuxième option. Une marge faible signale une hésitation entre deux options. |
| `accepted` | `True` si tous les garde-fous fournis sont satisfaits. |
| `rejection_reason` | Motif(s) de rejet : `confidence_too_low`, `margin_too_low`, `entailment_too_low`, `judge_score_too_low`. |
| `scores[clé]` | Détail par option : logits bruts `entailment`, `contradiction`, `neutral`, `decision_score`, `probability`, `entailment_prob`. |
| `judge_score` | Probabilité que la décision soit bonne selon le juge de confiance (`None` sans juge, voir § 6). |

### Relatif ou absolu ?

- `confidence` et `probability` sont **relatives** : elles comparent vos options entre elles. Même si aucune ne convient, l'une d'elles aura une forte probabilité.
- `entailment_prob` est **absolue** : c'est la probabilité que le texte implique l'option en elle-même. C'est elle qui baisse quand le message est hors-sujet.

## 6. Régler les garde-fous

```python
result = engine.classify(texte, options, threshold=0.6, min_margin=0.2, min_entailment=0.5)
if result.accepted:
    traiter_automatiquement(result.selected_key)
else:
    envoyer_en_revue_humaine(texte, result)
```

| Garde-fou | Protège contre | Point de départ |
|---|---|---|
| `min_entailment` | Les messages hors-sujet (aucune option ne convient). | 0,5 |
| `min_margin` | L'hésitation entre deux options proches. | 0,2 |
| `threshold` | Une confiance relative trop faible. | 0,6 |

Ces valeurs ne sont **qu'un point de départ**. L'échelle des scores dépend du modèle et du gabarit : le même seuil n'a pas le même effet d'un modèle à l'autre. Il faut les **calibrer sur vos données**.

### Calibrer sur vos données

1. Annotez 200 à 500 vrais messages (texte + bonne catégorie, `none` pour les hors-sujet). Protocole complet : [donnees-reelles.md](donnees-reelles.md).
2. Calibrez sur une moitié, vérifiez sur l'autre :

```python
from krito import evaluate_guardrails, load_examples, split_examples

calibration, validation = split_examples(load_examples("mes_messages_annotes.csv"), test_size=0.5)

cal = engine.calibrate(calibration, options, target_precision=0.90)
print(cal)                 # compromis précision / automatisation, seuils recommandés marqués ◄
engine.classify(texte, options, **cal.guardrails)    # seuils prêts à l'emploi

resultats = engine.classify_batch([ex.text for ex in validation], options)
print(evaluate_guardrails(resultats, [ex.label for ex in validation], **cal.guardrails))
```

`calibrate` cherche la combinaison de seuils (par défaut `min_entailment` et `min_margin` ; au choix parmi `threshold`, `min_margin`, `min_entailment`, `min_judge_score`) qui **automatise le plus de bonnes décisions** en respectant la précision visée parmi les décisions acceptées, hors-sujet acceptés comptés comme erreurs. Le résultat (`Calibration`) donne `precision`, `coverage` (part des messages acceptés), `automated` (part des messages du domaine traités automatiquement et correctement) et la liste des compromis possibles (`frontier`).

Ordre de grandeur : sur 290 tickets de support que le modèle n'avait jamais vus, une recherche de ce type (version 0.2 de l'exemple, sur le jeu complet) donnait **90 % de précision en traitant 74 % des messages automatiquement**, le reste partant en revue humaine. En ligne de commande : [l'exemple 06](../examples/06_calibration_seuil.py).

Si la précision visée est inatteignable, `cal.reached` vaut `False` et les seuils renvoyés donnent la meilleure précision possible. Dans ce cas, ajoutez un juge (ci-dessous) ou fine-tunez le modèle sur votre domaine (voir [entrainement.md](entrainement.md)).

### Juge de confiance (optionnel)

Au lieu de régler plusieurs seuils, un **juge** apprend sur vos données la probabilité qu'une décision soit bonne, à partir de tous ses signaux (confiance, marge, implication absolue, entropie…). Il demande `krito[learn]` (scikit-learn), mais aucun second modèle.

```python
from krito import ConfidenceJudge

juge_set, reste = split_examples(load_examples("mes_messages_annotes.csv"), test_size=0.6)
calibration, validation = split_examples(reste, test_size=0.5)

engine.fit_judge(juge_set, options)          # entraîne et attache le juge
cal = engine.calibrate(calibration, options, target_precision=0.90)   # règle min_judge_score
engine.judge.save("juge.json")                # quelques centaines d'octets, sans pickle

# en production
engine = KritoEngine.from_onnx(MODELE, judge=ConfidenceJudge.load("juge.json"), hypothesis_template=...)
r = engine.classify(texte, options, min_judge_score=0.8)
```

Le juge doit voir des **erreurs** et des **hors-sujet** pendant son entraînement, sinon il n'a rien à apprendre (erreur explicite). Entraînez-le, calibrez son seuil et mesurez sur trois jeux distincts ([exemple 09](../examples/09_juge_confiance.py)). Un juge est propre à un modèle, un gabarit et un jeu d'options : refaites-le si l'un d'eux change.

## 7. Traiter beaucoup de textes : les lots

Toutes les primitives existent en version par lots : `classify_batch`, `yes_no_batch`, `scale_batch`. Elles renvoient une liste de résultats **dans l'ordre reçu**, identiques à des appels un par un, mais regroupent les paires (texte, option) en appels au modèle de `batch_size` paires, en rapprochant les textes de longueurs voisines pour limiter le remplissage.

```python
resultats = engine.classify_batch(messages, options, min_entailment=0.5, batch_size=32)
auto = [(m, r.selected_key) for m, r in zip(messages, resultats) if r.accepted]
```

Sur CPU, `batch_size` entre 16 et 64 est un bon compromis ; plus grand consomme plus de mémoire sans gain notable. Sur GPU, augmentez-le.

## 8. Questions oui/non et échelles

### Oui / non

```python
r = engine.yes_no("Je veux être remboursé, le produit est arrivé cassé.",
                  "Le client demande un remboursement.", max_neutral=0.8)
r.answer        # True
r.probability   # P(oui)
r.neutral_prob  # élevée si le texte ne dit rien sur l'affirmation
```

- Formulez la question comme une **affirmation complète** : elle est utilisée telle quelle (le `hypothesis_template` du moteur ne s'applique pas).
- `probability` compare implication et contradiction. Si le texte ne parle pas du sujet, les deux sont faibles et la réponse est arbitraire : c'est ce que détecte `max_neutral`.
- Garde-fous : `threshold` (confiance minimale dans la réponse, oui ou non) et `max_neutral`.
- **Plusieurs étiquettes par message** : posez une affirmation par étiquette ; chaque réponse est indépendante.

### Échelle ordonnée

```python
URGENCE = ["pas urgente du tout", "peu urgente", "urgente", "extrêmement urgente"]
r = engine.scale(message, URGENCE, template="Cette demande est {}.")
r.selected_key  # niveau le plus probable
r.index         # son rang (0 = premier niveau)
r.expected      # rang moyen pondéré par les probabilités, ex. 2,4
r.spread        # écart-type en niveaux
```

Les niveaux se donnent **dans l'ordre**, du plus bas au plus haut (liste, ou dict `{clé: description}`). `template` remplace le gabarit du moteur pour cette échelle. Le résultat (`ScaleResult`) est un `DecisionResult` complet, avec en plus `index`, `expected` et `spread`.

Sur une échelle, hésiter entre deux niveaux voisins n'est pas grave : `expected` en tient compte (utile pour trier par urgence), et le garde-fou `max_spread` ne rejette que les distributions vraiment étalées (hésitation entre « pas urgent » et « très urgent »).

## 9. Taxonomie fixe : le mode supervisé

Si vos catégories ne changent pas et que vous avez au moins une trentaine d'exemples par catégorie, `KritoClassifier` apprend directement la taxonomie : des embeddings (`multilingual-e5-base` par défaut) et une régression logistique scikit-learn. Il est plus rapide que le zero-shot (un seul passage du texte, quel que soit le nombre de catégories) et souvent plus précis sur votre domaine (95 % sur le support dans l'[étude 1](../experiments/RESULTS.md)).

```python
from krito import KritoClassifier, OnnxEmbedder

clf = KritoClassifier(OnnxEmbedder("modeles/e5-base")).fit(entrainement)   # ou KritoClassifier() avec krito[torch]
r = clf.classify("Mon colis n'est jamais arrivé.", min_judge_score=0.8)
clf.save("classifieur.npz")
clf = KritoClassifier.load("classifieur.npz")      # recrée l'embedder à partir de sa référence
```

- `fit` accepte des couples `(texte, catégorie)`, des dicts ou `{catégorie: [textes]}`. Les hors-sujet (`None` / `"none"`) ne deviennent pas une catégorie : ils apprennent au juge à les rejeter.
- Un **juge** est entraîné automatiquement par validation croisée (`judge=False` pour s'en passer).
- Garde-fous : `threshold`, `min_margin`, `min_similarity` (similarité au plus proche exemple d'entraînement de la catégorie retenue, qui rejette les hors-sujet), `min_judge_score`. `clf.calibrate(exemples)` les règle, sur des exemples **distincts** de ceux de `fit`.
- Sans PyTorch : exportez le modèle d'embeddings une fois avec `experiments/export_embedder_onnx.py`.
- Pour ajouter une catégorie, il faut ré-entraîner (quelques secondes) ; c'est la contrepartie par rapport au zero-shot.

Exemple complet : [08_mode_supervise.py](../examples/08_mode_supervise.py).

## 10. Performances à attendre

Sur un processeur de 2013 (Intel i7-4770K), avec le modèle `krito-nli-fr-multi` et 6 options :

| Threads | Latence médiane | p95 | RAM max |
|:-:|:-:|:-:|:-:|
| 1 | 448 ms | 708 ms | 710 Mo |
| 2 | 246 ms | 380 ms | 710 Mo |
| 4 | 154 ms | 236 ms | 708 Mo |

La latence est à peu près proportionnelle au nombre d'options. Détails et méthodologie : [performances.md](performances.md).

## 11. Limites à connaître

- **Français d'abord.** Les modèles fine-tunés sont entraînés en français. Le modèle de base est multilingue, mais nous n'avons pas mesuré les autres langues.
- **Textes courts à moyens.** Au-delà de 512 tokens (environ 350 mots), la fin du texte est ignorée, avec un avertissement (`UserWarning`). Pour un long document, classez le passage pertinent (objet de l'e-mail et premier paragraphe, par exemple).
- **Un seul choix par `classify`.** Pour un message qui porte sur plusieurs sujets, posez une question `yes_no` par sujet (§ 8).
- **Données synthétiques.** Les modèles et les chiffres publiés proviennent de données écrites par des LLM. Les ordres de grandeur sont fiables, mais les chiffres absolus sont probablement optimistes. Validez sur vos propres messages.
- **La confiance n'est pas une garantie.** Même acceptée, une décision peut être fausse (exemple réel : « code promo non appliqué » classé *commercial* au lieu de *facturation*). Gardez une revue humaine sur une partie du flux et suivez le taux d'erreur.

## 12. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `FileNotFoundError: Aucun modèle ONNX` | Dossier de modèle incomplet. | Il faut `model.opt.onnx` (ou `model.int8.onnx` ou `model.onnx`), `tokenizer.json` et `labels.json`. |
| `ValueError: Le modèle doit être un NLI à 3 classes` | Modèle à 2 classes (`entailment` / `not_entailment`) ou non NLI. | Utilisez un modèle NLI à 3 classes. |
| `UserWarning: … tronquées` | Texte trop long. | Raccourcissez le texte, ou classez-en un extrait. |
| Tous les messages sont rejetés | Seuil `min_entailment` trop haut pour ce modèle ou ce gabarit. | Calibrez (§ 6), ou vérifiez que le gabarit ressemble à ceux de l'entraînement. |
| `ImportError: … krito[learn]` | Juge ou mode supervisé sans scikit-learn. | `pip install "krito[learn]"`. |
| `ValueError: Le juge a besoin de bonnes ET de mauvaises décisions` | Exemples trop faciles pour entraîner un juge. | Ajoutez des hors-sujet et des cas limites. |
| Beaucoup d'hésitations entre deux options | Options qui se recouvrent. | Précisez les descriptions ou fusionnez les options. |
| Latence élevée | Trop d'options, ou un seul thread. | `threads=2` à `4` ; routage en deux étapes ; `classify_batch` pour les traitements en masse ; mode supervisé si la taxonomie est fixe. |
| RAM plus élevée que prévu | Chargement de `model.int8.onnx` au lieu de `model.opt.onnx`. | Générez `model.opt.onnx` (voir [entrainement.md](entrainement.md)). |
