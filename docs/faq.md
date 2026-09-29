# FAQ

## Généralités

### Qu'est-ce que Krito, en une phrase ?
Un moteur open source qui prend des **décisions fermées** sur du texte (choisir une option parmi celles que vous fournissez) avec une **confiance chiffrée**. Il tourne **sur CPU**, dans environ 710 Mo de RAM, sans GPU ni service externe.

### À quoi ça sert concrètement ?
À toutes les décisions répétitives sur du texte dont les réponses possibles sont connues à l'avance :
- router des e-mails ou des tickets ;
- classer des avis clients ;
- repérer un e-mail d'hameçonnage ;
- trier des demandes internes (RH, informatique, juridique) ;
- étiqueter des documents ;
- filtrer ce qui mérite l'attention d'un humain.

### Qui est derrière Krito ?
[POLYMORFIS](https://www.polymorfis.com/), éditeur de logiciels et société de développement sur mesure. Krito est publié sous licence MIT.

### Est-ce gratuit, y compris pour un usage commercial ?
Oui. Le code est sous licence MIT : usage, modification et redistribution libres, y compris commerciaux, à condition de conserver la mention de copyright. Vérifiez aussi la licence du modèle que vous utilisez (voir « Licences » plus bas).

## Comparaisons

### Quelle différence avec un LLM (ChatGPT, Mistral, Llama…) ?
Un LLM *génère* du texte ; Krito *choisit* parmi vos options.

| | Krito | LLM local 27B (mesuré) |
|---|:-:|:-:|
| Précision moyenne sur nos 12 domaines | 95 % | 97 % |
| Matériel | CPU | GPU |
| Taille du modèle | 404 Mo | 17 Go |
| Latence | environ 0,15 à 0,45 s sur CPU | environ 3,8 s sur GPU |
| Réponse hors de vos options | impossible | possible sans contrainte de format |

Si vous avez besoin de rédiger, de résumer ou de raisonner, utilisez un LLM. Pour décider vite, souvent et à bas coût, Krito suffit.

### Et par rapport à Jev (TypeSafe) ?
Krito partage la même idée, popularisée par [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) : des décisions typées avec une confiance, plutôt que du texte libre. Krito n'est pas affilié à TypeSafe et nous ne l'avons pas comparé à Jev sur les mêmes données. Les différences de conception :
- Krito est **open source** et s'exécute **chez vous** ;
- Krito propose le choix d'une option, les questions oui/non et les échelles ordonnées, ainsi qu'un mode supervisé pour les taxonomies fixes ;
- ses modèles sont spécialisés en **français**.

### Pourquoi ne pas entraîner un classifieur classique (scikit-learn, TF-IDF) ?
Vous pouvez, et c'est parfois le bon choix. Nos mesures :
- TF-IDF + régression logistique plafonne à 82 % sur nos tickets de support ;
- des embeddings + régression logistique atteignent 94 à 95 %, mais seulement pour des **catégories fixées à l'avance** et entraînées.

Krito accepte de **nouvelles catégories à chaque appel**, sans réentraînement, avec une qualité comparable (95 % en moyenne sur 12 domaines). Et si votre taxonomie est fixe, Krito propose aussi ce mode : `KritoClassifier.fit(exemples)` (embeddings + régression logistique scikit-learn + juge de confiance), avec la même interface de résultat et de garde-fous.

## Utilisation

### Faut-il un GPU ?
Non. Tout est conçu et mesuré pour le CPU. Un GPU n'accélère que le fine-tuning.

### Combien de RAM faut-il ?
Environ **710 Mo** par processus pour l'utilisation. Un conteneur limité à 1 Go fonctionne (mesuré). Seule la **production** d'un nouveau modèle (export et quantification) demande environ 8 Go, une fois, hors ligne.

### Quelles plateformes sont supportées ?
Linux x86-64 est testé. Krito s'appuie sur ONNX Runtime et `tokenizers`, disponibles sous Windows, macOS et Linux ARM64, mais **nous ne l'avons pas testé** sur ces plateformes. Un petit serveur ARM ou une carte de type Raspberry Pi avec au moins 2 Go de RAM est plausible, mais non mesuré.

### Quelles langues ?
Les modèles fine-tunés sont entraînés et évalués en **français**. Le modèle de base est multilingue ; l'anglais et d'autres langues fonctionnent sans doute, mais la qualité n'a pas été mesurée pour les modèles fine-tunés.

### Combien d'options puis-je donner ?
Autant que nécessaire, mais le temps de calcul est **proportionnel** au nombre d'options. Au-delà d'une dizaine, faites un routage en deux étapes : d'abord le domaine, puis la catégorie (voir [l'exemple 03](../examples/03_multi_domaines.py)).

### Mes résultats sont moyens. Que faire ?
Par ordre d'efficacité :
1. **Réécrivez vos descriptions** : concrètes, avec des exemples de sujets. Dans notre exemple de routage, c'est passé de 2 bons routages sur 4 à 4 sur 4.
2. Vérifiez que vos options **ne se recouvrent pas**.
3. Utilisez le gabarit `"Ce texte concerne {}."`.
4. **Fine-tunez** sur quelques centaines de messages annotés (voir [entrainement.md](entrainement.md)). Sur 202 tickets, la précision est passée de 84,7 % à 91,7 %.

### Comment régler `min_entailment` ?
Sur **vos** données annotées, avec `engine.calibrate(exemples, options)` ([exemple 06](../examples/06_calibration_seuil.py)). L'échelle de la probabilité d'implication dépend du modèle et du gabarit : un seuil qui marche avec un modèle peut tout rejeter avec un autre. 0,5 est un point de départ, pas une vérité.

### La confiance est-elle une probabilité fiable ?
Pas au sens strict. `confidence` est **relative** à vos options : elle reste élevée même si aucune ne convient. `entailment_prob` est **absolue** et bien meilleure pour détecter les hors-sujet (AUROC de 95 à 96 % dans nos tests), mais sa valeur exacte dépend du modèle. Traitez-les comme des **scores à calibrer**, pas comme des pourcentages garantis. Pour un score plus fiable, entraînez un **juge de confiance** sur vos données (`engine.fit_judge`) : il estime directement la probabilité qu'une décision soit bonne.

### Un message peut-il avoir plusieurs catégories ?
Pas avec `classify`, qui choisit une option. Pour du multi-étiquette, posez une question `yes_no` par étiquette (« Le message parle de facturation. ») : chaque réponse est indépendante.

### Et les questions oui/non ?
`engine.yes_no(texte, "Le client demande un remboursement.")` renvoie `answer`, `probability` (P(oui)) et `neutral_prob`. Utilisez `max_neutral` pour ne pas répondre quand le texte ne parle pas du sujet. Pour une gradation (urgence, satisfaction), utilisez `engine.scale` avec des niveaux ordonnés.

### Que se passe-t-il avec un texte long ?
Au-delà de 512 tokens (environ 300 à 350 mots), la fin du texte est ignorée et Krito émet un `UserWarning`. Classez plutôt un extrait pertinent : l'objet et le premier paragraphe d'un e-mail, par exemple.

### Krito est-il utilisable depuis plusieurs threads ?
Oui. Un même moteur peut être appelé en parallèle (testé avec 8 threads : résultats identiques au séquentiel). Pour traiter un grand volume dans un seul processus, utilisez plutôt `classify_batch`.

## Données, sécurité, conformité

### Mes données quittent-elles mon infrastructure ?
Non. L'inférence est entièrement locale. Le seul accès réseau possible est le téléchargement initial du modèle depuis le Hugging Face Hub, que vous pouvez remplacer par un dossier local.

### Krito est-il conforme au RGPD ?
Krito ne stocke rien et n'envoie rien : il facilite la conformité, mais c'est **votre traitement** qui doit l'être (base légale, durée de conservation des textes que vous journalisez, information des personnes).

### Et l'AI Act européen ?
Les obligations dépendent de **l'usage**, pas de l'outil. Classer des tickets de support n'est pas un usage à haut risque. En revanche, trier des candidatures (recrutement) en est un, avec des obligations fortes de documentation, de supervision humaine et de gestion des risques. Krito aide sur deux points : les garde-fous permettent la supervision humaine, et chaque décision est traçable (option, confiance, motif de rejet). La conformité de votre système reste à établir au cas par cas.

### Sur quelles données les modèles ont-ils été entraînés ?
- Le modèle de base, `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, est public et sous licence MIT.
- Nos modèles fine-tunés l'ont été sur des **textes synthétiques** : 4 053 textes générés par `gemma4:26b` et vérifiés par `qwen3.8:27b` (étude 2), et 290 tickets écrits par Claude (étude 1).
- Aucune donnée client ni personnelle réelle n'a été utilisée.
- Tout est versionné dans `experiments/data/`.

### Licences

| Élément | Licence |
|---|---|
| Code de Krito | MIT |
| Modèle de base mDeBERTa-v3-base-xnli | MIT |
| Données d'entraînement générées | Soumises aux conditions d'utilisation des modèles générateurs (Gemma, Qwen) : à vérifier avant tout usage de redistribution. |

## Projet

### Comment contribuer ?
Voir [CONTRIBUTING.md](../CONTRIBUTING.md). Les contributions les plus utiles aujourd'hui :
- des **jeux de test réels anonymisés**, dans de nouveaux domaines (protocole : [donnees-reelles.md](donnees-reelles.md)) ;
- des mesures sur d'autres processeurs (ARM, CPU récents) ;
- des traductions ;
- des mesures des primitives oui/non et échelle sur des données annotées.

### Où signaler un bug ?
Dans les *issues* GitHub : <https://github.com/polymorfis/krito/issues>. Pour une faille de sécurité, suivez [SECURITY.md](../SECURITY.md).
