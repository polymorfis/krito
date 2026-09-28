# Exemples

Tous les exemples tournent **sur CPU**, sans GPU ni PyTorch, avec `krito[onnx]`.

| Exemple | Ce qu'il montre |
|---|---|
| [01_premiers_pas.py](01_premiers_pas.py) | Classer un message parmi des options décrites en langage naturel. |
| [02_garde_fous.py](02_garde_fous.py) | Automatiser ce qui est sûr, envoyer le reste en revue humaine. |
| [03_multi_domaines.py](03_multi_domaines.py) | Router en deux étapes (service puis catégorie), sans entraînement. |
| [04_api_fastapi.py](04_api_fastapi.py) | Exposer Krito en API HTTP avec documentation interactive. |
| [05_traitement_csv.py](05_traitement_csv.py) | Traiter un fichier CSV et marquer ce qui part en revue humaine. |
| [06_calibration_seuil.py](06_calibration_seuil.py) | Calibrer les garde-fous sur vos données annotées. |
| [docker_api/](docker_api/Dockerfile) | Image Docker de l'API, testée avec 1 Go de RAM et 2 CPU. |
| [serverless_lambda/](serverless_lambda/) | Fonction AWS Lambda en image conteneur, testée avec l'émulateur officiel. |

## Choix du modèle

Les exemples cherchent le modèle ([_modele.py](_modele.py)) dans cet ordre :

1. la variable `KRITO_MODEL` : un dossier local, ou un identifiant du Hugging Face Hub ;
2. `experiments/models/krito-nli-fr-multi`, produit par les expériences du dépôt ;
3. `polymorfis/krito-nli-fr-multi` sur le Hugging Face Hub.

```bash
uv sync --all-groups
uv run python examples/01_premiers_pas.py
KRITO_MODEL=/chemin/vers/modele uv run python examples/02_garde_fous.py
```

## Ce que donnent les exemples

Mesures sur un Intel i7-4770K (2013), modèle `krito-nli-fr-multi` en ONNX optimisé :

| Exemple | Résultat |
|---|---|
| 02 | 2 messages traités automatiquement, 2 hors-sujet envoyés en revue humaine. |
| 03 | 4 routages corrects sur 4 avec des descriptions concrètes ; seulement 2 sur 4 avec des intitulés vagues (« informatique », « rh »). |
| 05 | 20 messages en 3,6 s (182 ms par message) : 14 traités automatiquement, 6 en revue humaine. Une erreur passe en automatique (« code promo non appliqué » classé *commercial*). |
| 06 | Sur 290 tickets de support jamais vus à l'entraînement : 90 % de précision en traitant 74 % des messages automatiquement. 95 % n'est pas atteignable sans fine-tuning sur le domaine. |
| API Docker | Prête en 2,9 s ; environ 700 Mo de RAM sous une limite de 1 Go ; 160 à 280 ms par requête HTTP selon le nombre d'options (3 à 6). |
| Lambda | Démarrage à froid 1,8 s ; 167 ms par invocation à chaud ; 655 Mo de RAM. |

## API HTTP

```bash
uv run uvicorn --app-dir examples 04_api_fastapi:app --port 8000
# documentation interactive : http://localhost:8000/docs
curl -s -X POST localhost:8000/classify -H 'Content-Type: application/json' -d '{
  "context": "Mon colis est indiqué livré mais je ne l ai jamais reçu.",
  "options": {"facturation": "un problème de facturation", "livraison": "la livraison d un colis"},
  "min_entailment": 0.5
}'
```

## Docker

```bash
hf download polymorfis/krito-nli-fr-multi --local-dir modele     # ou copiez un export local
docker build -f examples/docker_api/Dockerfile -t krito-api .
docker run --rm -p 8000:8000 --memory=1g --cpus=2 krito-api
```

## AWS Lambda

```bash
docker build -f examples/serverless_lambda/Dockerfile -t krito-lambda .
docker run --rm -p 9000:8080 --memory=1g krito-lambda
curl -s localhost:9000/2015-03-31/functions/function/invocations \
     -d '{"context": "Mon colis est perdu", "options": {"a": "la livraison", "b": "la facturation"}}'
```

Poussez ensuite l'image sur Amazon ECR et créez la fonction à partir de cette image. Réglages conseillés : 1 024 Mo de mémoire et un délai de 30 s.
