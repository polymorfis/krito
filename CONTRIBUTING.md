# Contribuer à Krito

Merci de votre intérêt ! Krito vise une IA de décision **souveraine, sobre et déployable partout** : sur CPU, avec peu de mémoire, sans dépendre d'un service externe. Les contributions qui vont dans ce sens sont les bienvenues.

## Contributions les plus utiles

- **Jeux de test réels anonymisés** dans de nouveaux domaines (c'est ce qui manque le plus : nos données sont synthétiques).
- **Mesures sur d'autres matériels** : CPU récents, ARM, petites machines. Utilisez `benchmarks/bench_cpu.py`.
- **Réduction de la mémoire** : élagage du vocabulaire, distillation, autres runtimes.
- **Nouvelles primitives** : questions oui/non, échelles ordonnées.
- Documentation, traductions, exemples d'intégration.

## Mettre en place l'environnement

```bash
git clone https://github.com/polymorfis/krito && cd krito
uv sync --all-groups
uv run pytest -m "not integration"   # tests unitaires, rapides, sans modèle
uv run pytest                        # tous les tests (modèles nécessaires)
```

## Règles

- **Mesurer plutôt qu'affirmer.** Toute amélioration de qualité ou de performance s'accompagne d'une mesure reproductible (script et données).
- **Tests** : toute correction de bug s'accompagne d'un test qui échoue avant et passe après.
- **Style** : suivez le code existant (Python ≥ 3.11, annotations de type, messages d'erreur en français).
- **Données** : n'ajoutez jamais de données personnelles ni de données clients non anonymisées.
- **Commits** : messages courts et explicites ; une PR par sujet.

## Signaler un bug

Ouvrez une *issue* avec :
- la version de Krito ;
- le modèle utilisé ;
- un exemple minimal reproductible ;
- le résultat obtenu et le résultat attendu.

Pour une faille de sécurité, voir [SECURITY.md](SECURITY.md).
