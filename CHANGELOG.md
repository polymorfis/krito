# Journal des modifications

## 0.3.0 — non publiée

### Ajouts
- **Traitement par lots** : `classify_batch`, `yes_no_batch`, `scale_batch`. Paires regroupées en appels de `batch_size`, textes de longueurs voisines rapprochés ; résultats identiques aux appels unitaires, dans l'ordre reçu.
- **Primitive oui/non** `yes_no(contexte, affirmation)` → `YesNoResult` (`answer`, `probability`, `neutral_prob`), garde-fous `threshold` et `max_neutral`.
- **Échelle ordonnée** `scale(contexte, niveaux)` → `ScaleResult` (niveau, `index`, rang moyen `expected`, `spread`), gabarit propre à l'échelle, garde-fou `max_spread`.
- **Calibration** des garde-fous sur des exemples annotés : `engine.calibrate(...)` → `Calibration` (seuils recommandés pour une précision cible, compromis précision / automatisation) ; `calibrate` et `evaluate_guardrails` sur des résultats déjà calculés.
- **Juge de confiance** optionnel `ConfidenceJudge` (scikit-learn) : `engine.fit_judge(...)`, `judge_score` dans chaque décision, garde-fou `min_judge_score`, sauvegarde JSON sans pickle.
- **Mode supervisé** `KritoClassifier(embedder).fit(exemples)` : embeddings + régression logistique scikit-learn, juge appris hors pli, garde-fou `min_similarity`, `save` / `load`. Embedders `OnnxEmbedder` (sans PyTorch) et `SentenceTransformerEmbedder` ; export ONNX : `experiments/export_embedder_onnx.py`.
- **Exemples annotés** : `load_examples` (CSV, TSV, JSONL), `split_examples` (découpage stratifié), `Example`.
- **Données réelles** : protocole ([docs/donnees-reelles.md](docs/donnees-reelles.md)), contrôle qualité et accord inter-annotateurs (`experiments/check_annotations.py`), publication sur le Hugging Face Hub avec fiche de modèle (`experiments/publish_hub.py`).
- API FastAPI : `/classify_batch`, `/yes_no`, `/scale`. Exemples 07 (oui/non, échelle), 08 (mode supervisé), 09 (juge) ; 05 et 06 utilisent les lots et `calibrate`.
- Extra **`krito[learn]`** (scikit-learn), importé seulement à l'usage : le zero-shot reste à 154 Mo de dépendances.

- **Modèle léger TextCNN** `krito-textcnn-fr`, entraîné de zéro (sans poids pré-entraîné) et livré dans le paquet : `KritoEngine.from_textcnn()`. Backbone TextCNN partagé, bi-encodeur (présélection des `top_k` options au-delà de `threshold`) et cross-encodeur à alignement par attention produisant les 3 logits NLI. Backend ONNX Runtime sans PyTorch (`krito.textcnn.TextCNNBackend`), entraînement et export sur CPU (`krito-textcnn`, `krito.textcnn.train`), étude `experiments/run_textcnn.py`, exemple 10, [docs/modele-textcnn.md](docs/modele-textcnn.md).

### Changements
- Les lots ne répartissent plus les options d'un même contexte sur plusieurs appels pour les backends qui le demandent (`needs_whole_contexts`), comme le routage du TextCNN.
- `psutil` n'est plus une dépendance ; le script `encoder` (module inexistant) est remplacé par `krito-textcnn`.
- `DecisionResult` gagne le champ `judge_score` (`None` sans juge) ; les résultats sont définis dans `krito.results`.

## 0.2.0 — 2026-09-28

### Ajouts
- Backend **ONNX Runtime** sans PyTorch (`KritoEngine.from_onnx`), chargement depuis un dossier ou le Hugging Face Hub.
- Garde-fou **`min_entailment`** (implication absolue) pour rejeter les messages hors-sujet ; `entailment_prob` par option.
- **`hypothesis_template`** pour formuler les options en phrases.
- Modèle **multi-domaines** `krito-nli-fr-multi` (12 domaines, français) : 95 % de précision moyenne, 88 % sur un domaine jamais vu.
- Export ONNX avec **quantification int8 sélective** et **optimisation hors ligne** : 404 Mo, 710 Mo de RAM.
- Script de **fine-tuning sur vos données** (`experiments/finetune_custom.py`).
- Exemples : API FastAPI, Docker, AWS Lambda, traitement CSV, calibration des seuils.
- Documentation complète, landing page, études et données reproductibles.

### Corrections
- Les probabilités ne sont plus arrondies avant le choix de l'option.
- Erreur explicite pour les modèles NLI à 2 classes ou aux étiquettes inconnues.
- Avertissement en cas de texte tronqué.
- Backend ONNX sûr en multithread.

### Changements incompatibles
- Modèle par défaut du backend PyTorch : `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (multilingue) au lieu de `nli-deberta-v3-xsmall` (anglais).
- PyTorch devient optionnel : `krito[torch]` ou `krito[onnx]`.
- `OptionScore` gagne le champ `entailment_prob` ; les scores ne sont plus arrondis.

## 0.1.0

- Première version : `KritoEngine` avec un cross-encoder NLI PyTorch, garde-fous `threshold` et `min_margin`.
