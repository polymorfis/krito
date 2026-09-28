# Journal des modifications

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
