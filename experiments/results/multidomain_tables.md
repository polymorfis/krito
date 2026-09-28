### Précision sur les jeux gold (messages du domaine)

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

### AUROC hors-sujet sur les jeux gold

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

### Traités automatiquement à 90 % de précision (gold)

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

### Précision sur les textes générés du domaine exclu (~300 par domaine)

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

### Qualité du jeu généré

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
