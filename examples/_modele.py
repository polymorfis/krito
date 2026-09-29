"""Choix du modèle utilisé par les exemples.

Par ordre de priorité :
1. la variable d'environnement KRITO_MODEL (dossier local ou identifiant Hugging Face Hub) ;
2. le modèle multi-domaines produit par les expériences du dépôt (experiments/models) ;
3. le modèle publié sur le Hugging Face Hub.
"""

import os
from pathlib import Path

LOCAL = Path(__file__).resolve().parents[1] / "experiments" / "models" / "krito-nli-fr-multi"
HUB = "polymorfis/krito-nli-fr-multi"

MODEL = os.environ.get("KRITO_MODEL") or (str(LOCAL) if LOCAL.exists() else HUB)
TEMPLATE = "Ce texte concerne {}."


def embedder():
    """Modèle d'embeddings du mode supervisé (exemple 08).

    KRITO_EMBEDDER : dossier ONNX produit par experiments/export_embedder_onnx.py (sans PyTorch) ;
    sinon, ``intfloat/multilingual-e5-base`` via sentence-transformers (``krito[torch]``).
    """
    from krito import OnnxEmbedder, SentenceTransformerEmbedder

    path = os.environ.get("KRITO_EMBEDDER")
    return OnnxEmbedder(path, threads=4) if path else SentenceTransformerEmbedder()
