"""Modèle Krito léger entraîné de zéro (TextCNN + bi-encodeur + cross-encodeur).

L'inférence (``TextCNNBackend``) ne dépend que d'ONNX Runtime et de tokenizers ;
l'entraînement et l'export (``krito.textcnn.train``, ``krito.textcnn.model``) demandent PyTorch.
"""

from .backend import DEFAULT_TEXTCNN_MODEL, TextCNNBackend

__all__ = ["DEFAULT_TEXTCNN_MODEL", "TextCNNBackend"]
