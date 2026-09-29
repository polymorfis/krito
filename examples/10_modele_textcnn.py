"""Modèle léger TextCNN livré avec Krito : aucun téléchargement, quelques Mo, quelques ms par décision.

    uv run python examples/10_modele_textcnn.py

Moins précis que le modèle NLI fine-tuné (voir docs/modele-textcnn.md), mais utilisable
tout de suite, hors ligne, sur un très petit serveur. Au-delà de 10 options, le bi-encodeur
présélectionne les 5 plus proches et seules celles-ci passent par le cross-encodeur.
"""

import time

from krito import KritoEngine

engine = KritoEngine.from_textcnn(threads=1)  # gabarit d'hypothèse de l'entraînement appliqué

OPTIONS = {
    "billing": "un problème de facturation, de paiement ou de remboursement",
    "shipping": "la livraison, le suivi ou le retour d'un colis",
    "account": "la connexion, le mot de passe ou l'accès au compte",
    "bug": "un bug technique ou une panne du site ou de l'application",
}

for message in [
    "Le colis indiqué comme livré n'est jamais arrivé.",
    "Impossible de me connecter depuis que j'ai changé de mot de passe.",
    "J'ai été prélevé deux fois pour la même commande.",
]:
    start = time.perf_counter()
    result = engine.classify(message, OPTIONS, min_entailment=0.5)
    ms = (time.perf_counter() - start) * 1000
    verdict = result.selected_key if result.accepted else f"rejeté ({result.rejection_reason})"
    print(f"{ms:5.1f} ms  {verdict:40} {result.confidence:.0%}  {message}")
