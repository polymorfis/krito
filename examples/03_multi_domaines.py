"""Routage en deux étapes, sans aucun entraînement : d'abord le service, puis la catégorie.

Les options sont libres à chaque appel : on peut ajouter un service ou une catégorie
sans réentraîner le modèle.

    uv run python examples/03_multi_domaines.py
"""

from _modele import MODEL, TEMPLATE

from krito import KritoEngine

engine = KritoEngine.from_onnx(MODEL, threads=2, hypothesis_template=TEMPLATE)

# Des descriptions concrètes (exemples de sujets) donnent de bien meilleurs résultats
# qu'un intitulé vague comme « informatique ».
SERVICES = {
    "juridique": "une question juridique : contrat, droit du travail, statuts de société, litige ou procédure",
    "compta": "la comptabilité : factures clients ou fournisseurs, paiements, relances, TVA, trésorerie",
    "informatique": "l'informatique ou la cybersécurité : e-mail suspect, arnaque, mot de passe, virus, réseau",
    "rh": "les ressources humaines : congés, RTT, salaire, fiche de paie, formation, recrutement",
}
CATEGORIES = {
    "juridique": {"travail": "le droit du travail", "societes": "le droit des sociétés", "litige": "un litige commercial"},
    "compta": {"tva": "la TVA ou les impôts", "tresorerie": "la trésorerie ou un financement", "impayes": "des factures clients impayées"},
    "informatique": {"phishing": "un e-mail suspect ou une arnaque", "acces": "un mot de passe ou un accès bloqué", "reseau": "le réseau ou le VPN"},
    "rh": {"conges": "les congés ou les absences", "paie": "le salaire ou la fiche de paie", "formation": "une formation"},
}

emails = [
    "Le client Dupont ne nous a toujours pas réglé les trois factures de juin malgré nos relances.",
    "J'ai reçu un mail du « directeur » qui me demande d'acheter des cartes cadeaux en urgence.",
    "Un associé souhaite céder ses parts, faut-il l'accord des autres ?",
    "Il me reste combien de jours de RTT cette année ?",
]

for email in emails:
    service = engine.classify(email, SERVICES)
    categorie = engine.classify(email, CATEGORIES[service.selected_key])
    print(f"{service.selected_key:>12} / {categorie.selected_key:<11} ({categorie.confidence:.0%}) | {email}")
