"""Jeu de données bilingue (FR/EN) de tickets de support pour évaluer Krito.

Chaque exemple existe en français et en anglais (traduction fidèle) afin de
mesurer l'effet de la langue indépendamment du contenu.
``expected = None`` signifie qu'aucune option ne convient (hors-sujet) :
un bon moteur devrait rejeter ces cas.
"""

LABELS = {
    "fr": {
        "billing": "Problème de facturation ou de paiement",
        "shipping": "Problème de livraison ou de suivi de colis",
        "account": "Problème de connexion ou d'accès au compte",
        "cancel": "Demande de résiliation ou d'annulation d'abonnement",
        "bug": "Bug technique ou panne du service",
        "sales": "Question sur les tarifs ou les offres commerciales",
    },
    "en": {
        "billing": "Billing or payment problem",
        "shipping": "Delivery or parcel tracking problem",
        "account": "Login or account access problem",
        "cancel": "Subscription cancellation request",
        "bug": "Technical bug or service outage",
        "sales": "Question about pricing or commercial offers",
    },
}

TEMPLATES = {
    "fr": "Ce message concerne : {}.",
    "en": "This message is about: {}.",
}

# (expected, fr, en)
SAMPLES: list[tuple[str | None, str, str]] = [
    # billing
    ("billing", "J'ai été débité deux fois pour la même commande.", "I was charged twice for the same order."),
    ("billing", "Ma facture de ce mois-ci est plus élevée que prévu, pourquoi ?", "This month's invoice is higher than expected, why?"),
    ("billing", "Ma carte bancaire est refusée au moment de payer.", "My credit card is declined at checkout."),
    ("billing", "Je ne trouve pas ma facture PDF pour ma comptabilité.", "I can't find my PDF invoice for my accounting."),
    ("billing", "Le prélèvement SEPA a échoué et je reçois des relances.", "The direct debit failed and I keep getting reminders."),
    ("billing", "Vous m'avez facturé une option que je n'ai jamais demandée.", "You billed me for an add-on I never asked for."),
    # shipping
    ("shipping", "Le colis indiqué comme livré n'a jamais été déposé dans ma boîte aux lettres.", "The parcel marked as delivered was never put in my mailbox."),
    ("shipping", "Ma commande est bloquée en transit depuis dix jours.", "My order has been stuck in transit for ten days."),
    ("shipping", "Le numéro de suivi ne fonctionne pas sur le site du transporteur.", "The tracking number doesn't work on the carrier's website."),
    ("shipping", "J'ai reçu un carton abîmé avec la moitié des articles.", "I received a damaged box with half of the items."),
    ("shipping", "Le livreur n'est jamais passé alors que j'étais chez moi.", "The courier never came even though I was at home."),
    ("shipping", "Pouvez-vous changer l'adresse de livraison de ma commande en cours ?", "Can you change the delivery address of my current order?"),
    # account
    ("account", "J'ai oublié mon mot de passe et le lien de réinitialisation n'arrive pas.", "I forgot my password and the reset link never arrives."),
    ("account", "Mon compte est bloqué après trop de tentatives.", "My account is locked after too many attempts."),
    ("account", "Je ne reçois pas le code de double authentification par SMS.", "I don't receive the two-factor authentication SMS code."),
    ("account", "Quelqu'un s'est connecté à mon compte depuis un autre pays.", "Someone logged into my account from another country."),
    ("account", "Je n'arrive plus à me connecter depuis que j'ai changé d'adresse e-mail.", "I can't log in anymore since I changed my email address."),
    ("account", "L'authentification via Google me renvoie une erreur.", "Sign in with Google returns an error."),
    # cancel
    ("cancel", "Je veux résilier mon abonnement à la fin du mois.", "I want to cancel my subscription at the end of the month."),
    ("cancel", "Comment arrêter le renouvellement automatique ?", "How do I stop the automatic renewal?"),
    ("cancel", "Je souhaite interrompre mes versements mensuels immédiatement.", "I want to stop my monthly payments immediately."),
    ("cancel", "Merci de clôturer mon compte et de supprimer mon abonnement.", "Please close my account and end my subscription."),
    ("cancel", "Je ne veux plus de votre service, comment me désabonner ?", "I don't want your service anymore, how do I unsubscribe?"),
    ("cancel", "Je déménage à l'étranger et je dois mettre fin à mon contrat.", "I'm moving abroad and need to terminate my contract."),
    # bug
    ("bug", "L'utilisateur signale une erreur 500 lors du paiement par carte bleue.", "The user reports a 500 error during card payment."),
    ("bug", "L'application plante dès que j'ouvre l'onglet statistiques.", "The app crashes as soon as I open the statistics tab."),
    ("bug", "Le site est inaccessible depuis ce matin.", "The website has been down since this morning."),
    ("bug", "Les notifications n'apparaissent plus depuis la dernière mise à jour.", "Notifications stopped showing since the last update."),
    ("bug", "L'export CSV génère un fichier vide.", "The CSV export produces an empty file."),
    ("bug", "La page de recherche tourne en boucle sans jamais afficher de résultat.", "The search page keeps loading and never shows results."),
    # sales
    ("sales", "Combien coûte le passage au forfait Premium annuel ?", "How much does the annual Premium plan cost?"),
    ("sales", "Proposez-vous des remises pour les associations ?", "Do you offer discounts for non-profits?"),
    ("sales", "Quelle est la différence entre l'offre Pro et l'offre Entreprise ?", "What is the difference between the Pro and Enterprise plans?"),
    ("sales", "Y a-t-il une période d'essai gratuite ?", "Is there a free trial period?"),
    ("sales", "Je voudrais un devis pour cinquante licences.", "I'd like a quote for fifty licenses."),
    ("sales", "Vos prix sont-ils HT ou TTC ?", "Are your prices before or after tax?"),
    # hors-sujet : aucune option ne convient
    (None, "Quel temps fera-t-il demain à Lyon ?", "What will the weather be like in Lyon tomorrow?"),
    (None, "Pouvez-vous me donner une recette de tarte aux pommes ?", "Can you give me an apple pie recipe?"),
    (None, "Bravo à toute l'équipe, votre service est génial !", "Congrats to the whole team, your service is great!"),
    (None, "Qui a gagné le match de football hier soir ?", "Who won the football match last night?"),
    (None, "Je cherche un stage en marketing, recrutez-vous ?", "I'm looking for a marketing internship, are you hiring?"),
    (None, "asdf qwerty 12345", "asdf qwerty 12345"),
    (None, "Mon chat refuse de manger ses croquettes.", "My cat refuses to eat its kibble."),
    (None, "Quelle est la capitale de l'Australie ?", "What is the capital of Australia?"),
]
