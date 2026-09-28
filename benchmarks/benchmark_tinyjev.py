import time

from krito import KritoEngine

def run_benchmark():
    print("Initialisation du modèle pour le benchmark...")
    krito = KritoEngine(model_name="cross-encoder/nli-deberta-v3-xsmall")

    # Dataset de test synthétique avec cas clairs, subtils et hors-sujet
    dataset = [
        {
            "context": "Je n'arrive pas à télécharger ma facture au format PDF depuis mon espace.",
            "options": {
                "A": "Problème d'accès aux factures ou documents de paiement.",
                "B": "Demande de résiliation d'abonnement.",
                "C": "Question sur les délais de livraison physique.",
            },
            "expected": "A",
        },
        {
            "context": "Le colis indiqué comme livré n'a jamais été déposé dans ma boîte aux lettres.",
            "options": {
                "A": "Incident de paiement ou problème de carte bancaire.",
                "B": "Problème de suivi ou de livraison de colis.",
                "C": "Demande de réinitialisation de mot de passe.",
            },
            "expected": "B",
        },
        {
            "context": "Combien coûte le passage au forfait Premium annuel ?",
            "options": {
                "A": "Question sur les tarifs, abonnements et offres commerciales.",
                "B": "Demande de remboursement suite à une insatisfaction.",
                "C": "Bug technique lors de la connexion.",
            },
            "expected": "A",
        },
        # Cas ambigu (devrait être rejeté par le garde-fou margin/threshold)
        {
            "context": "Je souhaite interrompre mes versements mensuels immédiatement.",
            "options": {
                "A": "Demande de résiliation d'un abonnement mensuel.",
                "B": "Demande d'annulation d'un paiement en cours.",
                "C": "Réclamation relative au service client.",
            },
            "expected": "A",
        },
    ]

    threshold = 0.65
    min_margin = 0.30

    correct_global = 0
    correct_accepted = 0
    accepted_count = 0
    latencies = []

    print("\n--- Exécution du Benchmark ---\n")

    for idx, item in enumerate(dataset, 1):
        t0 = time.perf_counter()
        res = krito.classify(
            item["context"],
            item["options"],
            threshold=threshold,
            min_margin=min_margin,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        is_correct = res.selected_key == item["expected"]
        if is_correct:
            correct_global += 1

        if res.accepted:
            accepted_count += 1
            if is_correct:
                correct_accepted += 1

        status = "ACCEPTÉ" if res.accepted else f"REJETÉ ({res.rejection_reason})"
        print(f"Exemple {idx}: {'✓' if is_correct else '✗'}")
        print(f"  Contexte    : « {item['context']} »")
        print(f"  Choix       : {res.selected_key} (Attendu: {item['expected']})")
        print(
            f"  Confiance   : {res.confidence:.2%} | Marge: {res.margin:.2%} | Latence: {latency_ms:.2f} ms"
        )
        print(f"  Statut      : {status}\n")

    # Statistiques globales
    total = len(dataset)
    acc_global = (correct_global / total) * 100
    acc_accepted = (
        (correct_accepted / accepted_count) * 100 if accepted_count > 0 else 0.0
    )
    rejection_rate = ((total - accepted_count) / total) * 100
    avg_latency = sum(latencies) / len(latencies)

    print("================ Summary ================")
    print(f"Échantillons analysés : {total}")
    print(f"Latence moyenne      : {avg_latency:.2f} ms / requête")
    print(f"Précision Globale    : {acc_global:.1f}%")
    print(f"Précision Acceptée   : {acc_accepted:.1f}%")
    print(f"Taux de Rejet        : {rejection_rate:.1f}%")
    print("=========================================")


if __name__ == "__main__":
    run_benchmark()