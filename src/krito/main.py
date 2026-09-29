import argparse

from .krito import KritoEngine


def main():
    parser = argparse.ArgumentParser(prog="krito", description="Démonstration d'une décision Krito.")
    parser.add_argument("--textcnn", action="store_true",
                        help="Utiliser le modèle léger TextCNN livré avec Krito (aucun téléchargement).")
    args = parser.parse_args()
    if args.textcnn:
        krito = KritoEngine.from_textcnn()
    else:
        krito = KritoEngine(hypothesis_template="Ce message concerne {}.")
    context_test = "L'utilisateur signale une erreur 500 lors du paiement par carte bleue."
    options = {
        "A": "un problème d'infrastructure ou une panne technique",
        "B": "une demande de remboursement",
        "C": "une question commerciale",
    }
    result = krito.classify(context_test, options, min_entailment=0.5)
    print(result)


if __name__ == "__main__":
    main()
