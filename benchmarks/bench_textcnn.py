"""Latence et mémoire du modèle TextCNN livré avec Krito, avec l'installation minimale ``krito[onnx]``.

    uv venv /chemin/venv-krito && VIRTUAL_ENV=/chemin/venv-krito uv pip install ".[onnx]"
    /chemin/venv-krito/bin/python benchmarks/bench_textcnn.py [dossier_du_modele] [threads]

Mesure la latence avec 6 options (tout passe par le cross-encodeur) et avec 60 options
(le bi-encodeur présélectionne les ``top_k`` options transmises au cross-encodeur),
ainsi que le pic de mémoire du processus.
"""

import json
import resource
import sys
import time

t0 = time.perf_counter()
from krito import KritoEngine  # noqa: E402

model_dir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2
engine = KritoEngine.from_textcnn(model_dir, threads=threads)
t_load = time.perf_counter() - t0

OPTIONS = {
    "billing": "un problème de facturation ou de paiement", "shipping": "la livraison d'un colis",
    "account": "l'accès au compte", "cancel": "une résiliation d'abonnement",
    "bug": "une panne technique", "sales": "une question commerciale",
}
MANY = OPTIONS | {f"autre_{i}": f"un sujet administratif numéro {i}" for i in range(54)}
MESSAGES = [
    "Mon colis n'est jamais arrivé.",
    "J'ai été débité deux fois ce mois-ci, merci de corriger.",
    "Bonjour, depuis la mise à jour de ce matin l'application plante dès que j'ouvre mes factures, pouvez-vous regarder ?",
    "Je veux arrêter mon abonnement.",
    "Combien coûte l'offre Pro pour 20 personnes ?",
]


def latencies(options: dict[str, str]) -> dict:
    engine.classify(MESSAGES[0], options)
    out = []
    for _ in range(20):
        for message in MESSAGES:
            start = time.perf_counter()
            engine.classify(message, options)
            out.append((time.perf_counter() - start) * 1000)
    out.sort()
    return {"p50_ms": round(out[len(out) // 2], 2), "p95_ms": round(out[int(len(out) * 0.95)], 2)}


print(json.dumps({
    "threads": threads,
    "import_load_s": round(t_load, 2),
    "options_6": latencies(OPTIONS),
    "options_60": latencies(MANY),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
}, indent=2))
