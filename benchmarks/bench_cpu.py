"""Latence et mémoire de Krito sur CPU, avec l'installation minimale ``krito[onnx]``.

À lancer dans un environnement sans PyTorch pour mesurer ce qu'une entreprise installerait :

    uv venv /chemin/venv-krito && VIRTUAL_ENV=/chemin/venv-krito uv pip install ".[onnx]"
    /chemin/venv-krito/bin/python benchmarks/bench_cpu.py experiments/models/krito-nli-fr-multi 2

Sortie : temps d'import + chargement, première décision, latence p50 / p95 (6 options,
40 décisions) et pic de mémoire du processus.
"""

import json
import resource
import sys
import time

t0 = time.perf_counter()
from krito import KritoEngine  # noqa: E402

model_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/models/krito-nli-fr-multi"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 2
engine = KritoEngine.from_onnx(model_dir, threads=threads, hypothesis_template="Ce texte concerne {}.")
t_load = time.perf_counter() - t0

OPTIONS = {
    "billing": "un problème de facturation ou de paiement", "shipping": "la livraison d'un colis",
    "account": "l'accès au compte", "cancel": "une résiliation d'abonnement",
    "bug": "une panne technique", "sales": "une question commerciale",
}
MESSAGES = [
    "Mon colis n'est jamais arrivé.",
    "J'ai été débité deux fois ce mois-ci, merci de corriger.",
    "Bonjour, depuis la mise à jour de ce matin l'application plante dès que j'ouvre mes factures, pouvez-vous regarder ?",
    "Je veux arrêter mon abonnement.",
    "Combien coûte l'offre Pro pour 20 personnes ?",
]
engine.classify(MESSAGES[0], OPTIONS)
t_first = time.perf_counter() - t0
latencies = []
for _ in range(8):
    for message in MESSAGES:
        start = time.perf_counter()
        engine.classify(message, OPTIONS)
        latencies.append((time.perf_counter() - start) * 1000)
latencies.sort()
print(json.dumps({
    "threads": threads,
    "import_load_s": round(t_load, 2),
    "first_decision_s": round(t_first, 2),
    "p50_ms": round(latencies[len(latencies) // 2]),
    "p95_ms": round(latencies[int(0.95 * len(latencies))]),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
}))
