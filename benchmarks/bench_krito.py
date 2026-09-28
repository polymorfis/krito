"""Benchmark qualité + latence de Krito.

Exemples :
    uv run python benchmarks/bench_krito.py --lang fr
    uv run python benchmarks/bench_krito.py --lang en --template
    uv run python benchmarks/bench_krito.py --model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --lang fr --template
    uv run python benchmarks/bench_krito.py --device cpu --latency-only
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

T_START = time.perf_counter()
from krito.krito import KritoEngine  # noqa: E402  (mesure du coût d'import)

T_IMPORT = time.perf_counter() - T_START

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LABELS, SAMPLES, TEMPLATES  # noqa: E402


def expected_calibration_error(confs: list[float], correct: list[bool], bins: int = 10) -> float:
    total = len(confs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo < c <= hi or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        ece += len(idx) / total * abs(acc - conf)
    return ece


def run_quality(engine: KritoEngine, lang: str, template: bool, threshold: float, min_margin: float) -> dict:
    labels = LABELS[lang]
    if template:
        labels = {k: TEMPLATES[lang].format(v) for k, v in labels.items()}
    col = 1 if lang == "fr" else 2

    in_confs, in_correct, in_accepted = [], [], []
    ood_accepted, ood_confs, max_entail_in, max_entail_ood = [], [], [], []
    errors = []
    for sample in SAMPLES:
        expected, text = sample[0], sample[col]
        res = engine.classify(text, labels, threshold=threshold, min_margin=min_margin)
        best_entail = max(s.entailment for s in res.scores.values())
        if expected is None:
            ood_accepted.append(res.accepted)
            ood_confs.append(res.confidence)
            max_entail_ood.append(best_entail)
        else:
            ok = res.selected_key == expected
            in_confs.append(res.confidence)
            in_correct.append(ok)
            in_accepted.append(res.accepted)
            max_entail_in.append(best_entail)
            if not ok:
                errors.append((text, expected, res.selected_key, round(res.confidence, 3)))

    n = len(in_correct)
    accepted_idx = [i for i in range(n) if in_accepted[i]]
    sel_acc = sum(in_correct[i] for i in accepted_idx) / len(accepted_idx) if accepted_idx else float("nan")
    return {
        "n_in_domain": n,
        "n_ood": len(ood_accepted),
        "accuracy": sum(in_correct) / n,
        "mean_confidence": statistics.mean(in_confs),
        "ece": expected_calibration_error(in_confs, in_correct),
        "confident_errors_gt_0.9": sum(1 for c, ok in zip(in_confs, in_correct) if not ok and c > 0.9),
        "coverage": len(accepted_idx) / n,
        "selective_accuracy": sel_acc,
        "ood_accepted_rate": sum(ood_accepted) / len(ood_accepted),
        "ood_mean_confidence": statistics.mean(ood_confs),
        "mean_max_entail_logit_in": statistics.mean(max_entail_in),
        "mean_max_entail_logit_ood": statistics.mean(max_entail_ood),
        "errors": errors,
    }


def run_latency(engine: KritoEngine, n_options: int = 6, runs: int = 50) -> dict:
    labels = dict(list(LABELS["en"].items())[:n_options])
    text = SAMPLES[0][2]
    for _ in range(5):  # warm-up
        engine.classify(text, labels)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        engine.classify(text, labels)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    # Débit : N contextes traités en boucle (l'API n'offre pas de batch multi-contextes)
    texts = [s[2] for s in SAMPLES]
    t0 = time.perf_counter()
    for t in texts:
        engine.classify(t, labels)
    per_sec = len(texts) / (time.perf_counter() - t0)
    return {
        "options": n_options,
        "p50_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * len(times)) - 1],
        "decisions_per_sec": per_sec,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cross-encoder/nli-deberta-v3-xsmall")
    p.add_argument("--device", default=None)
    p.add_argument("--lang", choices=["fr", "en"], default="fr")
    p.add_argument("--template", action="store_true", help="Enrobe les options dans une phrase-hypothèse")
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--min-margin", type=float, default=0.30)
    p.add_argument("--latency-only", action="store_true")
    p.add_argument("--quality-only", action="store_true")
    args = p.parse_args()

    t0 = time.perf_counter()
    engine = KritoEngine(model_name=args.model, device=args.device)
    t_load = time.perf_counter() - t0

    out = {
        "model": args.model,
        "device": getattr(engine.backend, "device", "onnx"),
        "lang": args.lang,
        "template": args.template,
        "import_s": round(T_IMPORT, 2),
        "load_s": round(t_load, 2),
    }
    if not args.latency_only:
        out["quality"] = run_quality(engine, args.lang, args.template, args.threshold, args.min_margin)
    if not args.quality_only:
        out["latency"] = run_latency(engine)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=lambda x: round(x, 3)))


if __name__ == "__main__":
    main()
