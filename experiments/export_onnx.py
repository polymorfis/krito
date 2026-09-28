"""Entraîne le NLI final (tout le support), l'exporte en ONNX et le quantifie en int8.

Vérifie la fidélité (RH, jamais vu à l'entraînement) et mesure l'inférence sans PyTorch.

    uv run python export_onnx.py --model mdeberta
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import features
from lib import ROOT, SEED, evaluate, load
from run_nli_finetune import NLI, build_pairs
from run_zero_shot import nli_scores

MODELS = ROOT / "models"


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(key: str) -> Path:
    out = MODELS / f"krito-nli-fr-{key}"
    if (out / "model.int8.onnx").exists():
        return out
    ds = load("support")
    torch.manual_seed(SEED)
    nli = NLI(features.NLI_MODELS[key], fp32=True)
    nli.train(build_pairs(ds.texts, ds.labels, ds.classes, "support", random.Random(SEED)), epochs=3)
    out.mkdir(parents=True, exist_ok=True)
    nli.model.save_pretrained(out / "hf")
    nli.tok.save_pretrained(out / "hf")
    return to_onnx(nli, out)


def export_from_hf(out: Path) -> Path:
    """Exporte un modèle déjà entraîné, enregistré dans ``out/hf``."""
    if (out / "model.int8.onnx").exists():
        if not (out / "model.opt.onnx").exists():
            optimize_offline(out / "model.int8.onnx", out / "model.opt.onnx")
        return out
    return to_onnx(NLI(str(out / "hf"), fp32=True), out)


def to_onnx(nli: NLI, out: Path) -> Path:
    export_fp32(nli, out)
    return finalize(out)


def finalize(out: Path) -> Path:
    """Quantification int8 puis optimisation hors ligne (sans PyTorch en mémoire si possible)."""
    quantize(out / "model.onnx", out / "model.int8.onnx")
    optimize_offline(out / "model.int8.onnx", out / "model.opt.onnx")
    return out


def export_fp32(nli: NLI, out: Path) -> Path:
    nli.tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    (out / "labels.json").write_text(json.dumps({"entailment": nli.order[0], "contradiction": nli.order[1], "neutral": nli.order[2]}))

    model = _Wrapper(nli.model.cpu().eval())
    enc = nli.tok(["texte"], ["hypothèse"], return_tensors="pt")
    torch.onnx.export(
        model, (enc["input_ids"], enc["attention_mask"]), str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
        opset_version=17, dynamo=False,
    )
    return out


def optimize_offline(src: Path, dst: Path) -> None:
    """Applique une fois pour toutes les optimisations « basic » d'ONNX Runtime.

    Indépendantes du matériel, elles sont sinon refaites à chaque chargement et coûtent de
    la mémoire : pic de RAM mesuré 869 Mo -> 710 Mo, latence inchangée.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])


def quantize(src: Path, dst: Path) -> None:
    """Quantification int8 (dynamique) des poids.

    Mesures de sensibilité par groupe de couches (accord avec le fp32 sur le domaine RH) :
    query/key/value/attention.output : 97-100 %, intermediate : 93 %,
    **output.dense du bloc feed-forward : 80 %** (activations extrêmes après GELU).
    On quantifie donc tout sauf output.dense, ainsi que la table d'embeddings (Gather),
    qui représente l'essentiel des paramètres d'un modèle multilingue.
    Les MatMul entre deux activations (attention) ne sont jamais quantifiés.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    graph = onnx.load(str(src), load_external_data=False).graph
    inits = {i.name for i in graph.initializer}
    include = [
        n.name for n in graph.node
        if (n.op_type == "MatMul" and any(i in inits for i in n.input)
            and not ("/output/dense" in n.name and "/attention/" not in n.name))
        or (n.op_type == "Gather" and any(i in inits and "embeddings" in i for i in n.input))
    ]
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8,
                     op_types_to_quantize=["MatMul", "Gather"], nodes_to_quantize=include)


# Script exécuté dans un processus séparé : aucune importation de torch ni transformers.
ORT_SCRIPT = r'''
import json, sys, time
t0 = time.perf_counter()
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
model_dir, onnx_name, threads = sys.argv[1], sys.argv[2], int(sys.argv[3])
opts = ort.SessionOptions(); opts.intra_op_num_threads = threads
sess = ort.InferenceSession(f"{model_dir}/{onnx_name}", opts, providers=["CPUExecutionProvider"])
tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
tok.enable_truncation(128); tok.enable_padding()
labels = json.load(open(f"{model_dir}/labels.json"))
payload = json.load(sys.stdin)
def run(text, hyps):
    enc = tok.encode_batch([(text, h) for h in hyps])
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    return sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]
first = run(payload["texts"][0], payload["hyps"])
t_first = time.perf_counter() - t0
lat, out = [], []
for t in payload["texts"]:
    a = time.perf_counter(); lg = run(t, payload["hyps"]); lat.append(time.perf_counter() - a)
    out.append(lg[:, [labels["entailment"], labels["contradiction"], labels["neutral"]]].tolist())
print(json.dumps({"cold_start_s": t_first, "p50_ms": 1000 * float(np.median(lat)), "logits": out}))
'''


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] in ("--export-fp32", "--finalize"):
        # Étapes isolées, lancées dans un processus dédié par finetune_custom.py :
        # leurs pics de mémoire (≈ 5,4 Go et ≈ 7,3 Go) ne s'additionnent pas.
        out = Path(sys.argv[2])
        if sys.argv[1] == "--export-fp32":
            export_fp32(NLI(str(out / "hf"), device="cpu", fp32=True), out)
        else:
            finalize(out)
        return
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mdeberta", help="mdeberta (support seul) ou multi (déjà entraîné)")
    args = p.parse_args()
    out = export_from_hf(MODELS / "krito-nli-fr-multi") if args.model == "multi" else export(args.model)
    rh = load("rh")
    ref = features  # noqa
    script = out / "_ort_bench.py"
    script.write_text(ORT_SCRIPT)
    payload = json.dumps({"texts": rh.texts, "hyps": rh.hypotheses})

    torch_nli = NLI(str(out / "hf"), fp32=True)
    torch_logits = torch_nli.logits(rh.texts, rh.hypotheses)
    del torch_nli

    report = {"model": args.model}
    for onnx_name in ("model.onnx", "model.int8.onnx"):
        for threads in (1, 2):
            r = json.loads(subprocess.run([sys.executable, str(script), str(out), onnx_name, str(threads)],
                                          input=payload, capture_output=True, text=True, check=True).stdout)
            lg = np.array(r.pop("logits"))
            pred, conf = nli_scores(lg)["abs"]
            m = evaluate(rh.y, pred, conf)
            agree = float((pred == nli_scores(torch_logits)["abs"][0]).mean())
            key = f"{onnx_name}@{threads}thr"
            report[key] = {
                "size_mb": (out / onnx_name).stat().st_size / 1e6,
                "cold_start_s": r["cold_start_s"], "p50_ms_5_options": r["p50_ms"],
                "rh_accuracy": m["accuracy"], "rh_auroc_ood": m["auroc_ood"], "agreement_with_torch": agree,
            }
            print(key, {k: round(v, 3) for k, v in report[key].items()}, flush=True)
    (ROOT / "results" / f"onnx_{args.model}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
