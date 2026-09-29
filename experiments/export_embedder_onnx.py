"""Exporte un modèle d'embeddings (e5 par défaut) en ONNX pour ``KritoClassifier`` sans PyTorch.

Sortie : un dossier utilisable par ``OnnxEmbedder(dossier)`` : ``model.onnx`` (sorties =
états cachés des tokens, le pooling moyen est fait par Krito), ``tokenizer.json`` et, avec
``--int8``, ``model.int8.onnx`` (quantification dynamique des poids, environ 4 fois plus petit).

    uv run python experiments/export_embedder_onnx.py --out modeles/e5-base
    uv run python experiments/export_embedder_onnx.py --model intfloat/multilingual-e5-small --out modeles/e5-small --int8

Le script vérifie que les embeddings ONNX correspondent à ceux de sentence-transformers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

DEFAULT = "intfloat/multilingual-e5-base"


class _Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def export(model_name: str, out: Path) -> Path:
    from transformers import AutoModel, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    enc = tok(["query: texte"], return_tensors="pt")
    torch.onnx.export(
        _Wrapper(model), (enc["input_ids"], enc["attention_mask"]), str(out / "model.onnx"),
        input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                      "last_hidden_state": {0: "b", 1: "s"}},
        opset_version=17, dynamo=False,
    )
    return out


def quantize(out: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(out / "model.onnx"), str(out / "model.int8.onnx"), weight_type=QuantType.QUInt8)


def check(model_name: str, out: Path) -> float:
    """Écart maximal (cosinus) entre les embeddings ONNX et sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    from krito import OnnxEmbedder

    texts = ["Mon colis n'est jamais arrivé.", "J'ai été débité deux fois.", "Bonjour " * 50]
    ref = SentenceTransformer(model_name, device="cpu").encode(
        ["query: " + t for t in texts], normalize_embeddings=True)
    got = OnnxEmbedder(out, prefix="query: ").encode(texts)
    return float(1 - (ref * got).sum(1).min())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT, help="modèle d'embeddings du Hugging Face Hub")
    p.add_argument("--out", required=True, help="dossier de sortie")
    p.add_argument("--int8", action="store_true", help="ajoute une version quantifiée int8")
    args = p.parse_args()
    out = Path(args.out)
    (out / "model.int8.onnx").unlink(missing_ok=True)  # la vérification porte sur le fp32
    export(args.model, out)
    gap = check(args.model, out)
    print(f"écart max (1 − cosinus) entre ONNX fp32 et PyTorch : {gap:.2e}")
    np.testing.assert_array_less(gap, 1e-3)
    if args.int8:
        quantize(out)
        print("la version int8 est utilisée en priorité : vérifiez sa précision sur vos données.")
    print(f"modèle prêt : OnnxEmbedder({str(out)!r})")


if __name__ == "__main__":
    main()
