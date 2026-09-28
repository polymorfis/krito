"""Calcule et met en cache les sorties des modèles pré-entraînés (sans entraînement).

- NLI : logits (entailment, contradiction, neutral) pour chaque paire (texte, hypothèse)
- Embeddings : vecteurs normalisés des textes et des hypothèses
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lib import CACHE, load

NLI_MODELS = {
    "xsmall": "cross-encoder/nli-deberta-v3-xsmall",
    "deberta_base": "cross-encoder/nli-deberta-v3-base",
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "minilm_multi": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
}
EMB_MODELS = {
    "e5_small": "intfloat/multilingual-e5-small",
    "e5_base": "intfloat/multilingual-e5-base",
}


def label_order(config) -> list[int]:
    """Indices (entailment, contradiction, neutral) selon la config du modèle."""
    id2label = {int(k): str(v).lower() for k, v in config.id2label.items()}
    find = lambda s: next(i for i, l in id2label.items() if s in l)  # noqa: E731
    return [find("entail"), find("contradict"), find("neutral")]


def nli_logits(model_name: str, texts: list[str], hyps: list[str]) -> np.ndarray:
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    order = label_order(ce.model.config)
    pairs = [(t, h) for t in texts for h in hyps]
    raw = ce.predict(pairs, batch_size=16, apply_softmax=False, show_progress_bar=False)
    return np.asarray(raw)[:, order].reshape(len(texts), len(hyps), 3)


def embeddings(model_name: str, texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    prefix = "query: " if "e5" in model_name else ""
    return st.encode([prefix + t for t in texts], normalize_embeddings=True, batch_size=16)


def get(kind: str, key: str, dataset: str) -> dict[str, np.ndarray]:
    """Charge depuis le cache ou calcule."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{kind}_{key}_{dataset}.npz"
    if path.exists():
        return dict(np.load(path))
    ds = load(dataset)
    if kind == "nli":
        out = {"logits": nli_logits(NLI_MODELS[key], ds.texts, ds.hypotheses)}
    else:
        out = {
            "texts": embeddings(EMB_MODELS[key], ds.texts),
            "hyps": embeddings(EMB_MODELS[key], ds.hypotheses),
        }
    np.savez(path, **out)
    return out


if __name__ == "__main__":
    for dataset in ("support", "rh"):
        for key in NLI_MODELS:
            t0 = time.perf_counter()
            get("nli", key, dataset)
            print(f"nli {key:14} {dataset:8} {time.perf_counter() - t0:6.1f}s")
        for key in EMB_MODELS:
            t0 = time.perf_counter()
            get("emb", key, dataset)
            print(f"emb {key:14} {dataset:8} {time.perf_counter() - t0:6.1f}s")
