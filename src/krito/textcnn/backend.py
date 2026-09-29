"""Backend d'inférence du modèle TextCNN, sur CPU, sans PyTorch (ONNX Runtime + tokenizers).

Le dossier d'un modèle contient :

- ``encoder.onnx`` : backbone TextCNN + bi-encodeur, ``(input_ids, attention_mask)`` ->
  ``(token_features [B, L, H], sentence_embedding [B, D])`` ;
- ``cross.onnx`` : cross-encodeur, ``(p_tokens, p_mask, q_tokens, q_mask)`` -> ``nli_logits [B, 3]`` ;
- ``tokenizer.json`` et ``config.json`` (hyperparamètres, gabarit d'hypothèse, routage).

Chaque texte distinct (contexte ou option) n'est encodé qu'une fois par appel ; le
cross-encodeur ne travaille que sur les caractéristiques déjà calculées.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..krito import _warn_truncation

DEFAULT_TEXTCNN_MODEL = Path(__file__).parent / "krito-textcnn-fr"
PRUNED_LOGITS = (-10.0, 10.0, 0.0)  # identique à model.PRUNED_LOGITS (module sans PyTorch)
ENCODER_FILES = ("encoder.int8.onnx", "encoder.onnx")
CROSS_FILES = ("cross.int8.onnx", "cross.onnx")


class AdaptiveRouter:
    """Routage « System 1 » commun aux backends ONNX et PyTorch.

    Les sous-classes fournissent ``_encode(texts, max_length)`` et ``_cross(...)`` ; ce
    code regroupe les paires par contexte et n'envoie au cross-encodeur que les options
    retenues : toutes jusqu'à ``threshold`` options, sinon les ``top_k`` meilleures du
    bi-encodeur. Les options écartées reçoivent ``PRUNED_LOGITS`` (probabilité ≈ 0).
    """

    # Le routage compare les options d'un même contexte : KritoEngine ne les répartit pas sur plusieurs appels.
    needs_whole_contexts = True
    threshold: int
    top_k: int
    max_context_length: int
    max_option_length: int

    def _init_tokenizers(self, tokenizer_json: str) -> None:
        """Un tokenizer par longueur maximale, configuré une fois pour toutes : aucun état n'est
        modifié pendant les appels, le backend est utilisable depuis plusieurs threads.

        La troncature est réglée un token au-delà de la limite pour détecter les textes trop
        longs (``Encoding.overflowing`` n'est pas renseigné sans post-traitement de paires).
        """
        from tokenizers import Tokenizer

        self._tokenizers = {}
        for length in {self.max_context_length, self.max_option_length}:
            tok = Tokenizer.from_str(tokenizer_json)
            tok.enable_truncation(length + 1)
            tok.enable_padding(pad_id=0, pad_token="[PAD]")
            self._tokenizers[length] = tok

    def _tokenize(self, texts: list[str], max_length: int) -> tuple[np.ndarray, np.ndarray, int]:
        """-> (input_ids [B, L], attention_mask [B, L], nb de textes tronqués), L <= max_length."""
        enc = self._tokenizers[max_length].encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)[:, :max_length]
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        return ids, mask[:, :max_length], int((mask.sum(axis=1) > max_length).sum())

    def _encode(self, texts: list[str], max_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """-> (tokens [B, L, H], masque [B, L], embeddings [B, D], nb de textes tronqués)."""
        raise NotImplementedError

    def _cross(self, p_tok: np.ndarray, p_mask: np.ndarray, q_tok: np.ndarray, q_mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Logits NLI [n_paires, 3] ordonnés (entailment, contradiction, neutral)."""
        out = np.tile(np.asarray(PRUNED_LOGITS, dtype=np.float32), (len(pairs), 1))
        if not pairs:
            return out
        contexts = list(dict.fromkeys(p[0] for p in pairs))
        options = list(dict.fromkeys(p[1] for p in pairs))
        c_idx = {t: i for i, t in enumerate(contexts)}
        o_idx = {t: i for i, t in enumerate(options)}
        c_tok, c_mask, c_emb, n_trunc = self._encode(contexts, self.max_context_length)
        _warn_truncation(n_trunc, self.max_context_length)
        o_tok, o_mask, o_emb, _ = self._encode(options, self.max_option_length)

        groups: dict[int, list[int]] = {}
        for i, (c, _) in enumerate(pairs):
            groups.setdefault(c_idx[c], []).append(i)
        rows, ci, oi = [], [], []
        for c, members in groups.items():
            if len(members) > self.threshold:
                opts = np.array([o_idx[pairs[i][1]] for i in members])
                sims = o_emb[opts] @ c_emb[c]
                keep = np.argsort(-sims, kind="stable")[: self.top_k]
                members = [members[j] for j in sorted(keep)]
            rows += members
            ci += [c] * len(members)
            oi += [o_idx[pairs[i][1]] for i in members]
        out[rows] = self._cross(c_tok[ci], c_mask[ci], o_tok[oi], o_mask[oi])
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        """Vecteurs normalisés du bi-encodeur (utiles pour la recherche ou le cache d'options)."""
        return self._encode(texts, self.max_context_length)[2]


class TextCNNBackend(AdaptiveRouter):
    """Backend Krito pour le modèle TextCNN exporté en ONNX (extra ``krito[onnx]``).

    ``model_dir`` : dossier du modèle ; par défaut le modèle ``krito-textcnn-fr`` livré avec Krito.
    ``threshold`` / ``top_k`` : remplacent le routage enregistré dans ``config.json``.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike | None = None,
        *,
        threads: int | None = None,
        threshold: int | None = None,
        top_k: int | None = None,
    ):
        import onnxruntime as ort

        model_dir = Path(model_dir) if model_dir is not None else DEFAULT_TEXTCNN_MODEL
        self.config = json.loads((model_dir / "config.json").read_text())
        self.threshold = threshold if threshold is not None else self.config["threshold"]
        self.top_k = top_k if top_k is not None else self.config["top_k"]
        if self.threshold < 1 or self.top_k < 1:
            raise ValueError("threshold et top_k doivent être >= 1.")
        self.max_context_length = self.config["max_context_length"]
        self.max_option_length = self.config["max_option_length"]
        self.hypothesis_template = self.config["hypothesis_template"]

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads

        def session(names):
            path = next((model_dir / n for n in names if (model_dir / n).exists()), None)
            if path is None:
                raise FileNotFoundError(f"Aucun modèle ({', '.join(names)}) dans {model_dir}")
            return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        self.encoder = session(ENCODER_FILES)
        self.cross = session(CROSS_FILES)
        self._init_tokenizers((model_dir / "tokenizer.json").read_text(encoding="utf-8"))

    def _encode(self, texts, max_length):
        ids, mask, n_trunc = self._tokenize(texts, max_length)
        tokens, emb = self.encoder.run(None, {"input_ids": ids, "attention_mask": mask})
        return tokens, mask, emb, n_trunc

    def _cross(self, p_tok, p_mask, q_tok, q_mask):
        feeds = {"p_tokens": p_tok, "p_mask": p_mask, "q_tokens": q_tok, "q_mask": q_mask}
        return self.cross.run(None, feeds)[0]
