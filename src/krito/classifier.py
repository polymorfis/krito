"""Mode supervisé pour les taxonomies fixes : ``KritoClassifier.fit(exemples)``.

Quand les catégories ne changent pas et que l'on dispose de quelques dizaines d'exemples
par catégorie, un classifieur appris est plus précis et beaucoup plus rapide que le
zero-shot : **un seul** passage du texte dans un modèle d'embeddings, quel que soit le
nombre de catégories, puis une régression logistique.

Dans l'étude du dépôt (experiments/RESULTS.md), ``e5-base`` + régression logistique atteint
95 % de précision sur le support client, et un juge appris porte l'automatisation à 90 % de
précision de 52 % à 87 % des messages.

    clf = KritoClassifier(OnnxEmbedder("modeles/e5-base")).fit(exemples)
    clf.classify("Mon colis n'est jamais arrivé.", min_judge_score=0.8)

Les hors-sujet (étiquette ``None``) ne sont jamais une catégorie : ils servent à apprendre au
juge à les rejeter. Le garde-fou ``min_similarity`` (similarité au plus proche exemple
d'entraînement de la catégorie retenue) les rejette aussi sans juge.
"""

from __future__ import annotations

import io
import json
import os
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from .calibration import Calibration, calibrate
from .data import Example, as_examples, labels_of
from .judge import ConfidenceJudge, logistic_from_dict, logistic_to_dict, require_sklearn
from .results import ClassScore, DecisionResult, rejection_reasons

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
FORMAT = "krito-classifier/1"


class Embedder(Protocol):
    """Transforme des textes en vecteurs normalisés [n_textes, dimension]."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _default_prefix(model_name: str) -> str:
    # Les modèles e5 (« multilingual-e5-base »…) attendent un préfixe « query: » devant chaque texte.
    name = Path(str(model_name)).name.lower()
    return "query: " if re.search(r"(^|[-_])e5([-_]|$)", name) else ""


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class SentenceTransformerEmbedder:
    """Embeddings via sentence-transformers (extra ``krito[torch]``)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, *, prefix: str | None = None,
                 device: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.prefix = _default_prefix(model_name) if prefix is None else prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def spec(self) -> dict:
        return {"type": "sentence-transformers", "model": self.model_name, "prefix": self.prefix}

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode([self.prefix + t for t in texts], batch_size=self.batch_size,
                                normalize_embeddings=True, show_progress_bar=False)
        return _normalize(emb)


class OnnxEmbedder:
    """Embeddings via ONNX Runtime, sans PyTorch (extra ``krito[onnx]``).

    Le dossier contient ``model.onnx`` (ou ``model.int8.onnx`` / ``model.opt.onnx``) et
    ``tokenizer.json`` ; ``experiments/export_embedder_onnx.py`` le produit à partir d'un
    modèle du Hugging Face Hub. Pooling : moyenne des tokens, puis normalisation (e5).
    """

    def __init__(self, model_dir: str | os.PathLike, *, prefix: str | None = None, max_length: int = 512,
                 threads: int | None = None, batch_size: int = 32):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        from .krito import ONNX_MODEL_FILES, _resolve_model_dir

        self.model_dir = str(model_dir)
        path = _resolve_model_dir(model_dir)
        onnx_path = next((path / n for n in ONNX_MODEL_FILES if (path / n).exists()), None)
        if onnx_path is None:
            raise FileNotFoundError(f"Aucun modèle ONNX ({', '.join(ONNX_MODEL_FILES)}) dans {path}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.prefix = _default_prefix(self.model_dir) if prefix is None else prefix
        self.batch_size = batch_size

    @property
    def spec(self) -> dict:
        return {"type": "onnx", "model": self.model_dir, "prefix": self.prefix}

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(texts), self.batch_size):
            enc = self.tokenizer.encode_batch([self.prefix + t for t in texts[start:start + self.batch_size]])
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64), "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            hidden = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
            if hidden.ndim == 3:  # [lot, tokens, dim] : moyenne des tokens réels
                m = mask[..., None].astype(np.float32)
                hidden = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out.append(hidden)
        return _normalize(np.concatenate(out)) if out else np.zeros((0, 0), dtype=np.float32)


def _embedder_from_spec(spec: dict | None) -> Embedder:
    if not spec:
        raise ValueError("Ce classifieur utilise un embedder personnalisé : passez-le à load(..., embedder=...).")
    if spec["type"] == "sentence-transformers":
        return SentenceTransformerEmbedder(spec["model"], prefix=spec["prefix"])
    if spec["type"] == "onnx":
        return OnnxEmbedder(spec["model"], prefix=spec["prefix"])
    raise ValueError(f"Type d'embedder inconnu : {spec['type']}")


class KritoClassifier:
    """Classifieur supervisé : embeddings + régression logistique (+ juge de confiance).

    embedder : modèle d'embeddings (``OnnxEmbedder``, ``SentenceTransformerEmbedder`` ou le
        vôtre, avec une méthode ``encode(textes) -> np.ndarray``). Par défaut :
        ``SentenceTransformerEmbedder("intfloat/multilingual-e5-base")`` (``krito[torch]``).
    labels : descriptions facultatives des catégories, pour ``selected_label``.
    C : inverse de la régularisation de la ``LogisticRegression`` scikit-learn.
    """

    def __init__(self, embedder: Embedder | None = None, *, labels: Mapping[str, str] | None = None,
                 C: float = 10.0):
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
        self.labels = dict(labels or {})
        self.C = C
        self.classes_: list[str] = []
        self.judge: ConfidenceJudge | None = None
        self._model = None  # sklearn.linear_model.LogisticRegression
        self._train_emb: np.ndarray | None = None
        self._train_y: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ entraînement

    def fit(self, examples, *, judge: bool = True, n_folds: int = 5, seed: int = 42) -> KritoClassifier:
        """Apprend les catégories à partir d'exemples annotés.

        examples : couples ``(texte, catégorie)``, dicts ``{"text", "label"}`` ou
            ``{catégorie: [textes]}`` ; ``None`` / ``"none"`` pour les hors-sujet.
        judge : entraîne aussi un juge de confiance sur des décisions **hors pli**
            (validation croisée à ``n_folds`` plis), pour que sa confiance reflète des
            textes jamais vus. Ajoutez des hors-sujet aux exemples pour qu'il apprenne à les rejeter.
        """
        require_sklearn()
        examples = as_examples(examples)
        classes = labels_of(examples)
        if len(classes) < 2:
            raise ValueError("Il faut des exemples d'au moins 2 catégories.")
        unknown = set(self.labels) - set(classes)
        if unknown:
            raise ValueError(f"Catégories décrites mais sans exemple : {sorted(unknown)}.")
        emb = self.embedder.encode([ex.text for ex in examples])
        index = {c: i for i, c in enumerate(classes)}
        y = np.array([index[ex.label] if ex.label is not None else -1 for ex in examples])
        in_domain = y >= 0

        self.classes_ = classes
        self._fit_on(emb[in_domain], y[in_domain])
        self.judge = self._fit_judge(examples, emb, y, n_folds, seed) if judge else None
        return self

    def _fit_on(self, emb: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(C=self.C, max_iter=2000).fit(emb.astype(np.float64), y)
        self._train_emb = _normalize(emb)
        self._train_y = y

    def _fit_judge(self, examples: list[Example], emb: np.ndarray, y: np.ndarray, n_folds: int, seed: int):
        # Plis stratifiés sur toutes les étiquettes, hors-sujet compris (-1) ; leur nombre est
        # limité par la plus petite catégorie, pour que chaque pli d'entraînement les contienne toutes.
        n_folds = min(n_folds, int(np.bincount(y[y >= 0]).min()))
        if n_folds < 2:
            warnings.warn("Trop peu d'exemples par catégorie pour entraîner le juge de confiance : juge désactivé.",
                          stacklevel=3)
            return None
        from sklearn.model_selection import StratifiedKFold

        results, labels = [None] * len(examples), [ex.label for ex in examples]
        with warnings.catch_warnings():  # peu de hors-sujet : scikit-learn prévient, sans gravité
            warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
            splits = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(emb, y))
        for train, test in splits:
            train = train[y[train] >= 0]
            inner = KritoClassifier(self.embedder, labels=self.labels, C=self.C)
            inner.classes_ = self.classes_
            inner._fit_on(emb[train], y[train])
            for i, r in zip(test, inner._decide_many(emb[test], {}, 1.0)):
                results[i] = r
        kept = [(r, lab) for r, lab in zip(results, labels) if r is not None]
        try:
            return ConfidenceJudge().fit([r for r, _ in kept], [lab for _, lab in kept])
        except ValueError as exc:
            warnings.warn(f"Juge de confiance non entraîné : {exc}", stacklevel=3)
            return None

    # ------------------------------------------------------------------ décision

    def classify(
        self,
        text: str,
        *,
        threshold: float | None = None,
        min_margin: float | None = None,
        min_similarity: float | None = None,
        min_judge_score: float | None = None,
        temperature: float = 1.0,
    ) -> DecisionResult:
        """Classe un texte parmi les catégories apprises.

        Garde-fous (optionnels) : ``threshold`` (confiance), ``min_margin`` (écart avec la 2e
        catégorie), ``min_similarity`` (similarité au plus proche exemple d'entraînement de la
        catégorie retenue : rejette les hors-sujet), ``min_judge_score`` (juge de confiance).
        """
        return self.classify_batch([text], threshold=threshold, min_margin=min_margin,
                                   min_similarity=min_similarity, min_judge_score=min_judge_score,
                                   temperature=temperature)[0]

    def classify_batch(
        self,
        texts: Iterable[str],
        *,
        threshold: float | None = None,
        min_margin: float | None = None,
        min_similarity: float | None = None,
        min_judge_score: float | None = None,
        temperature: float = 1.0,
    ) -> list[DecisionResult]:
        """``classify`` sur plusieurs textes, dans l'ordre reçu (un seul appel à l'embedder)."""
        if self._model is None:
            raise RuntimeError("Le classifieur n'est pas entraîné : appelez fit() d'abord.")
        if isinstance(texts, str):
            raise ValueError("classify_batch attend une liste de textes, pas une chaîne.")
        texts = list(texts)
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("Le contexte doit être une chaîne non vide.")
        if temperature <= 0:
            raise ValueError("La température doit être strictement positive.")
        if min_judge_score is not None and self.judge is None:
            raise ValueError("min_judge_score demande un juge : fit(..., judge=True) avec des exemples variés.")
        if not texts:
            return []
        emb = _normalize(self.embedder.encode(texts))
        if emb.shape[1] != self._train_emb.shape[1]:
            raise ValueError(f"L'embedder produit des vecteurs de dimension {emb.shape[1]}, "
                             f"le classifieur a été entraîné en dimension {self._train_emb.shape[1]}.")
        limits = dict(threshold=threshold, min_margin=min_margin, min_similarity=min_similarity,
                      min_judge_score=min_judge_score)
        return self._decide_many(emb, limits, temperature)

    def _similarities(self, emb: np.ndarray) -> np.ndarray:
        """Similarité de chaque texte au plus proche exemple d'entraînement de chaque catégorie."""
        out = np.full((len(emb), len(self.classes_)), -1.0)
        for start in range(0, len(emb), 256):
            sims = emb[start:start + 256] @ self._train_emb.T
            for c in range(len(self.classes_)):
                members = self._train_y == c
                if members.any():
                    out[start:start + 256, c] = sims[:, members].max(1)
        return out

    def _decide_many(self, emb: np.ndarray, limits: dict, temperature: float) -> list[DecisionResult]:
        # Logits complets, même si une catégorie manque (plis de validation croisée)
        logits = np.full((len(emb), len(self.classes_)), -1e9)
        raw = self._model.decision_function(emb.astype(np.float64))
        if raw.ndim == 1:  # scikit-learn : une seule colonne pour 2 classes
            raw = np.column_stack([-raw / 2, raw / 2])
        logits[:, self._model.classes_] = raw
        probs = _softmax(logits / temperature, axis=1)
        sims = self._similarities(emb)
        results = []
        for lg, p, s in zip(logits, probs, sims):
            order = np.argsort(-p, kind="stable")
            key = self.classes_[order[0]]
            scores = {c: ClassScore(logit=float(lg[i]), probability=float(p[i]), similarity=float(s[i]))
                      for i, c in enumerate(self.classes_)}
            result = DecisionResult(
                selected_key=key, selected_label=self.labels.get(key, key),
                confidence=float(p[order[0]]), margin=float(p[order[0]] - p[order[1]]),
                accepted=True, rejection_reason=None, scores=scores,
            )
            judge_score = self.judge.score(result) if self.judge is not None else None
            values = dict(threshold=result.confidence, min_margin=result.margin,
                          min_similarity=scores[key].similarity, min_judge_score=judge_score)
            reasons = rejection_reasons(values, limits)
            results.append(replace(result, accepted=not reasons, rejection_reason=" | ".join(reasons) or None,
                                   judge_score=judge_score))
        return results

    # ------------------------------------------------------------------ calibration

    def calibrate(
        self,
        examples,
        *,
        target_precision: float = 0.95,
        guardrails: Sequence[str] | None = None,
        grid_size: int = 20,
    ) -> Calibration:
        """Calibre les garde-fous sur des exemples annotés **distincts** de ceux de ``fit``.

        guardrails : par défaut ``("min_judge_score",)`` s'il y a un juge, sinon
        ``("min_similarity", "min_margin")``.
        """
        examples = as_examples(examples)
        unknown = {ex.label for ex in examples if ex.label is not None} - set(self.classes_)
        if unknown:
            raise ValueError(f"Étiquettes inconnues du classifieur : {sorted(unknown)}.")
        results = self.classify_batch([ex.text for ex in examples])
        if guardrails is None:
            guardrails = ("min_judge_score",) if self.judge is not None else ("min_similarity", "min_margin")
        return calibrate(results, [ex.label for ex in examples], target_precision=target_precision,
                         guardrails=guardrails, grid_size=grid_size)

    # ------------------------------------------------------------------ sérialisation

    def save(self, path: str | os.PathLike) -> None:
        """Enregistre le classifieur dans un fichier ``.npz`` (sans pickle).

        Le modèle d'embeddings n'est pas copié, seulement sa référence : il doit rester
        disponible au chargement (ou être fourni à ``load``).
        """
        if self._model is None:
            raise RuntimeError("Le classifieur n'est pas entraîné.")
        meta = {
            "format": FORMAT,
            "classes": self.classes_,
            "labels": self.labels,
            "embedder": getattr(self.embedder, "spec", None),
            "judge": self.judge.to_dict() if self.judge is not None else None,
            "logistic": logistic_to_dict(self._model),
        }
        buf = io.BytesIO()
        np.savez_compressed(
            buf, meta=np.array(json.dumps(meta, ensure_ascii=False)), train_emb=self._train_emb.astype(np.float32),
            train_y=self._train_y,
        )
        Path(path).write_bytes(buf.getvalue())

    @classmethod
    def load(cls, path: str | os.PathLike, *, embedder: Embedder | None = None) -> KritoClassifier:
        """Charge un classifieur ; l'embedder est recréé à partir de sa référence si besoin."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path} n'est pas un classifieur Krito ({FORMAT}).")
            model = logistic_from_dict(meta["logistic"])
            train_emb, train_y = data["train_emb"], data["train_y"]
        clf = cls(embedder if embedder is not None else _embedder_from_spec(meta["embedder"]),
                  labels=meta["labels"], C=model.C)
        clf.classes_ = meta["classes"]
        clf._model, clf._train_emb, clf._train_y = model, train_emb, train_y
        clf.judge = ConfidenceJudge.from_dict(meta["judge"]) if meta["judge"] else None
        return clf
