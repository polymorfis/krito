from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
ONNX_MODEL_FILES = ("model.opt.onnx", "model.int8.onnx", "model.onnx")


@dataclass(frozen=True)
class OptionScore:
    """Détail des scores NLI bruts et de la décision pour une option."""

    entailment: float
    contradiction: float
    neutral: float
    decision_score: float
    probability: float
    entailment_prob: float


@dataclass(frozen=True)
class DecisionResult:
    """Résultat structuré de la prise de décision probabiliste."""

    selected_key: str
    selected_label: str
    confidence: float
    margin: float
    accepted: bool
    rejection_reason: str | None
    scores: dict[str, OptionScore]


class Backend(Protocol):
    """Calcule les logits NLI d'une liste de paires (prémisse, hypothèse).

    Renvoie un tableau [n_paires, 3] dans l'ordre (entailment, contradiction, neutral).
    """

    def logits(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


def map_label_indices(id2label: dict) -> list[int]:
    """Indices (entailment, contradiction, neutral) à partir de ``id2label``."""
    mapping = {}
    for idx, label in id2label.items():
        lbl = str(label).lower()
        for name, needle in (("entailment", "entail"), ("contradiction", "contradict"), ("neutral", "neutral")):
            if needle in lbl and not lbl.startswith("not"):
                mapping[name] = int(idx)
    missing = {"entailment", "contradiction", "neutral"} - mapping.keys()
    if missing:
        raise ValueError(
            f"Le modèle doit être un NLI à 3 classes (entailment/contradiction/neutral). "
            f"Labels trouvés : {sorted(map(str, id2label.values()))}."
        )
    return [mapping["entailment"], mapping["contradiction"], mapping["neutral"]]


def _warn_truncation(n_truncated: int, max_length: int) -> None:
    if n_truncated:
        warnings.warn(
            f"{n_truncated} paire(s) dépassent {max_length} tokens et ont été tronquées : "
            "la fin du contexte est ignorée par le modèle.",
            stacklevel=3,
        )


class CrossEncoderBackend:
    """Backend PyTorch via sentence-transformers (extra ``krito[torch]``)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from sentence_transformers import CrossEncoder

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CrossEncoder(model_name, device=device)
        if self.model.model is None:
            raise ValueError(f"Impossible de charger le modèle « {model_name} ».")
        self.order = map_label_indices(self.model.model.config.id2label)

    def logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        tok = self.model.tokenizer
        max_length = self.model.max_length or tok.model_max_length
        lengths = [len(ids) for ids in tok([p[0] for p in pairs], [p[1] for p in pairs])["input_ids"]]
        _warn_truncation(sum(n > max_length for n in lengths), max_length)
        raw = self.model.predict([list(p) for p in pairs], apply_softmax=False, show_progress_bar=False)
        return np.asarray(raw, dtype=np.float32)[:, self.order]


class OnnxBackend:
    """Backend ONNX Runtime, sans PyTorch (extra ``krito[onnx]``).

    Le dossier doit contenir un modèle (``model.opt.onnx``, ``model.int8.onnx`` ou ``model.onnx``,
    par ordre de préférence), ``tokenizer.json`` et ``labels.json`` (indices de
    entailment / contradiction / neutral). ``model.opt.onnx`` est le modèle int8 déjà optimisé
    hors ligne : même précision et même vitesse, environ 18 % de RAM en moins au chargement.
    """

    def __init__(self, model_dir: str | os.PathLike, *, max_length: int = 512, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        onnx_path = next((model_dir / n for n in ONNX_MODEL_FILES if (model_dir / n).exists()), None)
        if onnx_path is None:
            raise FileNotFoundError(f"Aucun modèle ONNX ({', '.join(ONNX_MODEL_FILES)}) dans {model_dir}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        # Configuré une fois pour toutes : aucun état n'est modifié pendant les appels,
        # ce qui rend le backend utilisable depuis plusieurs threads (serveur web).
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding()
        self.max_length = max_length
        labels = json.loads((model_dir / "labels.json").read_text())
        self.order = [labels["entailment"], labels["contradiction"], labels["neutral"]]

    def logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        enc = self.tokenizer.encode_batch(pairs)
        # Les parties coupées par la troncature sont conservées dans « overflowing ».
        _warn_truncation(sum(bool(e.overflowing) for e in enc), self.max_length)
        feeds = {
            "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
        }
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        return self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0][:, self.order]


def _resolve_model_dir(model_dir: str | os.PathLike) -> Path:
    """Dossier local tel quel ; sinon identifiant Hugging Face Hub téléchargé dans le cache.

    Seule la meilleure variante de modèle présente dans le dépôt est téléchargée.
    """
    path = Path(model_dir)
    if path.exists() or not isinstance(model_dir, str) or model_dir.count("/") != 1 or model_dir.startswith((".", "/")):
        return path
    from huggingface_hub import list_repo_files, snapshot_download

    available = set(list_repo_files(model_dir))
    model_file = next((f for f in ONNX_MODEL_FILES if f in available), None)
    if model_file is None:
        raise FileNotFoundError(f"Aucun modèle ONNX ({', '.join(ONNX_MODEL_FILES)}) dans le dépôt {model_dir}")
    return Path(snapshot_download(model_dir, allow_patterns=[model_file, "tokenizer.json", "labels.json"]))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class KritoEngine:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        backend: Backend | None = None,
        hypothesis_template: str = "{}",
    ):
        """
        model_name / device : modèle NLI du Hugging Face Hub, chargé avec PyTorch.
        backend : backend déjà construit (ex. ``OnnxBackend``) ; ignore model_name/device.
        hypothesis_template : gabarit appliqué à chaque description d'option,
            ex. ``"Ce message concerne {}."``. Par défaut, la description est utilisée telle quelle.
        """
        if "{}" not in hypothesis_template:
            raise ValueError("hypothesis_template doit contenir « {} ».")
        self.backend = backend if backend is not None else CrossEncoderBackend(model_name, device)
        self.hypothesis_template = hypothesis_template

    @classmethod
    def from_onnx(cls, model_dir: str | os.PathLike, *, threads: int | None = None, **kwargs) -> KritoEngine:
        """Moteur sans PyTorch, sur CPU, adapté au serverless et aux serveurs modestes.

        ``model_dir`` est un dossier local, ou un identifiant du Hugging Face Hub
        (``"organisation/modele"``) : les fichiers sont alors téléchargés puis mis en cache.
        """
        return cls(backend=OnnxBackend(_resolve_model_dir(model_dir), threads=threads), **kwargs)

    def classify(
        self,
        context: str,
        options: dict[str, str],
        *,
        threshold: float | None = None,
        min_margin: float | None = None,
        min_entailment: float | None = None,
        temperature: float = 1.0,
    ) -> DecisionResult:
        """Évalue chaque option face au contexte et renvoie une décision probabiliste.

        Garde-fous (tous optionnels) :
        - threshold : confiance minimale (probabilité relative entre options) ;
        - min_margin : écart minimal entre la 1re et la 2e option ;
        - min_entailment : probabilité d'implication *absolue* minimale de l'option retenue.
          C'est le garde-fou qui permet de rejeter un contexte hors-sujet, car la confiance
          relative reste élevée même quand aucune option ne convient.
        """
        self._validate_input(context, options)

        if temperature <= 0:
            raise ValueError("La température doit être strictement positive.")

        keys = list(options.keys())
        pairs = [(context, self.hypothesis_template.format(options[k])) for k in keys]

        # 1. Inférence : logits bruts [num_options, 3] ordonnés (E, C, N)
        logits = np.asarray(self.backend.logits(pairs), dtype=np.float64)
        ent, con, neu = logits[:, 0], logits[:, 1], logits[:, 2]

        # 2. Logit de décision NLI : Entailment - Contradiction
        decision_scores = ent - con

        # 3. Softmax normalisé entre les options, et implication absolue par option
        probs = _softmax(decision_scores / temperature)
        entail_probs = _softmax(logits, axis=1)[:, 0]

        scores = {
            key: OptionScore(
                entailment=float(ent[i]),
                contradiction=float(con[i]),
                neutral=float(neu[i]),
                decision_score=float(decision_scores[i]),
                probability=float(probs[i]),
                entailment_prob=float(entail_probs[i]),
            )
            for i, key in enumerate(keys)
        }

        # 4. Sélection, confiance et marge (sur les valeurs non arrondies ; ordre stable en cas d'égalité)
        order = np.argsort(-probs, kind="stable")
        selected_key = keys[order[0]]
        confidence = float(probs[order[0]])
        margin = confidence - float(probs[order[1]])

        # 5. Évaluation des garde-fous et motifs de rejet
        rejection_reasons = []
        if threshold is not None and confidence < threshold:
            rejection_reasons.append(f"confidence_too_low (<{threshold})")
        if min_margin is not None and margin < min_margin:
            rejection_reasons.append(f"margin_too_low (<{min_margin})")
        if min_entailment is not None and scores[selected_key].entailment_prob < min_entailment:
            rejection_reasons.append(f"entailment_too_low (<{min_entailment})")

        return DecisionResult(
            selected_key=selected_key,
            selected_label=options[selected_key],
            confidence=confidence,
            margin=margin,
            accepted=not rejection_reasons,
            rejection_reason=" | ".join(rejection_reasons) or None,
            scores=scores,
        )

    @staticmethod
    def _validate_input(context: str, options: dict[str, str]):
        if not isinstance(context, str) or not context.strip():
            raise ValueError("Le contexte doit être une chaîne non vide.")
        if not isinstance(options, dict) or len(options) < 2:
            raise ValueError(
                "Le moteur a besoin d'au moins 2 options pour choisir."
            )
        for key, desc in options.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("Les clés d'options ne peuvent pas être vides.")
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(
                    f"L'option « {key} » a une description vide."
                )
