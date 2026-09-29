from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from .calibration import Calibration, calibrate
from .data import as_examples
from .judge import ConfidenceJudge
from .results import DecisionResult, OptionScore, ScaleResult, YesNoResult, rejection_reasons

DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
ONNX_MODEL_FILES = ("model.opt.onnx", "model.int8.onnx", "model.onnx")


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


def _validate_context(context) -> None:
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Le contexte doit être une chaîne non vide.")


def _validate_options(options) -> None:
    if not isinstance(options, Mapping) or len(options) < 2:
        raise ValueError("Le moteur a besoin d'au moins 2 options pour choisir.")
    for key, desc in options.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Les clés d'options ne peuvent pas être vides.")
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError(f"L'option « {key} » a une description vide.")


def _validate_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError("La température doit être strictement positive.")


def _as_contexts(contexts) -> list[str]:
    if isinstance(contexts, str):
        raise ValueError("Les méthodes *_batch attendent une liste de contextes, pas une chaîne.")
    contexts = list(contexts)
    for context in contexts:
        _validate_context(context)
    return contexts


def _as_levels(levels) -> dict[str, str]:
    """Niveaux d'une échelle : liste de descriptions (clé = description) ou dict ordonné."""
    if isinstance(levels, Mapping):
        out = dict(levels)
    elif isinstance(levels, Sequence) and not isinstance(levels, str):
        out = {level: level for level in levels}
        if len(out) != len(levels):
            raise ValueError("Les niveaux de l'échelle doivent être distincts.")
    else:
        raise ValueError("Les niveaux doivent être une liste ordonnée de descriptions ou un dict ordonné.")
    _validate_options(out)
    return out


class KritoEngine:
    """Moteur de décision zero-shot : choix unique, oui/non et échelle ordonnée.

    Toutes les primitives reposent sur le même modèle NLI et existent en version unitaire
    (``classify``, ``yes_no``, ``scale``) et par lots (``*_batch``), plus rapide pour traiter
    beaucoup de textes : les paires sont regroupées en appels au modèle de ``batch_size`` paires.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        *,
        backend: Backend | None = None,
        hypothesis_template: str = "{}",
        judge: ConfidenceJudge | None = None,
    ):
        """
        model_name / device : modèle NLI du Hugging Face Hub, chargé avec PyTorch.
        backend : backend déjà construit (ex. ``OnnxBackend``) ; ignore model_name/device.
        hypothesis_template : gabarit appliqué à chaque description d'option,
            ex. ``"Ce message concerne {}."``. Par défaut, la description est utilisée telle quelle.
        judge : juge de confiance optionnel (``ConfidenceJudge``), entraîné sur vos données.
            Chaque décision reçoit alors un ``judge_score``, utilisable avec ``min_judge_score``.
        """
        if "{}" not in hypothesis_template:
            raise ValueError("hypothesis_template doit contenir « {} ».")
        self.backend = backend if backend is not None else CrossEncoderBackend(model_name, device)
        self.hypothesis_template = hypothesis_template
        self.judge = judge

    @classmethod
    def from_onnx(cls, model_dir: str | os.PathLike, *, threads: int | None = None, **kwargs) -> KritoEngine:
        """Moteur sans PyTorch, sur CPU, adapté au serverless et aux serveurs modestes.

        ``model_dir`` est un dossier local, ou un identifiant du Hugging Face Hub
        (``"organisation/modele"``) : les fichiers sont alors téléchargés puis mis en cache.
        """
        return cls(backend=OnnxBackend(_resolve_model_dir(model_dir), threads=threads), **kwargs)

    # ------------------------------------------------------------------ inférence

    def _infer(self, contexts: list[str], hypotheses: list[str], batch_size: int) -> np.ndarray:
        """Logits [n_contextes, n_hypothèses, 3] (E, C, N), par appels de ``batch_size`` paires.

        Les contextes sont regroupés par longueur : un lot de textes de tailles voisines
        demande moins de remplissage (padding), donc moins de calcul.
        """
        if batch_size < 1:
            raise ValueError("batch_size doit être au moins 1.")
        n_h = len(hypotheses)
        order = sorted(range(len(contexts)), key=lambda i: len(contexts[i]))
        pairs = [(i, j) for i in order for j in range(n_h)]
        out = np.empty((len(contexts), n_h, 3), dtype=np.float64)
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            logits = np.asarray(self.backend.logits([(contexts[i], hypotheses[j]) for i, j in chunk]),
                                dtype=np.float64)
            for (i, j), row in zip(chunk, logits):
                out[i, j] = row
        return out

    # ------------------------------------------------------------------ choix unique

    def classify(
        self,
        context: str,
        options: dict[str, str],
        *,
        threshold: float | None = None,
        min_margin: float | None = None,
        min_entailment: float | None = None,
        min_judge_score: float | None = None,
        temperature: float = 1.0,
    ) -> DecisionResult:
        """Évalue chaque option face au contexte et renvoie une décision probabiliste.

        Garde-fous (tous optionnels) :
        - threshold : confiance minimale (probabilité relative entre options) ;
        - min_margin : écart minimal entre la 1re et la 2e option ;
        - min_entailment : probabilité d'implication *absolue* minimale de l'option retenue.
          C'est le garde-fou qui permet de rejeter un contexte hors-sujet, car la confiance
          relative reste élevée même quand aucune option ne convient ;
        - min_judge_score : probabilité minimale que la décision soit bonne selon le juge
          de confiance (demande ``judge``).
        """
        _validate_context(context)
        return self.classify_batch(
            [context], options, threshold=threshold, min_margin=min_margin, min_entailment=min_entailment,
            min_judge_score=min_judge_score, temperature=temperature,
        )[0]

    def classify_batch(
        self,
        contexts: Iterable[str],
        options: dict[str, str],
        *,
        threshold: float | None = None,
        min_margin: float | None = None,
        min_entailment: float | None = None,
        min_judge_score: float | None = None,
        temperature: float = 1.0,
        batch_size: int = 32,
    ) -> list[DecisionResult]:
        """``classify`` sur plusieurs contextes avec les mêmes options, dans l'ordre reçu.

        batch_size : nombre de paires (contexte, option) envoyées au modèle par appel.
        L'augmenter accélère le traitement sur GPU ; sur CPU, 16 à 64 est un bon compromis
        entre débit et mémoire.
        """
        contexts = _as_contexts(contexts)
        _validate_options(options)
        _validate_temperature(temperature)
        if min_judge_score is not None and self.judge is None:
            raise ValueError("min_judge_score demande un juge de confiance (paramètre judge).")
        keys = list(options)
        logits = self._infer(contexts, [self.hypothesis_template.format(options[k]) for k in keys], batch_size)
        limits = dict(threshold=threshold, min_margin=min_margin, min_entailment=min_entailment,
                      min_judge_score=min_judge_score)
        return [self._decide(lg, keys, options, limits, temperature) for lg in logits]

    def _decide(self, logits: np.ndarray, keys: list[str], options: Mapping[str, str],
                limits: dict[str, float | None], temperature: float) -> DecisionResult:
        ent, con, neu = logits[:, 0], logits[:, 1], logits[:, 2]

        # Logit de décision NLI : Entailment − Contradiction ; softmax entre les options,
        # et implication absolue par option (softmax sur E, C, N).
        decision_scores = ent - con
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

        # Sélection, confiance et marge (valeurs non arrondies ; ordre stable en cas d'égalité)
        order = np.argsort(-probs, kind="stable")
        selected = keys[order[0]]
        confidence = float(probs[order[0]])
        result = DecisionResult(
            selected_key=selected,
            selected_label=options[selected],
            confidence=confidence,
            margin=confidence - float(probs[order[1]]),
            accepted=True,
            rejection_reason=None,
            scores=scores,
        )
        judge_score = self.judge.score(result) if self.judge is not None else None
        values = dict(threshold=result.confidence, min_margin=result.margin,
                      min_entailment=scores[selected].entailment_prob, min_judge_score=judge_score)
        reasons = rejection_reasons(values, limits)
        return replace(result, accepted=not reasons, rejection_reason=" | ".join(reasons) or None,
                       judge_score=judge_score)

    # ------------------------------------------------------------------ oui / non

    def yes_no(
        self,
        context: str,
        statement: str,
        *,
        threshold: float | None = None,
        max_neutral: float | None = None,
        temperature: float = 1.0,
    ) -> YesNoResult:
        """Répond oui ou non : le contexte implique-t-il l'affirmation ?

        La question se pose comme une affirmation complète, utilisée telle quelle (sans
        ``hypothesis_template``) : « Le client demande un remboursement. »
        ``probability`` = P(oui), calculée entre implication et contradiction.

        Garde-fous (optionnels) :
        - threshold : confiance minimale dans la réponse donnée (oui ou non) ;
        - max_neutral : probabilité « neutre » maximale. Un texte qui ne dit rien sur
          l'affirmation est neutre : sans ce garde-fou, la réponse y est arbitraire.
        """
        _validate_context(context)
        return self.yes_no_batch([context], statement, threshold=threshold, max_neutral=max_neutral,
                                 temperature=temperature)[0]

    def yes_no_batch(
        self,
        contexts: Iterable[str],
        statement: str,
        *,
        threshold: float | None = None,
        max_neutral: float | None = None,
        temperature: float = 1.0,
        batch_size: int = 32,
    ) -> list[YesNoResult]:
        """``yes_no`` sur plusieurs contextes pour la même affirmation, dans l'ordre reçu."""
        contexts = _as_contexts(contexts)
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("L'affirmation doit être une chaîne non vide.")
        _validate_temperature(temperature)
        results = []
        for ent, con, neu in self._infer(contexts, [statement], batch_size)[:, 0]:
            p_yes = float(1 / (1 + np.exp(-(ent - con) / temperature)))
            neutral_prob = float(_softmax(np.array([ent, con, neu]))[2])
            confidence = max(p_yes, 1 - p_yes)
            reasons = rejection_reasons(dict(threshold=confidence, max_neutral=neutral_prob),
                                        dict(threshold=threshold, max_neutral=max_neutral))
            results.append(YesNoResult(
                statement=statement, answer=p_yes >= 0.5, probability=p_yes, confidence=confidence,
                neutral_prob=neutral_prob, accepted=not reasons, rejection_reason=" | ".join(reasons) or None,
                entailment=float(ent), contradiction=float(con), neutral=float(neu),
            ))
        return results

    # ------------------------------------------------------------------ échelle ordonnée

    def scale(
        self,
        context: str,
        levels: Sequence[str] | Mapping[str, str],
        *,
        template: str | None = None,
        threshold: float | None = None,
        min_margin: float | None = None,
        min_entailment: float | None = None,
        min_judge_score: float | None = None,
        max_spread: float | None = None,
        temperature: float = 1.0,
    ) -> ScaleResult:
        """Situe le contexte sur une échelle ordonnée (urgence, satisfaction, gravité…).

        levels : niveaux **dans l'ordre**, du plus bas au plus haut : liste de descriptions,
            ou dict ``{clé: description}`` ordonné.
        template : gabarit propre à l'échelle, ex. ``"Le ton de ce message est {}."`` ;
            par défaut, celui du moteur.

        En plus d'une décision (niveau le plus probable, confiance, marge), le résultat donne
        ``expected``, le rang moyen pondéré par les probabilités, et ``spread``, l'écart-type en
        niveaux. Sur une échelle, hésiter entre deux niveaux voisins n'est pas grave :
        ``max_spread`` rejette seulement les distributions vraiment étalées.
        """
        _validate_context(context)
        return self.scale_batch(
            [context], levels, template=template, threshold=threshold, min_margin=min_margin,
            min_entailment=min_entailment, min_judge_score=min_judge_score, max_spread=max_spread,
            temperature=temperature,
        )[0]

    def scale_batch(
        self,
        contexts: Iterable[str],
        levels: Sequence[str] | Mapping[str, str],
        *,
        template: str | None = None,
        threshold: float | None = None,
        min_margin: float | None = None,
        min_entailment: float | None = None,
        min_judge_score: float | None = None,
        max_spread: float | None = None,
        temperature: float = 1.0,
        batch_size: int = 32,
    ) -> list[ScaleResult]:
        """``scale`` sur plusieurs contextes avec la même échelle, dans l'ordre reçu."""
        options = _as_levels(levels)
        template = self.hypothesis_template if template is None else template
        if "{}" not in template:
            raise ValueError("template doit contenir « {} ».")
        contexts = _as_contexts(contexts)
        _validate_temperature(temperature)
        if min_judge_score is not None and self.judge is None:
            raise ValueError("min_judge_score demande un juge de confiance (paramètre judge).")
        keys = list(options)
        logits = self._infer(contexts, [template.format(options[k]) for k in keys], batch_size)
        limits = dict(threshold=threshold, min_margin=min_margin, min_entailment=min_entailment,
                      min_judge_score=min_judge_score)
        ranks = np.arange(len(keys), dtype=np.float64)
        results = []
        for lg in logits:
            decision = self._decide(lg, keys, options, limits, temperature)
            probs = np.array([decision.scores[k].probability for k in keys])
            expected = float(probs @ ranks)
            spread = float(np.sqrt(probs @ (ranks - expected) ** 2))
            reasons = [decision.rejection_reason] if decision.rejection_reason else []
            reasons += rejection_reasons(dict(max_spread=spread), dict(max_spread=max_spread))
            base = {f.name: getattr(decision, f.name) for f in fields(DecisionResult)}
            base.update(accepted=not reasons, rejection_reason=" | ".join(reasons) or None)
            results.append(ScaleResult(**base, index=keys.index(decision.selected_key),
                                       expected=expected, spread=spread))
        return results

    # ------------------------------------------------------------------ calibration et juge

    def _annotated_results(self, examples, options: dict[str, str], temperature: float, batch_size: int):
        examples = as_examples(examples)
        unknown = {ex.label for ex in examples if ex.label is not None} - set(options)
        if unknown:
            raise ValueError(f"Étiquettes absentes des options : {sorted(unknown)}. "
                             "Utilisez None, « » ou « none » pour les hors-sujet.")
        results = self.classify_batch([ex.text for ex in examples], options,
                                      temperature=temperature, batch_size=batch_size)
        return results, [ex.label for ex in examples]

    def calibrate(
        self,
        examples,
        options: dict[str, str],
        *,
        target_precision: float = 0.95,
        guardrails: Sequence[str] | None = None,
        temperature: float = 1.0,
        batch_size: int = 32,
        grid_size: int = 20,
    ) -> Calibration:
        """Calibre les garde-fous sur des exemples annotés, pour une précision cible.

        examples : couples ``(texte, bonne_option)``, ``None`` ou ``"none"`` pour les hors-sujet
            (voir ``krito.data.load_examples``). Incluez des hors-sujet : sans eux, le seuil
            ``min_entailment`` ne peut pas être mesuré.
        guardrails : garde-fous à régler. Par défaut ``("min_judge_score",)`` si le moteur a un
            juge, sinon ``("min_entailment", "min_margin")``.

        Renvoie une ``Calibration`` : ``engine.classify(texte, options, **cal.guardrails)``.
        """
        results, labels = self._annotated_results(examples, options, temperature, batch_size)
        if guardrails is None:
            guardrails = ("min_judge_score",) if self.judge is not None else ("min_entailment", "min_margin")
        return calibrate(results, labels, target_precision=target_precision, guardrails=guardrails,
                         grid_size=grid_size)

    def fit_judge(
        self, examples, options: dict[str, str], *, C: float = 1.0, temperature: float = 1.0, batch_size: int = 32
    ) -> ConfidenceJudge:
        """Entraîne un juge de confiance sur des exemples annotés et l'attache au moteur.

        Les exemples doivent contenir des erreurs du modèle et des hors-sujet, sinon il n'y a
        rien à apprendre. Calibrez ensuite ``min_judge_score`` sur d'**autres** exemples.
        """
        results, labels = self._annotated_results(examples, options, temperature, batch_size)
        self.judge = ConfidenceJudge(C).fit(results, labels)
        return self.judge

    # ------------------------------------------------------------------ compatibilité

    @staticmethod
    def _validate_input(context: str, options: dict[str, str]):
        _validate_context(context)
        _validate_options(options)
