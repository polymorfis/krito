"""Tests unitaires de KritoEngine, sans charger de vrai modèle.

Un faux backend renvoie des logits contrôlés, ce qui rend les tests rapides et déterministes.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from krito import DecisionResult, KritoEngine, OptionScore
from krito.krito import _warn_truncation, map_label_indices


class FakeBackend:
    """Backend dont on fixe les logits (E, C, N) et qui mémorise les paires reçues."""

    def __init__(self):
        self.next_logits: list[list[float]] = []
        self.last_pairs: list[tuple[str, str]] | None = None

    def logits(self, pairs):
        self.last_pairs = pairs
        assert len(pairs) == len(self.next_logits)
        return np.array(self.next_logits, dtype=np.float32)


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def engine(backend):
    return KritoEngine(backend=backend)


def logits(ent: float, con: float, neu: float = 0.0) -> list[float]:
    return [ent, con, neu]


OPTIONS = {"A": "option a", "B": "option b", "C": "option c"}


# --------------------------------------------------------------------------
# Correspondance des labels du modèle
# --------------------------------------------------------------------------


class TestLabelMapping:
    def test_standard_order(self):
        assert map_label_indices({0: "contradiction", 1: "entailment", 2: "neutral"}) == [1, 0, 2]

    def test_case_and_string_indices(self):
        assert map_label_indices({"0": "ENTAILMENT", "1": "NEUTRAL", "2": "CONTRADICTION"}) == [0, 2, 1]

    def test_generic_labels_are_rejected(self):
        with pytest.raises(ValueError, match="3 classes"):
            map_label_indices({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})

    def test_two_class_model_is_rejected_explicitly(self):
        with pytest.raises(ValueError, match="3 classes"):
            map_label_indices({0: "entailment", 1: "not_entailment"})


# --------------------------------------------------------------------------
# Construction du moteur
# --------------------------------------------------------------------------


class TestInit:
    def test_uses_injected_backend(self, backend):
        assert KritoEngine(backend=backend).backend is backend

    def test_template_must_contain_placeholder(self, backend):
        with pytest.raises(ValueError, match="hypothesis_template"):
            KritoEngine(backend=backend, hypothesis_template="Ce message concerne")

    def test_template_is_applied(self, backend):
        eng = KritoEngine(backend=backend, hypothesis_template="Ce message concerne {}.")
        backend.next_logits = [logits(1, 0), logits(0, 1)]
        eng.classify("ctx", {"A": "la facturation", "B": "la livraison"})
        assert backend.last_pairs == [
            ("ctx", "Ce message concerne la facturation."),
            ("ctx", "Ce message concerne la livraison."),
        ]

    def test_from_onnx_hub_id_is_downloaded(self, tmp_path, monkeypatch):
        pytest.importorskip("huggingface_hub")
        import huggingface_hub

        from krito.krito import _resolve_model_dir

        calls = []
        monkeypatch.setattr(huggingface_hub, "list_repo_files",
                            lambda repo: ["README.md", "model.int8.onnx", "model.opt.onnx", "tokenizer.json", "labels.json"])
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda repo, allow_patterns: calls.append((repo, allow_patterns)) or str(tmp_path))
        assert _resolve_model_dir("polymorfis/krito-nli-fr-multi") == tmp_path
        # Seule la variante préférée est téléchargée
        assert calls == [("polymorfis/krito-nli-fr-multi", ["model.opt.onnx", "tokenizer.json", "labels.json"])]

    def test_local_paths_are_not_downloaded(self, tmp_path):
        from krito.krito import _resolve_model_dir

        assert _resolve_model_dir(tmp_path) == tmp_path
        assert _resolve_model_dir("./modeles/krito") == Path("./modeles/krito")

    def test_from_onnx_missing_dir(self, tmp_path):
        pytest.importorskip("onnxruntime")
        with pytest.raises(FileNotFoundError):
            KritoEngine.from_onnx(tmp_path)


# --------------------------------------------------------------------------
# Validation des entrées
# --------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("context", ["", "   ", None, 42])
    def test_invalid_context(self, engine, context):
        with pytest.raises(ValueError, match="contexte"):
            engine.classify(context, OPTIONS)

    @pytest.mark.parametrize("options", [{}, {"A": "seule"}, None, ["a", "b"]])
    def test_needs_at_least_two_options(self, engine, options):
        with pytest.raises(ValueError, match="2 options"):
            engine.classify("contexte", options)

    @pytest.mark.parametrize("key", ["", "  ", 1])
    def test_invalid_option_key(self, engine, key):
        with pytest.raises(ValueError, match="clés"):
            engine.classify("contexte", {key: "desc", "B": "desc b"})

    @pytest.mark.parametrize("desc", ["", "  ", None])
    def test_empty_option_description(self, engine, desc):
        with pytest.raises(ValueError, match="description vide"):
            engine.classify("contexte", {"A": desc, "B": "desc b"})

    @pytest.mark.parametrize("temperature", [0, -1.0])
    def test_non_positive_temperature(self, engine, backend, temperature):
        backend.next_logits = [logits(1, 0), logits(0, 1)]
        with pytest.raises(ValueError, match="température"):
            engine.classify("contexte", {"A": "a", "B": "b"}, temperature=temperature)


# --------------------------------------------------------------------------
# Logique de décision
# --------------------------------------------------------------------------


class TestDecision:
    def test_pairs_sent_to_backend(self, engine, backend):
        backend.next_logits = [logits(1, 0), logits(0, 0), logits(0, 1)]
        engine.classify("ctx", OPTIONS)
        assert backend.last_pairs == [("ctx", "option a"), ("ctx", "option b"), ("ctx", "option c")]

    def test_selects_highest_entailment_minus_contradiction(self, engine, backend):
        backend.next_logits = [logits(1, 0), logits(3, 0), logits(0, 2)]
        res = engine.classify("ctx", OPTIONS)
        assert isinstance(res, DecisionResult)
        assert res.selected_key == "B"
        assert res.selected_label == "option b"
        assert res.scores["A"].decision_score == pytest.approx(1.0)
        assert res.scores["B"].decision_score == pytest.approx(3.0)
        assert res.scores["C"].decision_score == pytest.approx(-2.0)

    def test_neutral_logit_is_ignored_for_decision(self, engine, backend):
        backend.next_logits = [logits(1, 0, neu=100), logits(2, 0, neu=-100)]
        res = engine.classify("ctx", {"A": "a", "B": "b"})
        assert res.selected_key == "B"
        assert res.scores["A"].neutral == pytest.approx(100.0)

    def test_probabilities_are_softmax_of_decision_scores(self, engine, backend):
        backend.next_logits = [logits(1, 0), logits(2, 0), logits(0, 0)]
        res = engine.classify("ctx", OPTIONS)
        z = [1.0, 2.0, 0.0]
        denom = sum(math.exp(v) for v in z)
        for key, v in zip("ABC", z):
            assert res.scores[key].probability == pytest.approx(math.exp(v) / denom)
        assert sum(s.probability for s in res.scores.values()) == pytest.approx(1.0)

    def test_entailment_prob_is_absolute_per_option(self, engine, backend):
        backend.next_logits = [logits(2, 0, 0), logits(0, 0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"})
        assert res.scores["A"].entailment_prob == pytest.approx(math.exp(2) / (math.exp(2) + 2))
        assert res.scores["B"].entailment_prob == pytest.approx(1 / 3)

    def test_confidence_and_margin(self, engine, backend):
        backend.next_logits = [logits(1, 0), logits(2, 0), logits(0, 0)]
        res = engine.classify("ctx", OPTIONS)
        assert res.confidence == res.scores["B"].probability
        assert res.margin == pytest.approx(res.scores["B"].probability - res.scores["A"].probability)

    def test_temperature_flattens_distribution(self, engine, backend):
        backend.next_logits = [logits(3, 0), logits(0, 0)]
        sharp = engine.classify("ctx", {"A": "a", "B": "b"}, temperature=0.5)
        flat = engine.classify("ctx", {"A": "a", "B": "b"}, temperature=10.0)
        assert sharp.selected_key == flat.selected_key == "A"
        assert sharp.confidence > flat.confidence
        assert flat.confidence == pytest.approx(0.5744, abs=1e-3)

    def test_scores_keep_option_order_and_keys(self, engine, backend):
        backend.next_logits = [logits(0, 0)] * 3
        res = engine.classify("ctx", {"z": "zz", "a": "aa", "m": "mm"})
        assert list(res.scores) == ["z", "a", "m"]
        assert all(isinstance(s, OptionScore) for s in res.scores.values())

    def test_result_is_immutable(self, engine, backend):
        backend.next_logits = [logits(1, 0), logits(0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"})
        with pytest.raises(AttributeError):
            res.selected_key = "B"

    def test_exact_tie_keeps_first_option(self, engine, backend):
        backend.next_logits = [logits(1, 0), logits(1, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"})
        assert res.selected_key == "A"
        assert res.margin == 0.0

    def test_rounding_does_not_change_winner(self, engine, backend):
        # B est strictement meilleure d'un écart minuscule : elle doit gagner.
        backend.next_logits = [logits(1.0, 0), logits(1.0001, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"})
        assert res.selected_key == "B"


# --------------------------------------------------------------------------
# Garde-fous
# --------------------------------------------------------------------------


class TestGuardrails:
    def test_accepted_without_guardrails(self, engine, backend):
        backend.next_logits = [logits(0, 0), logits(0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"})
        assert res.accepted is True
        assert res.rejection_reason is None

    def test_rejected_by_threshold(self, engine, backend):
        backend.next_logits = [logits(0.1, 0), logits(0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"}, threshold=0.9)
        assert res.accepted is False
        assert res.rejection_reason == "confidence_too_low (<0.9)"

    def test_rejected_by_margin(self, engine, backend):
        backend.next_logits = [logits(0.1, 0), logits(0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"}, min_margin=0.5)
        assert res.rejection_reason == "margin_too_low (<0.5)"

    def test_all_reasons_are_reported(self, engine, backend):
        backend.next_logits = [logits(0.1, 0), logits(0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"}, threshold=0.9, min_margin=0.5, min_entailment=0.9)
        assert res.rejection_reason == (
            "confidence_too_low (<0.9) | margin_too_low (<0.5) | entailment_too_low (<0.9)"
        )

    def test_accepted_when_guardrails_satisfied(self, engine, backend):
        backend.next_logits = [logits(5, 0), logits(0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"}, threshold=0.9, min_margin=0.5, min_entailment=0.9)
        assert res.accepted is True

    def test_threshold_boundary_is_inclusive(self, engine, backend):
        backend.next_logits = [logits(0, 0), logits(0, 0)]
        res = engine.classify("ctx", {"A": "a", "B": "b"}, threshold=0.5)
        assert res.accepted is True

    def test_relative_guardrails_accept_when_every_option_is_contradicted(self, engine, backend):
        # Limite connue de la confiance relative : documentée par ce test.
        backend.next_logits = [logits(-6, 6), logits(-2, 3), logits(-6, 6)]
        res = engine.classify("ctx", OPTIONS, threshold=0.65, min_margin=0.3)
        assert res.accepted is True

    def test_min_entailment_rejects_when_every_option_is_contradicted(self, engine, backend):
        backend.next_logits = [logits(-6, 6), logits(-2, 3), logits(-6, 6)]
        res = engine.classify("ctx", OPTIONS, threshold=0.65, min_margin=0.3, min_entailment=0.5)
        assert res.accepted is False
        assert res.rejection_reason == "entailment_too_low (<0.5)"


class TestTruncationWarning:
    def test_warns_when_pairs_are_truncated(self):
        with pytest.warns(UserWarning, match="tronquées"):
            _warn_truncation(2, 512)

    def test_silent_when_nothing_truncated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_truncation(0, 512)
