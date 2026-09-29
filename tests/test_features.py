"""Tests unitaires des fonctionnalités v0.3 : lots, oui/non, échelle, calibration, juge,
mode supervisé et exemples annotés. Aucun vrai modèle n'est chargé."""

from __future__ import annotations

import importlib.util
import json
import math
import warnings
import zlib

import numpy as np
import pytest

from krito import (
    Calibration,
    ClassScore,
    ConfidenceJudge,
    DecisionResult,
    Example,
    KritoClassifier,
    KritoEngine,
    OptionScore,
    ScaleResult,
    YesNoResult,
    calibrate,
    evaluate_guardrails,
    load_examples,
    split_examples,
)
from krito.data import as_examples


requires_sklearn = pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,
                                      reason='demande scikit-learn : pip install "krito[learn]"')


class TableBackend:
    """Backend dont les logits dépendent du contenu de la paire, pas de sa position.

    ``table[(contexte, hypothèse)] = (E, C, N)`` ; les paires inconnues valent (0, 0, 0).
    """

    def __init__(self, table=None):
        self.table = table or {}
        self.calls: list[list[tuple[str, str]]] = []

    def logits(self, pairs):
        self.calls.append(list(pairs))
        return np.array([self.table.get(p, (0.0, 0.0, 0.0)) for p in pairs], dtype=np.float32)


OPTIONS = {"billing": "la facturation", "shipping": "la livraison", "account": "le compte"}


def support_table() -> dict:
    """Contextes « facturation », « livraison », « compte » et hors-sujet, bien séparés."""
    table = {}
    for i in range(20):
        for key, hyp in OPTIONS.items():
            for other in OPTIONS:
                ctx = f"{other} {i}"
                table[(ctx, hyp)] = (4.0, -2.0, 0.0) if other == key else (-2.0, 3.0, 0.0)
            # Hors-sujet : aucune option n'est impliquée, la 1re l'emporte de peu
            table[(f"hors {i}", hyp)] = (-1.0 + (0.3 if key == "billing" else 0.0), 1.0, 3.0)
            # Ambigus : facturation et livraison proches, avec la bonne réponse « shipping »
            table[(f"ambigu {i}", hyp)] = {"billing": (2.0, 0.0, 0.0), "shipping": (1.8, 0.0, 0.0)}.get(
                key, (-2.0, 3.0, 0.0))
    return table


def support_examples() -> list[tuple[str, str | None]]:
    ex = [(f"{k} {i}", k) for i in range(20) for k in OPTIONS]
    ex += [(f"hors {i}", None) for i in range(20)]
    ex += [(f"ambigu {i}", "shipping") for i in range(20)]
    return ex


# --------------------------------------------------------------------------
# Traitement par lots
# --------------------------------------------------------------------------


class TestBatch:
    def test_batch_matches_single_calls(self):
        backend = TableBackend(support_table())
        engine = KritoEngine(backend=backend)
        texts = ["billing 1", "hors 3", "shipping 2", "account 0", "ambigu 5"]
        batch = engine.classify_batch(texts, OPTIONS, min_entailment=0.5)
        single = [engine.classify(t, OPTIONS, min_entailment=0.5) for t in texts]
        assert batch == single
        assert [r.selected_key for r in batch[:4:2]] == ["billing", "shipping"]

    def test_batch_size_controls_backend_calls(self):
        backend = TableBackend(support_table())
        engine = KritoEngine(backend=backend)
        engine.classify_batch([f"billing {i}" for i in range(10)], OPTIONS, batch_size=7)
        assert [len(c) for c in backend.calls] == [7, 7, 7, 7, 2]

    def test_contexts_grouped_by_length_but_order_preserved(self):
        backend = TableBackend(support_table())
        engine = KritoEngine(backend=backend)
        texts = ["shipping 12", "account 1", "billing 3"]
        res = engine.classify_batch(texts, OPTIONS, batch_size=100)
        assert [r.selected_key for r in res] == ["shipping", "account", "billing"]
        sent = [p[0] for p in backend.calls[0]]
        assert sent.index("shipping 12") > sent.index("account 1")  # le plus long en dernier

    def test_empty_batch(self):
        assert KritoEngine(backend=TableBackend()).classify_batch([], OPTIONS) == []

    def test_batch_validation(self):
        engine = KritoEngine(backend=TableBackend())
        with pytest.raises(ValueError, match="liste"):
            engine.classify_batch("un seul texte", OPTIONS)
        with pytest.raises(ValueError, match="contexte"):
            engine.classify_batch(["ok", ""], OPTIONS)
        with pytest.raises(ValueError, match="batch_size"):
            engine.classify_batch(["ok"], OPTIONS, batch_size=0)

    def test_min_judge_score_requires_judge(self):
        engine = KritoEngine(backend=TableBackend())
        with pytest.raises(ValueError, match="juge"):
            engine.classify("ok", OPTIONS, min_judge_score=0.5)


# --------------------------------------------------------------------------
# Oui / non
# --------------------------------------------------------------------------


class TestYesNo:
    STATEMENT = "Le client demande un remboursement."

    def engine(self, e, c, n=0.0):
        return KritoEngine(backend=TableBackend({("ctx", self.STATEMENT): (e, c, n)}),
                           hypothesis_template="Ce message concerne {}.")

    def test_yes(self):
        r = self.engine(3.0, -1.0).yes_no("ctx", self.STATEMENT)
        assert isinstance(r, YesNoResult)
        assert r.answer is True
        assert r.probability == pytest.approx(1 / (1 + math.exp(-4)))
        assert r.confidence == pytest.approx(r.probability)
        assert r.accepted and r.rejection_reason is None

    def test_no(self):
        r = self.engine(-1.0, 2.0).yes_no("ctx", self.STATEMENT)
        assert r.answer is False
        assert r.confidence == pytest.approx(1 - r.probability)

    def test_statement_is_used_verbatim(self):
        engine = self.engine(1.0, 0.0)
        engine.yes_no("ctx", self.STATEMENT)
        assert engine.backend.calls[0] == [("ctx", self.STATEMENT)]

    def test_neutral_guardrail(self):
        r = self.engine(0.5, 0.0, 4.0).yes_no("ctx", self.STATEMENT, max_neutral=0.5)
        assert r.neutral_prob == pytest.approx(math.exp(4) / (math.exp(0.5) + 1 + math.exp(4)))
        assert r.accepted is False
        assert r.rejection_reason == "neutral_too_high (>0.5)"

    def test_threshold_guardrail(self):
        r = self.engine(0.2, 0.0).yes_no("ctx", self.STATEMENT, threshold=0.9, max_neutral=0.99)
        assert r.rejection_reason == "confidence_too_low (<0.9)"

    def test_temperature(self):
        sharp = self.engine(1.0, 0.0).yes_no("ctx", self.STATEMENT, temperature=0.5)
        flat = self.engine(1.0, 0.0).yes_no("ctx", self.STATEMENT, temperature=5.0)
        assert sharp.probability > flat.probability > 0.5

    def test_batch(self):
        backend = TableBackend({("a", "S."): (3, 0, 0), ("b", "S."): (0, 3, 0)})
        res = KritoEngine(backend=backend).yes_no_batch(["a", "b"], "S.")
        assert [r.answer for r in res] == [True, False]

    def test_validation(self):
        engine = self.engine(0, 0)
        with pytest.raises(ValueError, match="affirmation"):
            engine.yes_no("ctx", "  ")
        with pytest.raises(ValueError, match="température"):
            engine.yes_no("ctx", self.STATEMENT, temperature=0)


# --------------------------------------------------------------------------
# Échelle ordonnée
# --------------------------------------------------------------------------


LEVELS = ["pas urgent", "assez urgent", "très urgent"]


def scale_engine(values, template="{}"):
    backend = TableBackend({("ctx", template.format(lvl)): (v, 0.0, 0.0) for lvl, v in zip(LEVELS, values)})
    return KritoEngine(backend=backend, hypothesis_template="Ce message concerne {}.")


class TestScale:
    def test_selects_level_and_computes_position(self):
        r = scale_engine([0.0, 1.0, 3.0]).scale("ctx", LEVELS, template="{}")
        assert isinstance(r, ScaleResult) and isinstance(r, DecisionResult)
        assert r.selected_key == "très urgent" and r.index == 2
        p = np.exp([0.0, 1.0, 3.0]) / np.exp([0.0, 1.0, 3.0]).sum()
        assert r.expected == pytest.approx(p @ [0, 1, 2])
        assert r.spread == pytest.approx(math.sqrt(p @ (np.arange(3) - r.expected) ** 2))

    def test_uses_engine_template_by_default(self):
        engine = scale_engine([0, 0, 0], template="Ce message concerne {}.")
        engine.scale("ctx", LEVELS)
        assert engine.backend.calls[0][0] == ("ctx", "Ce message concerne pas urgent.")

    def test_custom_template(self):
        engine = scale_engine([3.0, 0, 0], template="Ce message est {}.")
        r = engine.scale("ctx", LEVELS, template="Ce message est {}.")
        assert r.index == 0

    def test_dict_levels_keep_order(self):
        engine = KritoEngine(backend=TableBackend({("ctx", "élevée"): (5, 0, 0)}))
        r = engine.scale("ctx", {"low": "faible", "mid": "moyenne", "high": "élevée"})
        assert r.selected_key == "high" and r.selected_label == "élevée" and r.index == 2

    def test_spread_guardrail(self):
        # Hésitation entre les deux extrêmes : distribution étalée
        r = scale_engine([2.0, -5.0, 2.0]).scale("ctx", LEVELS, template="{}", max_spread=0.5)
        assert r.spread == pytest.approx(1.0, abs=1e-3)
        assert r.rejection_reason == "spread_too_high (>0.5)"
        # Hésitation entre voisins : acceptée
        r = scale_engine([-5.0, 2.0, 2.0]).scale("ctx", LEVELS, template="{}", max_spread=0.6)
        assert r.accepted

    def test_combines_reasons(self):
        r = scale_engine([2.0, -5.0, 2.0]).scale("ctx", LEVELS, template="{}", threshold=0.9, max_spread=0.5)
        assert r.rejection_reason == "confidence_too_low (<0.9) | spread_too_high (>0.5)"

    def test_validation(self):
        engine = scale_engine([0, 0, 0])
        with pytest.raises(ValueError, match="distincts"):
            engine.scale("ctx", ["a", "a"])
        with pytest.raises(ValueError, match="liste ordonnée"):
            engine.scale("ctx", "abc")
        with pytest.raises(ValueError, match="2 options"):
            engine.scale("ctx", ["seul"])
        with pytest.raises(ValueError, match="template"):
            engine.scale("ctx", LEVELS, template="sans marqueur")

    def test_batch(self):
        engine = scale_engine([0.0, 0.0, 4.0], template="{}")
        res = engine.scale_batch(["ctx", "autre"], LEVELS, template="{}")
        assert [r.index for r in res] == [2, 0]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


class TestCalibration:
    def test_finds_guardrails_rejecting_off_topic_and_ambiguous(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        cal = engine.calibrate(support_examples(), OPTIONS, target_precision=0.95)
        assert isinstance(cal, Calibration)
        assert cal.reached
        assert cal.precision == 1.0
        # 60 bonnes décisions possibles (les ambigus sont mal classés) sur 80 exemples du domaine
        assert cal.automated == pytest.approx(60 / 80)
        assert cal.n_examples == 100 and cal.n_in_domain == 80
        assert set(cal.guardrails) <= {"min_entailment", "min_margin"}
        # Les seuils recommandés s'appliquent tels quels
        res = engine.classify_batch([t for t, _ in support_examples()], OPTIONS, **cal.guardrails)
        assert sum(r.accepted for r in res) == 60

    def test_baseline_without_guardrails(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        results = engine.classify_batch([t for t, _ in support_examples()], OPTIONS)
        labels = [lab for _, lab in support_examples()]
        p = evaluate_guardrails(results, labels)
        assert p.coverage == 1.0 and p.precision == pytest.approx(60 / 100)
        p = evaluate_guardrails(results, labels, min_entailment=0.5)
        assert p.accepted == 80 and p.precision == pytest.approx(60 / 80)

    def test_unreachable_target_returns_best_precision(self):
        results = [_result("a", 0.9, 0.9), _result("a", 0.8, 0.8)]
        cal = calibrate(results, ["b", "b"], target_precision=0.95, guardrails=["threshold"])
        assert not cal.reached
        assert "non atteinte" in str(cal)

    def test_max_guardrail_direction(self):
        results = [_scale(0.2), _scale(0.3), _scale(1.2)]
        cal = calibrate(results, ["x", "x", "y"], target_precision=1.0, guardrails=["max_spread"])
        assert cal.reached and cal.guardrails["max_spread"] >= 0.3 and cal.automated == pytest.approx(2 / 3)

    def test_str_lists_frontier(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        cal = engine.calibrate(support_examples(), OPTIONS)
        text = str(cal)
        assert all(name in text for name in cal.guardrails)
        assert "Recommandé" in text and "◄" in text and "100 exemples annotés, dont 20 hors-sujet" in text

    def test_unknown_labels_are_rejected(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        with pytest.raises(ValueError, match="absentes des options"):
            engine.calibrate([("billing 1", "factu")], OPTIONS)

    def test_unknown_guardrail(self):
        with pytest.raises(ValueError, match="inconnu"):
            calibrate([_result("a", 0.9, 0.5)], ["a"], guardrails=["min_foo"])

    def test_signal_requires_matching_model(self):
        with pytest.raises(ValueError, match="KritoClassifier"):
            calibrate([_result("a", 0.9, 0.5)], ["a"], guardrails=["min_similarity"])


def _result(key, confidence, entail) -> DecisionResult:
    scores = {k: OptionScore(0, 0, 0, 0, confidence if k == key else 1 - confidence, entail) for k in ("a", "b")}
    return DecisionResult(key, key, confidence, 2 * confidence - 1, True, None, scores)


def _scale(spread) -> ScaleResult:
    base = _result("x", 0.6, 0.5)
    scores = {"x": base.scores["a"], "y": base.scores["b"]}
    return ScaleResult(selected_key="x", selected_label="x", confidence=0.6, margin=0.2, accepted=True,
                       rejection_reason=None, scores=scores, index=0, expected=0.4, spread=spread)


# --------------------------------------------------------------------------
# Juge de confiance
# --------------------------------------------------------------------------


@requires_sklearn
class TestJudge:
    def test_fit_attach_and_guardrail(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        judge = engine.fit_judge(support_examples(), OPTIONS)
        assert engine.judge is judge
        good = engine.classify("billing 3", OPTIONS, min_judge_score=0.5)
        off = engine.classify("hors 3", OPTIONS, min_judge_score=0.5)
        ambiguous = engine.classify("ambigu 3", OPTIONS, min_judge_score=0.5)
        assert good.judge_score > 0.9 and good.accepted
        assert off.judge_score < 0.1 and off.rejection_reason == "judge_score_too_low (<0.5)"
        assert ambiguous.judge_score < 0.5

    def test_calibrate_uses_judge_by_default(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        engine.fit_judge(support_examples(), OPTIONS)
        cal = engine.calibrate(support_examples(), OPTIONS)
        assert list(cal.guardrails) == ["min_judge_score"]
        assert cal.precision == 1.0 and cal.automated == pytest.approx(0.75)

    def test_needs_good_and_bad_decisions(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        with pytest.raises(ValueError, match="bonnes ET de mauvaises"):
            engine.fit_judge([("billing 1", "billing"), ("shipping 2", "shipping")], OPTIONS)

    def test_save_load_roundtrip(self, tmp_path):
        engine = KritoEngine(backend=TableBackend(support_table()))
        judge = engine.fit_judge(support_examples(), OPTIONS)
        judge.save(tmp_path / "juge.json")
        loaded = ConfidenceJudge.load(tmp_path / "juge.json")
        r = engine.classify("ambigu 1", OPTIONS)
        assert loaded.score(r) == pytest.approx(judge.score(r))
        assert json.loads((tmp_path / "juge.json").read_text())["kind"] == "nli"
        # Utilisable directement au chargement du moteur
        other = KritoEngine(backend=TableBackend(support_table()), judge=loaded)
        assert other.classify("ambigu 1", OPTIONS).judge_score == pytest.approx(r.judge_score)

    def test_rejects_other_model_kind(self):
        engine = KritoEngine(backend=TableBackend(support_table()))
        judge = engine.fit_judge(support_examples(), OPTIONS)
        sup = DecisionResult("a", "a", 0.9, 0.8, True, None,
                             {"a": ClassScore(1, 0.9, 0.8), "b": ClassScore(0, 0.1, 0.2)})
        with pytest.raises(ValueError, match="entraîné sur des décisions"):
            judge.score(sup)

    def test_unfitted(self):
        with pytest.raises(RuntimeError):
            ConfidenceJudge().score(_result("a", 0.9, 0.9))


# --------------------------------------------------------------------------
# Mode supervisé
# --------------------------------------------------------------------------


class FakeEmbedder:
    """Embeddings déterministes : une direction par mot-clé, plus un bruit propre au texte."""

    KEYWORDS = ("facture", "colis", "mot de passe", "recette")

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            rng = np.random.default_rng(zlib.crc32(t.encode()))
            v = rng.normal(scale=0.35, size=8)
            for i, kw in enumerate(self.KEYWORDS):
                if kw in t:
                    v[i] += 2.0
            out.append(v)
        return np.array(out)


def supervised_examples():
    ex = {
        "billing": [f"ma facture numéro {i} est fausse" for i in range(12)],
        "shipping": [f"mon colis {i} est en retard" for i in range(12)],
        "account": [f"mot de passe perdu, tentative {i}" for i in range(12)],
        None: [f"une recette de cuisine {i}" for i in range(10)],
    }
    # Quelques exemples ambigus, pour que le juge voie des erreurs
    ex["billing"] += [f"colis et facture {i}" for i in range(4)]
    ex["shipping"] += [f"facture du colis {i}" for i in range(4)]
    return ex


@requires_sklearn
class TestClassifier:
    def test_fit_and_classify(self):
        emb = FakeEmbedder()
        clf = KritoClassifier(emb, labels={"billing": "la facturation"}).fit(supervised_examples())
        assert clf.classes_ == ["billing", "shipping", "account"]
        r = clf.classify("ma facture est trop élevée")
        assert r.selected_key == "billing" and r.selected_label == "la facturation"
        assert isinstance(r.scores["billing"], ClassScore)
        assert sum(s.probability for s in r.scores.values()) == pytest.approx(1.0)
        assert clf.classify("mon colis n'arrive pas").selected_label == "shipping"
        assert clf.judge is not None and r.judge_score is not None

    def test_batch_calls_embedder_once(self):
        emb = FakeEmbedder()
        clf = KritoClassifier(emb).fit(supervised_examples(), judge=False)
        emb.calls = 0
        res = clf.classify_batch(["facture", "colis", "mot de passe"])
        assert emb.calls == 1
        assert [r.selected_key for r in res] == ["billing", "shipping", "account"]

    def test_off_topic_rejected_by_similarity_and_judge(self):
        clf = KritoClassifier(FakeEmbedder()).fit(supervised_examples())
        on = clf.classify("ma facture numéro 99 est fausse")
        off = clf.classify("une recette de gâteau au chocolat")
        assert on.scores[on.selected_key].similarity > off.scores[off.selected_key].similarity
        assert on.judge_score > off.judge_score
        assert not clf.classify("une recette de gâteau au chocolat", min_judge_score=0.5).accepted

    def test_calibrate_on_held_out(self):
        train, test = split_examples(supervised_examples(), 0.4, seed=1)
        clf = KritoClassifier(FakeEmbedder()).fit(train)
        cal = clf.calibrate(test, target_precision=0.9)
        assert list(cal.guardrails) == ["min_judge_score"]
        cal = clf.calibrate(test, guardrails=["min_similarity", "min_margin"], target_precision=0.9)
        assert cal.reached

    def test_save_load(self, tmp_path):
        emb = FakeEmbedder()
        clf = KritoClassifier(emb, labels={"billing": "la facturation"}).fit(supervised_examples())
        clf.save(tmp_path / "clf.npz")
        with pytest.raises(ValueError, match="embedder personnalisé"):
            KritoClassifier.load(tmp_path / "clf.npz")
        loaded = KritoClassifier.load(tmp_path / "clf.npz", embedder=emb)
        texts = ["facture du colis 99", "une recette"]
        assert loaded.classify_batch(texts) == clf.classify_batch(texts)

    def test_validation(self):
        clf = KritoClassifier(FakeEmbedder())
        with pytest.raises(RuntimeError, match="entraîné"):
            clf.classify("x")
        with pytest.raises(ValueError, match="2 catégories"):
            clf.fit([("facture", "billing"), ("recette", None)])
        with pytest.raises(ValueError, match="sans exemple"):
            KritoClassifier(FakeEmbedder(), labels={"sales": "ventes"}).fit(supervised_examples())
        clf.fit(supervised_examples(), judge=False)
        with pytest.raises(ValueError, match="juge"):
            clf.classify("facture", min_judge_score=0.5)
        with pytest.raises(ValueError, match="Étiquettes inconnues"):
            clf.calibrate([("facture", "sales")])

    def test_single_off_topic_example_is_enough_for_judge(self):
        ex = supervised_examples()
        ex[None] = ex[None][:1]
        assert KritoClassifier(FakeEmbedder()).fit(ex).judge is not None

    def test_judge_disabled_when_too_few_examples(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clf = KritoClassifier(FakeEmbedder()).fit([("facture", "billing"), ("colis", "shipping")])
        assert clf.judge is None
        assert any("juge" in str(w.message) for w in caught)


# --------------------------------------------------------------------------
# Exemples annotés et régression logistique
# --------------------------------------------------------------------------


class TestData:
    def test_formats(self):
        expected = [Example("a", "x"), Example("b", None)]
        assert as_examples([("a", "x"), ("b", "none")]) == expected
        assert as_examples([{"text": "a", "label": "x"}, {"text": "b", "label": ""}]) == expected
        assert as_examples({"x": ["a"], None: ["b"]}) == expected

    def test_invalid(self):
        with pytest.raises(ValueError):
            as_examples([])
        with pytest.raises(ValueError, match="texte"):
            as_examples([("", "x")])
        with pytest.raises(ValueError, match="liste de textes"):
            as_examples({"x": "abc"})

    @pytest.mark.parametrize("suffix", [".csv", ".tsv", ".jsonl"])
    def test_load_files(self, tmp_path, suffix):
        path = tmp_path / f"annotes{suffix}"
        if suffix == ".jsonl":
            path.write_text('{"text": "Colis perdu, virgule", "label": "shipping"}\n\n'
                            '{"text": "Recette", "label": "none"}\n', encoding="utf-8")
        else:
            sep = "\t" if suffix == ".tsv" else ","
            path.write_text(f'label{sep}text\nshipping{sep}"Colis perdu, virgule"\nnone{sep}Recette\n',
                            encoding="utf-8")
        assert load_examples(path) == [Example("Colis perdu, virgule", "shipping"), Example("Recette", None)]

    def test_load_missing_column(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("message,categorie\na,b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="colonnes"):
            load_examples(path)
        assert load_examples(path, text_column="message", label_column="categorie") == [Example("a", "b")]

    def test_load_real_repo_file(self):
        from pathlib import Path

        examples = load_examples(Path(__file__).parents[1] / "examples" / "data" / "annotes.csv")
        assert len(examples) == 290
        assert sum(ex.label is None for ex in examples) == 50

    def test_split_is_stratified_and_reproducible(self):
        ex = [(f"a{i}", "a") for i in range(10)] + [(f"b{i}", "b") for i in range(20)] + [("seul", "c")]
        train, test = split_examples(ex, 0.3, seed=3)
        assert len(test) == 3 + 6 and len(train) == 22
        assert sum(e.label == "a" for e in test) == 3
        assert any(e.text == "seul" for e in train)
        assert split_examples(ex, 0.3, seed=3) == (train, test)
        with pytest.raises(ValueError):
            split_examples(ex, 1.5)


# --------------------------------------------------------------------------
# Embedders et dépendances optionnelles
# --------------------------------------------------------------------------


class TestEmbedders:
    @pytest.mark.parametrize(("name", "prefix"), [
        ("intfloat/multilingual-e5-base", "query: "), ("modeles/e5-small", "query: "),
        ("/tmp/55103e54/tiny", ""), ("BAAI/bge-m3", ""),
    ])
    def test_default_prefix(self, name, prefix):
        from krito.classifier import _default_prefix

        assert _default_prefix(name) == prefix

    def test_onnx_embedder_mean_pooling(self, tmp_path):
        """Mini-modèle ONNX : chaque token a un vecteur fixe ; le padding doit être ignoré."""
        pytest.importorskip("onnxruntime")
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper
        from tokenizers import Tokenizer, models, pre_tokenizers

        from krito import OnnxEmbedder

        vocab = {"[PAD]": 0, "[UNK]": 1, "colis": 2, "facture": 3}
        tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.save(str(tmp_path / "tokenizer.json"))
        table = np.array([[0, 0, 5], [0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        graph = helper.make_graph(
            [helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"])], "tiny",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["b", "s"]),
             helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["b", "s"])],
            [helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "s", 3])],
            [helper.make_tensor("table", TensorProto.FLOAT, table.shape, table.flatten())],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        onnx.save(model, tmp_path / "model.onnx")

        emb = OnnxEmbedder(tmp_path, batch_size=2).encode(["colis", "colis facture", "facture"])
        expected = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(emb, expected, atol=1e-6)

    def test_missing_scikit_learn_is_explicit(self, monkeypatch):
        import importlib.util

        find_spec = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "sklearn" else find_spec(name, *a))
        engine = KritoEngine(backend=TableBackend(support_table()))
        with pytest.raises(ImportError, match=r"krito\[learn\]"):
            engine.fit_judge(support_examples(), OPTIONS)
        with pytest.raises(ImportError, match=r"krito\[learn\]"):
            KritoClassifier(FakeEmbedder()).fit(supervised_examples())
        # Le zero-shot et la calibration n'en ont pas besoin
        assert engine.calibrate(support_examples(), OPTIONS).reached
