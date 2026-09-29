"""Tests du modèle léger TextCNN (backend ONNX livré avec Krito, et modèle PyTorch).

Le modèle livré pèse quelques Mo : ces tests sont rapides et tournent sans PyTorch.
Les tests d'entraînement et d'export ne tournent que si PyTorch est installé.
"""

from __future__ import annotations

import numpy as np
import pytest

from krito import KritoEngine

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from krito.textcnn import DEFAULT_TEXTCNN_MODEL, TextCNNBackend  # noqa: E402
from krito.textcnn.backend import PRUNED_LOGITS  # noqa: E402

SUPPORT = {
    "billing": "un problème de facturation, de paiement ou de remboursement",
    "shipping": "la livraison, le suivi ou le retour d'un colis",
    "account": "la connexion, le mot de passe ou l'accès au compte",
    "bug": "un bug technique ou une panne du site ou de l'application",
}
MANY = SUPPORT | {f"autre_{i}": f"un sujet administratif numéro {i}" for i in range(11)}


@pytest.fixture(scope="module")
def backend():
    return TextCNNBackend(threads=1)


@pytest.fixture(scope="module")
def engine(backend):
    return KritoEngine(backend=backend, hypothesis_template=backend.hypothesis_template)


def test_default_model_is_packaged():
    assert (DEFAULT_TEXTCNN_MODEL / "config.json").exists()
    assert (DEFAULT_TEXTCNN_MODEL / "tokenizer.json").exists()


def test_from_textcnn_uses_training_template():
    engine = KritoEngine.from_textcnn(threads=1)
    assert engine.hypothesis_template == engine.backend.hypothesis_template
    assert KritoEngine.from_textcnn(hypothesis_template="Il s'agit de {}.").hypothesis_template == "Il s'agit de {}."


def test_result_structure(engine):
    res = engine.classify("Mon colis n'est jamais arrivé.", SUPPORT)
    assert res.selected_key in SUPPORT
    assert sum(s.probability for s in res.scores.values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= s.entailment_prob <= 1.0 for s in res.scores.values())


def test_is_deterministic(engine):
    assert engine.classify("J'ai été débité deux fois.", SUPPORT) == engine.classify("J'ai été débité deux fois.", SUPPORT)


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("Le colis indiqué comme livré n'est jamais arrivé dans ma boîte aux lettres.", "shipping"),
        ("J'ai oublié mon mot de passe et je ne peux plus me connecter.", "account"),
    ],
)
def test_obvious_french_cases(engine, context, expected):
    assert engine.classify(context, SUPPORT).selected_key == expected


def test_logits_shape_and_empty(backend):
    assert backend.logits([]).shape == (0, 3)
    out = backend.logits([("a", "b"), ("c", "d"), ("a", "d")])
    assert out.shape == (3, 3) and out.dtype == np.float32


def test_batched_contexts_match_single_calls(backend):
    """Paires de plusieurs contextes dans un même appel : même résultat que séparément (padding neutre)."""
    hyps = [backend.hypothesis_template.format(d) for d in SUPPORT.values()]
    short, long = "Colis perdu.", "Bonjour, j'ai été prélevé deux fois ce mois-ci pour la même commande, merci de corriger."
    mixed = backend.logits([(c, h) for c in (short, long) for h in hyps])
    alone = np.vstack([backend.logits([(c, h) for h in hyps]) for c in (short, long)])
    np.testing.assert_allclose(mixed, alone, atol=1e-4)


def test_few_options_all_go_through_cross_encoder(backend):
    pairs = [("Mon colis n'est jamais arrivé.", backend.hypothesis_template.format(d)) for d in SUPPORT.values()]
    assert not (backend.logits(pairs) == np.float32(PRUNED_LOGITS)).all(axis=1).any()


def test_many_options_are_pruned_to_top_k(backend):
    pairs = [("Mon colis n'est jamais arrivé.", backend.hypothesis_template.format(d)) for d in MANY.values()]
    assert len(pairs) > backend.threshold
    pruned = (backend.logits(pairs) == np.float32(PRUNED_LOGITS)).all(axis=1)
    assert (~pruned).sum() == backend.top_k


def test_routing_overrides():
    b = TextCNNBackend(threshold=100, top_k=2)
    pairs = [("Mon colis n'est jamais arrivé.", d) for d in MANY.values()]
    assert not (b.logits(pairs) == np.float32(PRUNED_LOGITS)).all(axis=1).any()
    b = TextCNNBackend(threshold=1, top_k=2)
    assert (~(b.logits(pairs) == np.float32(PRUNED_LOGITS)).all(axis=1)).sum() == 2
    with pytest.raises(ValueError):
        TextCNNBackend(top_k=0)


def test_pruned_options_get_near_zero_probability(engine):
    res = engine.classify("Mon colis n'est jamais arrivé.", MANY)
    low = [s for s in res.scores.values() if s.decision_score == PRUNED_LOGITS[0] - PRUNED_LOGITS[1]]
    assert len(low) == len(MANY) - engine.backend.top_k
    assert all(s.probability < 1e-6 and s.entailment_prob < 1e-6 for s in low)


def test_embeddings_are_normalized(backend):
    emb = backend.embed(["Mon colis n'est jamais arrivé.", "Facture incorrecte"])
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


def test_long_context_warns(engine):
    with pytest.warns(UserWarning, match="tronquées"):
        engine.classify("Bonjour. " * 600 + "Mon colis n'est jamais arrivé.", SUPPORT)


def test_missing_model_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        (tmp_path / "config.json").write_text(
            '{"threshold": 10, "top_k": 5, "max_context_length": 8, "max_option_length": 8, "hypothesis_template": "{}"}'
        )
        TextCNNBackend(tmp_path)


# --------------------------------------------------------------------------
# PyTorch : modèle, entraînement et export
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _tiny_config(**kw):
    from krito.textcnn.model import TextCNNConfig

    return TextCNNConfig(vocab_size=50, embed_dim=16, num_filters=8, embedding_dim=8, **kw)


def test_backbone_accepts_inputs_shorter_than_kernels(torch):
    from krito.textcnn.model import AdaptiveSystem1Classifier

    model = AdaptiveSystem1Classifier(_tiny_config()).eval()
    ids = torch.tensor([[5, 6]])
    tokens, emb = model.encode(ids, torch.ones_like(ids))
    assert tokens.shape == (1, 2, model.backbone.out_dim) and emb.shape == (1, 8)


def test_padding_does_not_change_outputs(torch):
    from krito.textcnn.model import AdaptiveSystem1Classifier

    model = AdaptiveSystem1Classifier(_tiny_config()).eval()
    ids = torch.tensor([[5, 6, 7, 8]])
    padded = torch.tensor([[5, 6, 7, 8, 0, 0, 0]])
    q = torch.tensor([[9, 10, 11]])
    with torch.no_grad():
        a = model(ids, torch.ones_like(ids), q, torch.ones_like(q))
        b = model(padded, (padded != 0).long(), q, torch.ones_like(q))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


def test_predict_routes_adaptively(torch):
    from krito.textcnn.model import PRUNED_LOGITS as TORCH_PRUNED
    from krito.textcnn.model import AdaptiveSystem1Classifier

    assert tuple(TORCH_PRUNED) == tuple(PRUNED_LOGITS)
    model = AdaptiveSystem1Classifier(_tiny_config(threshold=3, top_k=2)).eval()
    ctx = torch.randint(2, 50, (1, 12))
    for n, expected in ((3, 3), (7, 2)):
        opts = torch.randint(2, 50, (n, 5))
        logits, selected = model.predict(ctx, torch.ones_like(ctx), opts, torch.ones_like(opts))
        assert logits.shape == (n, 3) and selected.numel() == expected


def test_train_export_roundtrip(torch, tmp_path):
    pytest.importorskip("onnxscript")
    from krito.textcnn.train import Domain, TorchTextCNNBackend, evaluate, export_onnx, train

    classes = {"a": "la livraison d'un colis", "b": "la facturation"}
    examples = [("a", "Mon colis est perdu."), ("b", "Ma facture est fausse."), ("none", "Il fait beau.")] * 8
    backend = train([Domain("d", classes, examples)], epochs=1, vocab_size=200, embed_dim=16, num_filters=8,
                    embedding_dim=8, log=lambda *_: None, device="cpu")
    assert 0.0 <= evaluate(backend, Domain("d", classes, examples))["accuracy"] <= 1.0
    backend.save(tmp_path / "torch")
    reloaded = TorchTextCNNBackend.load(tmp_path / "torch")
    export_onnx(backend, tmp_path / "onnx")  # vérifie lui-même la parité PyTorch / ONNX
    pairs = [("Mon colis est perdu.", "Ce texte concerne la facturation.")]
    np.testing.assert_allclose(reloaded.logits(pairs), TextCNNBackend(tmp_path / "onnx").logits(pairs), atol=1e-4)
    engine = KritoEngine.from_textcnn(tmp_path / "onnx")
    assert engine.classify("Mon colis est perdu.", classes).selected_key in classes


def test_batch_api_keeps_routing_per_context(engine):
    """classify_batch avec un petit batch_size : mêmes décisions que classify, routage intact."""
    texts = ["Mon colis n'est jamais arrivé.", "J'ai oublié mon mot de passe.", "Facture en double."]
    batch = engine.classify_batch(texts, MANY, batch_size=4)
    assert batch == [engine.classify(t, MANY) for t in texts]
    for res in batch:
        kept = [s for s in res.scores.values() if s.decision_score != PRUNED_LOGITS[0] - PRUNED_LOGITS[1]]
        assert len(kept) == engine.backend.top_k
