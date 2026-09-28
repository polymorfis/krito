"""Tests d'intégration avec de vrais modèles.

Lancer uniquement ceux-ci : ``uv run pytest -m integration``
Les ignorer :              ``uv run pytest -m "not integration"``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from krito import KritoEngine

pytestmark = pytest.mark.integration

MODELS = Path(__file__).parents[1] / "experiments" / "models"
# Modèle multi-domaines s'il existe, sinon celui de la première étude (support seul)
ONNX_DIR = next((MODELS / n for n in ("krito-nli-fr-multi", "krito-nli-fr-mdeberta") if (MODELS / n).exists()),
                MODELS / "krito-nli-fr-multi")
TEMPLATE = "Ce message concerne {}."
SUPPORT = {
    "billing": "un problème de facturation, de paiement ou de remboursement",
    "shipping": "la livraison, le suivi ou le retour d'un colis",
    "account": "la connexion, le mot de passe ou l'accès au compte",
    "bug": "un bug technique ou une panne du site ou de l'application",
}


@pytest.fixture(scope="module")
def engine():
    try:
        return KritoEngine(hypothesis_template=TEMPLATE)
    except Exception as exc:  # modèle absent du cache et pas de réseau
        pytest.skip(f"Modèle indisponible : {exc}")


@pytest.fixture(scope="module")
def onnx_engine():
    pytest.importorskip("onnxruntime")
    if not ONNX_DIR.exists():
        pytest.skip("Modèle ONNX fine-tuné absent (voir experiments/export_onnx.py)")
    return KritoEngine.from_onnx(ONNX_DIR, hypothesis_template=TEMPLATE)


def test_result_structure(engine):
    res = engine.classify("Mon colis n'est jamais arrivé.", SUPPORT)
    assert res.selected_key in SUPPORT
    assert set(res.scores) == set(SUPPORT)
    assert sum(s.probability for s in res.scores.values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= res.margin <= res.confidence <= 1.0
    assert all(0.0 <= s.entailment_prob <= 1.0 for s in res.scores.values())


def test_is_deterministic(engine):
    first = engine.classify("J'ai été débité deux fois.", SUPPORT)
    second = engine.classify("J'ai été débité deux fois.", SUPPORT)
    assert first == second


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("Le colis indiqué comme livré n'est jamais arrivé dans ma boîte aux lettres.", "shipping"),
        ("J'ai été débité deux fois sur ma carte pour la même commande.", "billing"),
        ("J'ai oublié mon mot de passe et je ne peux plus me connecter.", "account"),
    ],
)
def test_obvious_french_cases(engine, context, expected):
    assert engine.classify(context, SUPPORT).selected_key == expected


def test_long_context_warns(engine):
    with pytest.warns(UserWarning, match="tronquées"):
        engine.classify("Bonjour. " * 600 + "Mon colis n'est jamais arrivé.", SUPPORT)


# ---------------------------------------------------------------- ONNX


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        # Phrases absentes des données d'entraînement du modèle fine-tuné
        ("Le tableau de bord affiche une erreur 502 depuis ce matin.", "bug"),
        ("Je n'arrive pas à télécharger ma facture du mois de mars.", "billing"),
        ("Mon paquet est bloqué à la douane depuis lundi.", "shipping"),
    ],
)
def test_onnx_finetuned_cases(onnx_engine, context, expected):
    assert onnx_engine.classify(context, SUPPORT).selected_key == expected


def test_onnx_rejects_off_topic(onnx_engine):
    res = onnx_engine.classify("Pouvez-vous me donner une recette de crêpes ?", SUPPORT, min_entailment=0.5)
    assert res.accepted is False


@pytest.mark.parametrize(
    ("context", "options", "expected"),
    [
        # Domaines et formulations absents de tout entraînement
        ("Mon chien boite depuis hier et refuse de manger, je peux passer à la clinique ?",
         {"urgence": "une urgence vétérinaire", "vaccin": "une vaccination", "toilettage": "un toilettage"}, "urgence"),
        ("Le vol AF1234 a été annulé, comment obtenir l'indemnisation prévue par le règlement européen ?",
         {"refund": "un remboursement ou une indemnisation", "booking": "une nouvelle réservation",
          "baggage": "un bagage perdu"}, "refund"),
    ],
)
def test_onnx_unseen_domain(onnx_engine, context, options, expected):
    assert onnx_engine.classify(context, options).selected_key == expected


def test_onnx_entailment_ranks_on_topic_above_off_topic(onnx_engine):
    # L'échelle absolue de P(entailment) dépend du modèle et du gabarit : le seuil
    # min_entailment se calibre sur des données de validation. Ce qui est garanti,
    # c'est l'ordre : un message du domaine obtient une implication plus forte.
    on = onnx_engine.classify("Le site affiche une page blanche depuis la mise à jour.", SUPPORT)
    off = onnx_engine.classify("Pouvez-vous me donner une recette de crêpes ?", SUPPORT)
    assert on.selected_key == "bug"
    assert on.scores["bug"].entailment_prob > off.scores[off.selected_key].entailment_prob


def test_onnx_long_context_warns(onnx_engine):
    with pytest.warns(UserWarning, match="tronquées"):
        onnx_engine.classify("Bonjour. " * 600 + "Mon colis n'est jamais arrivé.", SUPPORT)


def test_onnx_is_thread_safe(onnx_engine):
    # Un serveur web appelle le même moteur depuis plusieurs threads.
    from concurrent.futures import ThreadPoolExecutor

    texts = ["Mon colis est perdu.", "J'ai été débité deux fois.", "Mot de passe refusé.",
             "Le site est en panne.", "Bonjour. " * 400 + "Facture en double."] * 4
    expected = [onnx_engine.classify(t, SUPPORT) for t in texts]
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda t: onnx_engine.classify(t, SUPPORT), texts))
    assert [r.selected_key for r in got] == [r.selected_key for r in expected]
    assert [round(r.confidence, 6) for r in got] == [round(r.confidence, 6) for r in expected]
