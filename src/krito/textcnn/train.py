"""Entraînement de zéro et export ONNX du modèle TextCNN (extra ``krito[torch]``).

Chaque texte annoté produit des paires (texte, hypothèse) :

- hypothèse de sa catégorie -> *entailment* ;
- hypothèse d'une autre catégorie du même domaine -> *contradiction* ;
- hypothèse d'une catégorie d'un **autre** domaine -> *neutral* (sujet sans rapport) ;
- texte hors-sujet (``none``) + hypothèses de son domaine -> *contradiction*.

Deux objectifs sont optimisés ensemble : la classification NLI du cross-encodeur et un
objectif contrastif du bi-encodeur (le texte doit être plus proche de sa catégorie que des
autres catégories de son domaine), qui sert au routage quand les options sont nombreuses.

    krito-textcnn --domain messages.csv categories.json --out modeles/krito-textcnn-maboite \\
        --eval messages_test.csv categories.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .backend import CROSS_FILES, ENCODER_FILES, AdaptiveRouter, TextCNNBackend
from .model import AdaptiveSystem1Classifier, TextCNNConfig

NONE = "none"
PAD, UNK = "[PAD]", "[UNK]"
TEMPLATES = [
    "Ce texte concerne {}.",
    "Ce message concerne {}.",
    "Le sujet de ce texte est {}.",
    "Il s'agit de {}.",
    "Ce texte porte sur {}.",
]
ENTAILMENT, CONTRADICTION, NEUTRAL = 0, 1, 2


@dataclass
class Domain:
    """Une taxonomie (clé -> description) et ses exemples annotés (label, texte)."""

    name: str
    classes: dict[str, str]
    examples: list[tuple[str, str]]

    @classmethod
    def from_files(cls, csv_path: str | Path, taxonomy_path: str | Path) -> Domain:
        classes = json.loads(Path(taxonomy_path).read_text(encoding="utf-8"))
        with open(csv_path, encoding="utf-8") as f:
            rows = [(r["label"].strip(), r["text"]) for r in csv.DictReader(f)]
        unknown = {label for label, _ in rows} - set(classes) - {NONE}
        if unknown:
            raise ValueError(f"{csv_path} : labels absents de la taxonomie : {sorted(unknown)}")
        return cls(Path(csv_path).stem, classes, rows)


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------


def train_tokenizer(texts: list[str], vocab_size: int = 8000):
    """WordPiece entraîné sur le corpus : minuscules, sans accents (robuste aux fautes de frappe)."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tok = Tokenizer(models.WordPiece(unk_token=UNK))
    tok.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.StripAccents(), normalizers.Lowercase(), normalizers.Strip()]
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[PAD, UNK])
    tok.train_from_iterator(texts, trainer)
    assert tok.token_to_id(PAD) == 0 and tok.token_to_id(UNK) == 1
    return tok


# --------------------------------------------------------------------------
# Backend PyTorch (entraînement, évaluation avant export)
# --------------------------------------------------------------------------


class TorchTextCNNBackend(AdaptiveRouter):
    """Même interface et même routage que ``TextCNNBackend``, avec le modèle PyTorch."""

    def __init__(self, model: AdaptiveSystem1Classifier, tokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        cfg = model.config
        self.config = cfg
        self.threshold, self.top_k = cfg.threshold, cfg.top_k
        self.max_context_length, self.max_option_length = cfg.max_context_length, cfg.max_option_length
        self.hypothesis_template = cfg.hypothesis_template
        self._init_tokenizers(tokenizer.to_str())

    def tensors(self, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        ids, mask, n_trunc = self._tokenize(texts, max_length)
        return torch.from_numpy(ids).to(self.device), torch.from_numpy(mask).to(self.device), n_trunc

    @torch.no_grad()
    def _encode(self, texts, max_length):
        self.model.eval()
        ids, mask, n_trunc = self.tensors(texts, max_length)
        tokens, emb = self.model.encode(ids, mask)
        return tokens.cpu().numpy(), mask.cpu().numpy(), emb.cpu().numpy(), n_trunc

    @torch.no_grad()
    def _cross(self, p_tok, p_mask, q_tok, q_mask):
        self.model.eval()
        t = [torch.from_numpy(np.asarray(x)).to(self.device) for x in (p_tok, p_mask, q_tok, q_mask)]
        return self.model.cross_encoder(*t).cpu().numpy()

    def save(self, out_dir: str | Path) -> None:
        """Poids PyTorch + tokenizer + configuration (pour réentraîner ou réexporter)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model.pt")
        self.tokenizer.save(str(out / "tokenizer.json"))
        self.config.save(out / "config.json")

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu") -> TorchTextCNNBackend:
        from tokenizers import Tokenizer

        d = Path(model_dir)
        model = AdaptiveSystem1Classifier(TextCNNConfig.load(d / "config.json"))
        model.load_state_dict(torch.load(d / "model.pt", map_location=device, weights_only=True))
        return cls(model.eval(), Tokenizer.from_file(str(d / "tokenizer.json")), device)


# --------------------------------------------------------------------------
# Entraînement
# --------------------------------------------------------------------------


def _batch_pairs(batch, domains, rng, templates, n_neg, n_cross):
    """Hypothèses de la batch et paires (indice texte, indice hypothèse, cible NLI)."""
    hyps, hyp_index = [], {}
    for d in sorted({d for d, _, _ in batch}):
        for k, desc in domains[d].classes.items():
            hyp_index[(d, k)] = len(hyps)
            hyps.append(rng.choice(templates).format(desc))
    pairs, positives = [], []
    for i, (d, label, _) in enumerate(batch):
        keys = list(domains[d].classes)
        if label != NONE:
            pairs.append((i, hyp_index[(d, label)], ENTAILMENT))
            positives.append((i, hyp_index[(d, label)], d))
        negs = [k for k in keys if k != label]
        for k in rng.sample(negs, min(n_neg, len(negs))):
            pairs.append((i, hyp_index[(d, k)], CONTRADICTION))
        others = [h for (od, _), h in hyp_index.items() if od != d]
        for h in rng.sample(others, min(n_cross, len(others))):
            pairs.append((i, h, NEUTRAL))
    return hyps, hyp_index, pairs, positives


def train(
    domains: list[Domain],
    *,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-3,
    weight_decay: float = 0.01,
    n_neg: int = 2,
    n_cross: int = 1,
    word_dropout: float = 0.1,
    bi_weight: float = 1.0,
    bi_temperature: float = 0.05,
    vocab_size: int = 8000,
    templates: list[str] = TEMPLATES,
    seed: int = 42,
    device: str | None = None,
    log=print,
    **config_kwargs,
) -> TorchTextCNNBackend:
    """Entraîne tokenizer et modèle à partir de zéro ; renvoie un backend prêt à l'emploi."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(seed)
    torch.manual_seed(seed)

    corpus = [t for d in domains for _, t in d.examples]
    corpus += [tpl.format(desc) for d in domains for desc in d.classes.values() for tpl in templates]
    tokenizer = train_tokenizer(corpus, vocab_size)
    config = TextCNNConfig(vocab_size=tokenizer.get_vocab_size(), hypothesis_template=templates[0], **config_kwargs)
    model = AdaptiveSystem1Classifier(config)
    backend = TorchTextCNNBackend(model, tokenizer, device)
    log(f"  {len(corpus)} textes, vocabulaire {config.vocab_size}, {model.num_parameters() / 1e6:.2f} M paramètres")

    items = [(i, label, text) for i, d in enumerate(domains) for label, text in d.examples]
    steps = epochs * math.ceil(len(items) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )

    for epoch in range(epochs):
        model.train()
        rng.shuffle(items)
        t0, total, n = time.perf_counter(), 0.0, 0
        for b in range(0, len(items), batch_size):
            batch = items[b : b + batch_size]
            hyps, hyp_index, pairs, positives = _batch_pairs(batch, domains, rng, templates, n_neg, n_cross)
            c_ids, c_mask, _ = backend.tensors([t for _, _, t in batch], config.max_context_length)
            if word_dropout:
                drop = (torch.rand(c_ids.shape, device=device) < word_dropout) & (c_mask == 1)
                c_ids = c_ids.masked_fill(drop, 1)  # [UNK]
            h_ids, h_mask, _ = backend.tensors(hyps, config.max_option_length)
            c_tok, c_emb = model.encode(c_ids, c_mask)
            h_tok, h_emb = model.encode(h_ids, h_mask)

            ti, hi, target = (torch.tensor(x, device=device) for x in zip(*pairs))
            logits = model.cross_encoder(c_tok[ti], c_mask[ti], h_tok[hi], h_mask[hi])
            loss = F.cross_entropy(logits, target)

            if positives and bi_weight:
                pi, ph, pd = zip(*positives)
                hyp_domain = torch.tensor([d for d, _ in hyp_index], device=device)
                sims = (c_emb[list(pi)] @ h_emb.T) / bi_temperature
                sims = sims.masked_fill(hyp_domain.unsqueeze(0) != torch.tensor(pd, device=device).unsqueeze(1), -1e4)
                loss = loss + bi_weight * F.cross_entropy(sims, torch.tensor(ph, device=device))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
            n += len(batch)
        log(f"  époque {epoch + 1}/{epochs} : perte {total / n:.3f} ({time.perf_counter() - t0:.1f} s)")
    model.eval()
    return backend


# --------------------------------------------------------------------------
# Évaluation et export
# --------------------------------------------------------------------------


def evaluate(backend: AdaptiveRouter, domain: Domain, template: str | None = None) -> dict:
    """Précision (textes du domaine) et logits bruts [n_textes, n_classes, 3], scoring Krito (E − C)."""
    template = template or backend.hypothesis_template
    keys = list(domain.classes)
    hyps = [template.format(domain.classes[k]) for k in keys]
    texts = [t for _, t in domain.examples]
    logits = backend.logits([(t, h) for t in texts for h in hyps]).reshape(len(texts), len(keys), 3)
    pred = (logits[..., 0] - logits[..., 1]).argmax(1)
    y = np.array([keys.index(label) if label in keys else -1 for label, _ in domain.examples])
    ind = y >= 0
    return {"accuracy": float((pred[ind] == y[ind]).mean()) if ind.any() else float("nan"), "logits": logits, "y": y}


class _EncoderExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model.encode(input_ids, attention_mask)


def export_onnx(backend: TorchTextCNNBackend, out_dir: str | Path, *, quantize: bool = False) -> Path:
    """Écrit ``encoder.onnx``, ``cross.onnx``, ``tokenizer.json`` et ``config.json``, puis
    vérifie que le backend ONNX reproduit les logits PyTorch."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = backend.model.to("cpu").eval()
    backend.device = "cpu"
    batch, seq, seq_q = torch.export.Dim("batch"), torch.export.Dim("seq"), torch.export.Dim("seq_q")
    ids = torch.randint(2, model.config.vocab_size, (2, 9))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        tok, _ = model.encode(ids, mask)
    for stale in (*ENCODER_FILES, *CROSS_FILES):  # une variante int8 d'un export précédent serait prioritaire
        (out / stale).unlink(missing_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*axis name.*will not be used")  # noms d'axes, sans effet
        torch.onnx.export(
            _EncoderExport(model).eval(), (ids, mask), out / "encoder.onnx", dynamo=True, external_data=False,
            verbose=False, input_names=["input_ids", "attention_mask"],
            output_names=["token_features", "sentence_embedding"],
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
        )
        q_tok, q_mask = tok[:, :5].contiguous(), mask[:, :5].contiguous()
        torch.onnx.export(
            model.cross_encoder, (tok, mask, q_tok, q_mask), out / "cross.onnx", dynamo=True, external_data=False,
            verbose=False, input_names=["p_tokens", "p_mask", "q_tokens", "q_mask"], output_names=["nli_logits"],
            dynamic_shapes={"p": {0: batch, 1: seq}, "p_mask": {0: batch, 1: seq},
                            "q": {0: batch, 1: seq_q}, "q_mask": {0: batch, 1: seq_q}},
        )
    backend.tokenizer.save(str(out / "tokenizer.json"))
    model.config.save(out / "config.json")
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        for name in ("encoder", "cross"):
            quantize_dynamic(out / f"{name}.onnx", out / f"{name}.int8.onnx", weight_type=QuantType.QInt8)
            (out / f"{name}.onnx").unlink()

    sample = [("Mon colis n'est jamais arrivé.", "Ce texte concerne la livraison."),
              ("Mon colis n'est jamais arrivé.", "Ce texte concerne la facturation."),
              ("Bonjour, je souhaite résilier mon abonnement dès que possible, merci.", "Il s'agit d'une résiliation.")]
    expected, got = backend.logits(sample), TextCNNBackend(out).logits(sample)
    if not quantize and not np.allclose(expected, got, atol=1e-3):
        raise RuntimeError(f"L'export ONNX diverge du modèle PyTorch :\n{expected}\n{got}")
    return out


# --------------------------------------------------------------------------
# Ligne de commande
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="krito-textcnn",
        description="Entraîne de zéro le modèle TextCNN de Krito sur vos données, puis l'exporte en ONNX.",
    )
    p.add_argument("--domain", nargs=2, action="append", required=True, metavar=("CSV", "TAXONOMIE"),
                   help="CSV annoté (colonnes text, label ; « none » = hors-sujet) et sa taxonomie JSON. Répétable.")
    p.add_argument("--eval", nargs=2, action="append", default=[], metavar=("CSV", "TAXONOMIE"),
                   help="Jeu d'évaluation, distinct des données d'entraînement. Répétable.")
    p.add_argument("--out", required=True, help="Dossier de sortie (utilisable par KritoEngine.from_textcnn).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--threshold", type=int, default=10, help="Au-delà, le bi-encodeur présélectionne les options.")
    p.add_argument("--top-k", type=int, default=5, help="Options transmises au cross-encodeur après présélection.")
    p.add_argument("--quantize", action="store_true", help="Quantification int8 dynamique (modèle ~4x plus petit).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    domains = [Domain.from_files(c, t) for c, t in args.domain]
    evals = [Domain.from_files(c, t) for c, t in args.eval]
    print(f"Entraînement sur {sum(len(d.examples) for d in domains)} exemples ({len(domains)} domaine(s))")
    backend = train(domains, epochs=args.epochs, vocab_size=args.vocab_size, seed=args.seed,
                    threshold=args.threshold, top_k=args.top_k)
    for d in evals:
        print(f"Précision sur {d.name} : {evaluate(backend, d)['accuracy']:.1%}")
    backend.save(Path(args.out) / "torch")
    out = export_onnx(backend, args.out, quantize=args.quantize)
    print(f"Modèle exporté dans {out}\n  KritoEngine.from_textcnn({str(out)!r})")


if __name__ == "__main__":
    sys.exit(main())
