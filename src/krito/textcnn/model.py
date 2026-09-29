"""Modèle Krito léger, entraîné de zéro : TextCNN + bi-encodeur + cross-encodeur.

Architecture (aucun poids pré-entraîné, environ 1,6 M de paramètres) :

- ``TextCNNBackbone`` : embeddings de sous-mots, convolutions 1D multi-noyaux puis une
  convolution dilatée résiduelle. Produit des caractéristiques **par token** (pour le
  cross-encodeur) et un vecteur **par texte** (pooling max + moyenne masqué).
- ``BiEncoder`` : projette le vecteur d'un texte dans un espace normalisé. Contexte et
  options sont encodés **indépendamment** : le score est un simple produit scalaire,
  ce qui permet de présélectionner les options quand elles sont nombreuses.
- ``CrossEncoder`` : fait interagir le contexte et l'option token à token (alignement
  par attention, à la manière d'ESIM), puis produit 3 logits NLI
  (entailment, contradiction, neutral), comme les cross-encoders NLI utilisés par Krito.
- ``AdaptiveSystem1Classifier`` : orchestre les deux. Jusqu'à ``threshold`` options,
  toutes passent par le cross-encodeur ; au-delà, le bi-encodeur retient les ``top_k``
  meilleures et seules celles-ci passent par le cross-encodeur.

Le backbone est partagé : chaque texte (contexte ou option) n'est encodé qu'une fois,
le cross-encodeur ne travaille que sur les caractéristiques déjà calculées.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Logits (E, C, N) attribués aux options écartées par le bi-encodeur : probabilité ≈ 0.
PRUNED_LOGITS = (-10.0, 10.0, 0.0)
_MASK_VALUE = -1e4  # plutôt que -inf : pas de NaN sur une ligne entièrement masquée


@dataclass
class TextCNNConfig:
    vocab_size: int
    embed_dim: int = 128
    num_filters: int = 64
    kernel_sizes: tuple[int, ...] = (1, 3, 5)
    embedding_dim: int = 128  # dimension de l'espace du bi-encodeur
    dropout: float = 0.2
    max_context_length: int = 256
    max_option_length: int = 64
    threshold: int = 10
    top_k: int = 5
    hypothesis_template: str = "Ce texte concerne {}."
    labels: dict[str, int] = field(default_factory=lambda: {"entailment": 0, "contradiction": 1, "neutral": 2})

    def __post_init__(self):
        self.kernel_sizes = tuple(self.kernel_sizes)
        if any(k % 2 == 0 for k in self.kernel_sizes):
            raise ValueError("Les tailles de noyau doivent être impaires (la longueur de séquence est conservée).")
        if self.top_k < 1 or self.threshold < 1:
            raise ValueError("threshold et top_k doivent être >= 1.")

    @property
    def hidden_dim(self) -> int:
        return self.num_filters * len(self.kernel_sizes)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> TextCNNConfig:
        return cls(**json.loads(Path(path).read_text()))


def masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatène max et moyenne sur les seuls tokens réels. x : [B, L, H], mask : [B, L] -> [B, 2H]."""
    m = mask.unsqueeze(-1)
    x_max = x.masked_fill(m == 0, _MASK_VALUE).max(dim=1).values
    x_mean = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return torch.cat([x_max, x_mean], dim=-1)


class TextCNNBackbone(nn.Module):
    """Extraction de caractéristiques par CNN 1D, token par token.

    Les convolutions gardent la longueur de la séquence (padding « same »), ce qui accepte
    des textes plus courts que le plus grand noyau et fournit une représentation par token.
    Les positions de padding sont remises à zéro après chaque couche : elles n'influencent
    ni les voisins, ni le pooling.
    """

    def __init__(self, config: TextCNNConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(config.embed_dim, config.num_filters, kernel_size=k, padding=k // 2)
            for k in config.kernel_sizes
        )
        h = config.hidden_dim
        # Convolution dilatée résiduelle : élargit le contexte vu par chaque token (~9 tokens).
        self.context_conv = nn.Conv1d(h, h, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.LayerNorm(h)
        self.dropout = nn.Dropout(config.dropout)
        self.out_dim = h

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids, attention_mask : [B, L] -> caractéristiques par token [B, L, H]."""
        m = attention_mask.unsqueeze(1).to(torch.float32)  # [B, 1, L]
        x = self.dropout(self.embedding(input_ids)).transpose(1, 2)  # [B, E, L]
        x = torch.cat([F.gelu(conv(x)) for conv in self.convs], dim=1) * m
        x = x + F.gelu(self.context_conv(self.dropout(x))) * m
        return self.norm(x.transpose(1, 2)) * m.transpose(1, 2)


class BiEncoder(nn.Module):
    """Vecteur normalisé d'un texte, comparable par produit scalaire."""

    def __init__(self, in_dim: int, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Linear(2 * in_dim, output_dim)

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = masked_pool(tokens, attention_mask.to(tokens.dtype))
        return F.normalize(self.projection(pooled), p=2, dim=-1)


class CrossEncoder(nn.Module):
    """Interaction token à token entre contexte (P) et option (Q), 3 logits NLI en sortie.

    Chaque token de P est aligné par attention sur les tokens de Q (et inversement), puis
    comparé à son alignement : [x, x̃, x − x̃, x ⊙ x̃]. Une simple concaténation
    contexte + option passée au même CNN mélange les deux textes sans les comparer.
    """

    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.scale = 1.0 / math.sqrt(in_dim)
        self.compare = nn.Sequential(nn.Linear(4 * in_dim, in_dim), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 3),
        )

    def forward(self, p: torch.Tensor, p_mask: torch.Tensor, q: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        """p : [B, Lp, H], q : [B, Lq, H], masques [B, L] -> logits [B, 3] (E, C, N)."""
        p_mask = p_mask.to(p.dtype)
        q_mask = q_mask.to(q.dtype)
        e = torch.bmm(p, q.transpose(1, 2)) * self.scale  # [B, Lp, Lq]
        p_aligned = torch.bmm(F.softmax(e.masked_fill(q_mask.unsqueeze(1) == 0, _MASK_VALUE), dim=2), q)
        q_aligned = torch.bmm(F.softmax(e.masked_fill(p_mask.unsqueeze(2) == 0, _MASK_VALUE), dim=1).transpose(1, 2), p)
        vp = masked_pool(self.compare(torch.cat([p, p_aligned, p - p_aligned, p * p_aligned], -1)), p_mask)
        vq = masked_pool(self.compare(torch.cat([q, q_aligned, q - q_aligned, q * q_aligned], -1)), q_mask)
        return self.classifier(torch.cat([vp, vq], dim=-1))


class AdaptiveSystem1Classifier(nn.Module):
    """Orchestration bi-encodeur / cross-encodeur autour d'un backbone TextCNN partagé."""

    def __init__(self, config: TextCNNConfig):
        super().__init__()
        self.config = config
        self.backbone = TextCNNBackbone(config)
        self.bi_encoder = BiEncoder(self.backbone.out_dim, config.embedding_dim)
        self.cross_encoder = CrossEncoder(self.backbone.out_dim, config.dropout)

    @property
    def threshold(self) -> int:
        return self.config.threshold

    @property
    def top_k(self) -> int:
        return self.config.top_k

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (caractéristiques par token [B, L, H], vecteur du bi-encodeur [B, D])."""
        tokens = self.backbone(input_ids, attention_mask)
        return tokens, self.bi_encoder(tokens, attention_mask)

    def forward(self, p_ids, p_mask, q_ids, q_mask) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Paires alignées (contexte i, option i) -> (logits NLI [B, 3], vecteur contexte, vecteur option)."""
        p_tok, p_emb = self.encode(p_ids, p_mask)
        q_tok, q_emb = self.encode(q_ids, q_mask)
        return self.cross_encoder(p_tok, p_mask, q_tok, q_mask), p_emb, q_emb

    @torch.no_grad()
    def predict(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        options_ids: torch.Tensor,
        options_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un contexte [1, Lp] face à n options [n, Lq].

        Renvoie (logits NLI [n, 3], indices des options évaluées par le cross-encodeur).
        Les options écartées par le bi-encodeur reçoivent ``PRUNED_LOGITS``.
        """
        n = options_ids.size(0)
        c_tok, c_emb = self.encode(context_ids, context_mask)
        o_tok, o_emb = self.encode(options_ids, options_mask)
        if n <= self.threshold:
            selected = torch.arange(n, device=options_ids.device)
        else:
            bi_scores = (c_emb @ o_emb.T).squeeze(0)
            selected = torch.topk(bi_scores, k=min(self.top_k, n)).indices
        k = selected.numel()
        cross = self.cross_encoder(
            c_tok.expand(k, -1, -1), context_mask.expand(k, -1), o_tok[selected], options_mask[selected]
        )
        logits = torch.tensor(PRUNED_LOGITS, dtype=cross.dtype, device=cross.device).repeat(n, 1)
        logits[selected] = cross
        return logits, selected

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
