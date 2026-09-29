import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNBackbone(nn.Module):
    """Moteur d'extraction de caractéristiques basé sur CNN 1D."""
    def __init__(self, vocab_size: int, embed_dim: int = 128, num_filters: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutions sur 3, 4 et 5 mots
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in (3, 4, 5)
        ])
        self.out_dim = num_filters * len(self.convs) # 64 * 3 = 192 features

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids).transpose(1, 2) # [batch, embed_dim, seq_len]
        
        # Application des filtres + Max Pooling
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(-1)).squeeze(-1) for c in conv_outputs]
        
        return torch.cat(pooled, dim=1) # [batch, 192]

class BiEncoder(nn.Module):
    def __init__(self, backbone: TextCNNBackbone, output_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(backbone.out_dim, output_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        features = self.backbone(input_ids)
        output = self.projection(features)
        return F.normalize(output, p=2, dim=-1)

class CrossEncoder(nn.Module):
    def __init__(self, backbone: TextCNNBackbone, hidden_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(backbone.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, pair_ids: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pair_ids)
        logits = self.classifier(features)
        return logits.squeeze(-1)
    
class AdaptiveSystem1Classifier(nn.Module):
    def __init__(self, vocab_size: int, threshold: int = 10, top_k: int = 5):
        super().__init__()
        self.threshold = threshold
        self.top_k = top_k
        
        # Backbone TextCNN partagé
        self.backbone = TextCNNBackbone(vocab_size=vocab_size, embed_dim=128, num_filters=64)
        
        self.bi_encoder = BiEncoder(backbone=self.backbone)
        self.cross_encoder = CrossEncoder(backbone=self.backbone)

    def predict(self, context_ids: torch.Tensor, options_ids: torch.Tensor, pairs_ids: torch.Tensor):
        num_options = options_ids.size(0)

        if num_options <= self.threshold:
            cross_logits = self.cross_encoder(pairs_ids)
            probs = F.softmax(cross_logits, dim=-1)
            indices = torch.arange(num_options, device=options_ids.device)
            return probs, indices

        ctx_emb = self.bi_encoder(context_ids)
        opt_embs = self.bi_encoder(options_ids)
        bi_scores = torch.matmul(ctx_emb, opt_embs.T).squeeze(0)

        k_val = min(self.top_k, num_options)
        _, top_k_indices = torch.topk(bi_scores, k=k_val)

        selected_pairs = pairs_ids[top_k_indices]
        cross_logits = self.cross_encoder(selected_pairs)
        probs = F.softmax(cross_logits, dim=-1)

        return probs, top_k_indices
    
import os
import time
import psutil
import torch

def benchmark_inference(model, context_ids, options_ids, pairs_ids):
    process = psutil.Process(os.getpid())
    
    # Force le nettoyage du garbage collector PyTorch avant la mesure
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 1. Mesure de l'empreinte mémoire avant inférence
    ram_before_mb = process.memory_info().rss / (1024 * 1024)
    vram_before_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # 2. Chronométrage de la passe avant (Inférence)
    start_time = time.perf_counter()
    
    with torch.no_grad(): # Désactive les gradients pour économiser la mémoire
        probs, indices = model.predict(context_ids, options_ids, pairs_ids)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize() # Attente de la fin des opérations asynchrones GPU
        
    end_time = time.perf_counter()

    # 3. Mesure de l'empreinte mémoire après inférence
    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    vram_after_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    metrics = {
        "latency_ms": round((end_time - start_time) * 1000, 3),
        "ram_total_mb": round(ram_after_mb, 2),
        "ram_delta_mb": round(ram_after_mb - ram_before_mb, 2),
        "vram_used_mb": round(vram_after_mb, 2)
    }

    return probs, indices, metrics

def main():

    # Initialisation du modèle
    vocab_size = 10_000
    model = AdaptiveSystem1Classifier(vocab_size=vocab_size, threshold=10, top_k=5)
    model.eval()

    # Dummy data pour simuler une requête avec 15 options (déclenche la branche hybride)
    context_ids = torch.randint(1, vocab_size, (1, 20))
    options_ids = torch.randint(1, vocab_size, (15, 10))
    pairs_ids = torch.randint(1, vocab_size, (15, 30))

    # Exécution du benchmark
    probs, top_indices, stats = benchmark_inference(model, context_ids, options_ids, pairs_ids)

    print(f"Latence        : {stats['latency_ms']} ms")
    print(f"Mémoire RAM    : {stats['ram_total_mb']} Mo (Delta: {stats['ram_delta_mb']} Mo)")
    print(f"Mémoire VRAM   : {stats['vram_used_mb']} Mo")