from __future__ import annotations

import torch
import torch.nn.functional as F


def recall_at_k_surrogate_loss(similarity: torch.Tensor, labels: torch.Tensor, k: int = 1, tau_1: float = 1.0, tau_2: float = 0.01) -> torch.Tensor:
    """Differentiable surrogate for Recall@k.

    similarity: [B, B] similarity matrix.
    labels: [B] target indices (normally arange(B) in paired training).
    """
    batch_size = similarity.size(0)
    positive_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    negative_mask = ~positive_mask

    total = similarity.new_tensor(0.0)
    for i in range(batch_size):
        sim_query = similarity[i]
        pos_sim = sim_query[positive_mask[i]]
        neg_sim = sim_query[negative_mask[i]]
        surrogate = similarity.new_tensor(0.0)
        for pos in pos_sim:
            rank = torch.sum(torch.sigmoid((neg_sim - pos) / tau_2))
            surrogate = surrogate + torch.sigmoid((k - 1 - rank) / tau_1)
        if len(pos_sim) > 0:
            surrogate = surrogate / len(pos_sim)
        total = total + (1.0 - surrogate)
    return total / batch_size



def retrieval_loss(
    query_features: torch.Tensor,
    target_features: torch.Tensor,
    scale: float = 100.0,
    recall_weight_at_1: float = 0.4,
    recall_weight_at_5: float = 0.15,
) -> torch.Tensor:
    labels = torch.arange(query_features.size(0), device=query_features.device, dtype=torch.long)
    cosine = query_features @ target_features.T
    logits = scale * cosine
    ce = F.cross_entropy(logits, labels)
    r1 = recall_at_k_surrogate_loss(cosine, labels, k=1)
    r5 = recall_at_k_surrogate_loss(cosine, labels, k=5)
    return ce + recall_weight_at_1 * r1 + recall_weight_at_5 * r5
