import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, vocab_size = logits.shape
    if targets.shape != (batch_size,):
        raise ValueError(f"Targets shape {targets.shape} does not match logits shape {logits.shape}")
    log_probs = torch.log_softmax(logits, dim=-1)
    true_class_log_probs = log_probs[torch.arange(batch_size), targets]
    nll_per_sample = -true_class_log_probs
    return nll_per_sample.mean()

