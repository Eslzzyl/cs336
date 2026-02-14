import os
from os import path
from pathlib import Path

import numpy as np
import torch
from einx import rearrange
from tqdm import tqdm

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.dataset_cache import build_or_load_encoded_dataset_memmap
from cs336_basics.gradient import gradient_clipping
from cs336_basics.loss import cross_entropy
from cs336_basics.lr_scheduler import lr_cosine_schedule
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer


def train(
    dataset_path: str,
    output_dir: str,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str] | None = None,
    max_iter: int = 10000,
    save_interval: int = 1000,
    validation_interval: int = 1000,
    batch_size: int = 8,
    context_length: int = 16,
    d_model: int = 64,
    num_heads: int = 4,
    d_ff: int = 128,
    vocab_size: int = 10000,
    num_layers: int = 3,
    rope_theta: float = 10000.0,
    max_learning_rate: float = 1e-3,
    min_learning_rate: float = 1e-5,
    warmup_iters: int = 100,
    cosine_cycle_iters: int = 10000,
    max_grad_norm: float = 1.0,
    ckpt_path: Path | str | None = None,
):
    # decide target device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"target device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # initialize model, optimizer and tokenizer
    model = TransformerLM().to(device)
    optimizer = AdamW(model.parameters(), lr=max_learning_rate)
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    print("model initialized")

    # (optional) load checkpoint
    if ckpt_path:
        print(f"Loading checkpoint at {ckpt_path}")
        load_checkpoint(ckpt_path, model, optimizer)
        print("checkpoint loaded")

    # tokenize dataset (use cached memmap if available)
    dataset_root = Path(dataset_path).parent
    cache_filename = path.splitext(path.basename(dataset_path))[0] + ".npy"
    cache_path = path.join(dataset_root, cache_filename)

    # using np.int16 since we are using vocab size = 10000, less than 65536
    dataset_memmap = build_or_load_encoded_dataset_memmap(dataset_path, tokenizer, cache_path, dtype=np.int16)
    # convert to plain numpy array view (memmap supports slicing etc.)
    dataset = np.asarray(dataset_memmap)
    print("dataset created")

    # main training loop
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()

        # get data
        input_tensor, target_tensor = get_batch(dataset, batch_size, context_length, device)

        # forward pass
        logits = model(input_tensor)

        # resize tensors
        logits = rearrange("batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size", logits)
        target_tensor = rearrange("batch_size seq_len -> (batch_size seq_len)", target_tensor)

        # calculate loss
        loss = cross_entropy(logits, target_tensor)
        # backward pass
        loss.backward()

        # gradient clipping
        gradient_clipping(model.parameters(), max_grad_norm)

        # update learning rate
        lr = lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # optimize
        optimizer.step()

        if (it + 1) % save_interval == 0:
            ckpt_path_now = path.join(output_dir, f"ckpt_{it + 1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_path_now)
            print(f"checkpoint saved to {ckpt_path_now}")

    final_ckpt_path = path.join(output_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, max_iter, final_ckpt_path)
