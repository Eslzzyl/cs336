from os import path
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
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
    ckpt_path: Path | str | None = None,
):
    # decide target device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # initialize model, optimizer and tokenizer
    model = TransformerLM()
    optimizer = AdamW(model.parameters())
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    # (optional) load checkpoint
    if ckpt_path:
        print(f"Loading checkpoint at {ckpt_path}")
        load_checkpoint(ckpt_path, model, optimizer)

    # tokenize dataset
    with open(dataset_path, encoding="utf-8") as f:
        dataset_str = f.read()
    dataset = np.array(tokenizer.encode(dataset_str))

    # main training loop
    for _ in tqdm(range(max_iter)):
        # get data
        input_tensor, target_tensor = get_batch(dataset, batch_size, context_length, device)

        # forward pass
        logits = model(input_tensor)

        # calculate loss
        loss = cross_entropy(logits, target_tensor)
        # backward pass
        loss.backward()

        # optimize
        optimizer.step()

    final_ckpt_path = path.join(output_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, max_iter, final_ckpt_path)
