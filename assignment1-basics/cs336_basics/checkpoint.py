import torch
import os
from typing import BinaryIO, IO


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(ckpt, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    ckpt = torch.load(src, weights_only=True)
    if "model" not in ckpt:
        raise ValueError("The checkpoint doesn't contain 'model' key")
    if "optimizer" not in ckpt:
        raise ValueError("The checkpoint doesn't contain 'optimizer' key")
    if "iteration" not in ckpt:
        raise ValueError("The checkpoint doesn't contain 'iteration' key")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]
