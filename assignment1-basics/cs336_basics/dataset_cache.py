"""
Utilities to build / load an encoded dataset cache using numpy memmap (.npy).

Function:
- build_or_load_encoded_dataset_memmap(dataset_path, tokenizer, cache_path, chunk_chars=200_000, dtype=np.int32, overwrite=False)

Behavior:
- If cache_path exists, loads it via np.load(mmap_mode='r+').
- Otherwise, reads the dataset file in chunks (accumulating lines up to chunk_chars),
  encodes each chunk with `tokenizer.encode(...)`, collects id-chunks, creates a
  .npy memmap and writes the concatenated ids into it (atomic write via .tmp + os.replace).
"""

from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm


def build_or_load_encoded_dataset_memmap(
    dataset_path: str,
    tokenizer,
    cache_path: str,
    chunk_chars: int = 200_000,
    dtype: type = np.int32,
    overwrite: bool = False,
) -> np.ndarray:
    """
    Build or load a cached encoded dataset as a numpy .npy memmap.

    Args:
        dataset_path: path to the raw text dataset (utf-8).
        tokenizer: instance providing `encode(str) -> list[int]` (and optimized).
        cache_path: path to .npy file to create / load.
        chunk_chars: approximate chars per chunk to feed into tokenizer.encode.
        dtype: integer dtype for storing token ids (np.int32 recommended).
        overwrite: if True and cache_path exists, rebuild it.

    Returns:
        A numpy array (np.memmap) containing the concatenated token ids.
    """
    # If cache exists and not overwrite, load via mmap
    if os.path.exists(cache_path) and not overwrite:
        print(f"Loading encoded dataset memmap from {cache_path}")
        return np.load(cache_path, mmap_mode="r+")

    print(f"Building encoded dataset and writing to {cache_path} (chunk_chars={chunk_chars})")

    # Step 1: iterate file and produce encoded chunks
    chunks: list[np.ndarray] = []
    total_tokens = 0

    with open(dataset_path, encoding="utf-8") as f:
        buf_lines = []
        buf_chars = 0
        for line in tqdm(f, desc="encoding dataset lines", unit="line"):
            buf_lines.append(line)
            buf_chars += len(line)
            if buf_chars >= chunk_chars:
                text = "".join(buf_lines)
                ids = tokenizer.encode(text)
                arr = np.array(ids, dtype=dtype)
                chunks.append(arr)
                total_tokens += arr.size
                buf_lines = []
                buf_chars = 0

        # final partial chunk
        if buf_lines:
            text = "".join(buf_lines)
            ids = tokenizer.encode(text)
            arr = np.array(ids, dtype=dtype)
            chunks.append(arr)
            total_tokens += arr.size

    # Step 2: create .npy memmap as a temp file and copy chunks into it
    tmp_path = cache_path + ".tmp"
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)

    print(f"Allocating memmap file of length {total_tokens} tokens (dtype={dtype})")
    mem = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype, shape=(total_tokens,))

    pos = 0
    for arr in tqdm(chunks, desc="writing token chunks to memmap", unit="chunk"):
        end = pos + arr.size
        mem[pos:end] = arr
        pos = end

    mem.flush()
    del mem  # close memmap file handle before replace

    # Atomic replace
    os.replace(tmp_path, cache_path)
    print(f"Cache saved to {cache_path}")

    # Load as mmap for returning
    return np.load(cache_path, mmap_mode="r+")
