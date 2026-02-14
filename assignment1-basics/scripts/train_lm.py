import os
from os import path

from cs336_basics.train import train

data_root = "./data"
dataset_root = "dataset"
tokenizer_root = "tokenizer"
lm_root = "lm"
lm_path = path.join(data_root, lm_root)
os.makedirs(lm_path, exist_ok=True)

tokenizer_path = path.join(data_root, tokenizer_root)
vocab_path = path.join(tokenizer_path, "vocab.json")
merges_path = path.join(tokenizer_path, "merges.txt")

dataset_path = path.join(data_root, dataset_root, "TinyStories-train.txt")


def main():
    train(
        dataset_path=dataset_path,
        output_dir=lm_path,
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=["<|endoftext|>"],
        batch_size=128,
        max_iter=1280000,
        save_interval=2000,
    )


if __name__ == "__main__":
    main()
