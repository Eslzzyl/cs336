import json
import os
from os import path

from cs336_basics.bpe import train_bpe

data_root = "./data"
dataset_root = "dataset"
tokenizer_root = "tokenizer"
tokenizer_path = path.join(data_root, tokenizer_root)
os.makedirs(tokenizer_path, exist_ok=True)

dataset_path = path.join(data_root, dataset_root, "TinyStories-train.txt")
vocab_path = path.join(tokenizer_path, "vocab.json")
merges_path = path.join(tokenizer_path, "merges.txt")


def main():
    vocab, merges = train_bpe(dataset_path, vocab_size=10000, special_tokens=["<|endoftext|>"])
    reverse_vocab_dict: dict[str, int] = {v.decode("latin-1"): k for k, v in vocab.items()}
    merges_to_write = [(a + b" " + b).decode("latin-1") for a, b in merges]

    with open(vocab_path, "w") as f:
        json.dump(reverse_vocab_dict, f, separators=(",", ":"), ensure_ascii=False)
    with open(merges_path, "w") as f:
        f.write("\n".join(merges_to_write))


if __name__ == "__main__":
    main()
