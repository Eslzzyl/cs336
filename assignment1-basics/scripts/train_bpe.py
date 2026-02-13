import json
from os import path

from cs336_basics.bpe import train_bpe

data_root = "./data"
dataset_path = path.join(data_root, "TinyStories-valid.txt")
vocab_path = path.join(data_root, "vocab.json")
merges_path = path.join(data_root, "merges.txt")


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
