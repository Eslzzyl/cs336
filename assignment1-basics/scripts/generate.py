from os import path

import torch

from cs336_basics.decode import decode
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

data_root = "./data"
tokenizer_root = "tokenizer"
lm_root = "lm"
lm_name = "ckpt_final.pt"

tokenizer_path = path.join(data_root, tokenizer_root)
vocab_path = path.join(tokenizer_path, "vocab.json")
merges_path = path.join(tokenizer_path, "merges.txt")

checkpoint_path = path.join(data_root, lm_root, lm_name)

endoftext_token = "<|endoftext|>"


def main():
    # decide target device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"target device: {device}")

    # initialize model and tokenizer
    model = TransformerLM().to(device)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, [endoftext_token])
    endoftext_token_id = tokenizer.encode(endoftext_token)[0]
    print("model initialized")

    prompt = input("Input prompt:")
    encoded_prompt = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor(encoded_prompt, dtype=torch.long, device=device)

    result = decode(
        model=model,
        prompt=prompt_tensor,
        endoftext_token=endoftext_token_id,
        max_tokens=16,
        temperature=0.7,
        top_k=20,
        top_p=0.6,
    )

    result_token_list = result.tolist()
    decoded_result = tokenizer.decode(result_token_list)
    print(decoded_result)


if __name__ == "__main__":
    main()
