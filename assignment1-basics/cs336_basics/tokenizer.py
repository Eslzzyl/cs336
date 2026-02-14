from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import cast

import regex as re

from cs336_basics.bpe import PAT


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> list[list[bytes]]:
    """
    执行预分词过程。注意这里和 bpe.py 中不同，不能排除 special_tokens
    """
    if special_tokens:
        # 为了让正则优先匹配更长的 special token（避免被其子串先匹配），按长度降序排序
        sorted_specials = cast(list[str], sorted(special_tokens, key=len, reverse=True))
        escaped = [re.escape(s) for s in sorted_specials]
        pattern = "(" + "|".join(escaped) + ")"
        parts = re.split(pattern, text)
        special_set = set(special_tokens)
    else:
        parts = [text]
        special_set = set()

    pre_tokens: list[list[bytes]] = []
    for part in parts:
        if not part:
            continue
        # 如果 part 是 exact match 的 special token，则把整个 token 当成一个 bytes 单元（不拆字节）
        if part in special_set:
            pre_tokens.append([part.encode("utf-8")])
        else:
            for match in PAT.finditer(part):
                pre_token = match.group(0)
                byte_tokens = [bytes([b]) for b in pre_token.encode("utf-8")]
                pre_tokens.append(byte_tokens)
    return pre_tokens


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Args:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # Build reverse vocab mapping bytes -> id for fast lookup
        self._reverse_vocab: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        self._merge_ranks: dict[tuple[bytes, bytes], int] = {}
        for idx, pair in enumerate(self.merges):
            self._merge_ranks[pair] = idx

    @staticmethod
    def from_files(vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Tokenizer:
        """
        Args:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab = {v: k.encode("latin-1") for k, v in vocab_data.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for merge in f:
                parts = merge.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("latin-1"), parts[1].encode("latin-1")))
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Step 1: pre-tokenize
        pre_tokens = pre_tokenize(text, self.special_tokens)

        ids: list[int] = []
        for curr_token in pre_tokens:
            # curr_token: list[bytes], 每个 bytes 是单个字节字符（b'a' 等）
            # 使用 merges rank 来进行贪心合并：每次只在当前 token 的相邻 pairs 中
            # 选择优先级最高（rank 最小）的 pair 合并直到没有可合并对为止。

            if len(curr_token) == 0:
                continue

            # 转为可变列表
            token_list = list(curr_token)

            while True:
                # 生成所有相邻对并找出在 merge_rules 中优先级最小的那个
                best_idx = -1
                best_rank = None
                # pairs_count = len(token_list) - 1
                for i in range(len(token_list) - 1):
                    pair = (token_list[i], token_list[i + 1])
                    rank = self._merge_ranks.get(pair)
                    if rank is not None:
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_idx = i
                if best_idx == -1:
                    break
                # 合并 best_idx 和 best_idx+1
                new_token = token_list[best_idx] + token_list[best_idx + 1]
                token_list[best_idx : best_idx + 2] = [new_token]

            # 将 token_list 转换为 ids（使用已缓存的 reverse vocab）
            for token in token_list:
                if token in self._reverse_vocab:
                    ids.append(self._reverse_vocab[token])
                else:
                    # fallback：把 token 拆回单字节
                    # 理论上这不应该发生，只要 vocab 和 merges 是一致的
                    for b in token:
                        single = bytes([b])
                        if single in self._reverse_vocab:
                            ids.append(self._reverse_vocab[single])
                        else:
                            raise KeyError(f"Byte token {single!r} not found in vocab")
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decode_bytes = b""
        for id in ids:
            decode_bytes += self.vocab[id]
        # auto replace the malformed bytes with the oﬀicial Unicode replacement character U+FFFD
        return decode_bytes.decode("utf-8", errors="replace")
