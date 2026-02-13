import json
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe import PAT, split_special_tokens


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> list[list[bytes]]:
    """
    执行预分词过程。注意这里和 bpe.py 中不同，不能排除 special_tokens
    """
    if special_tokens:
        # 为了让正则优先匹配更长的 special token（避免被其子串先匹配），按长度降序排序
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
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

    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Args:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab = {v: k.encode("utf-8") for k, v in vocab_data.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for merge in f:
                parts = merge.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))
        self.__init__(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Step 1: pre-tokenize
        pre_tokens = pre_tokenize(text, self.special_tokens)

        # Step 2: apply merges
        # Build reverse vocab mapping bytes -> id for fast lookup
        reverse_vocab_dict: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        ids: list[int] = []
        for curr_token in pre_tokens:
            # 重复扫描 merges
            while True:
                merged_this_round = False
                for a, b in self.merges:
                    idx = -1
                    for i in range(len(curr_token) - 1):
                        if curr_token[i] == a and curr_token[i + 1] == b:
                            idx = i
                            break
                    if idx != -1:
                        # 说明找到了匹配的 merge，执行
                        new_token = curr_token[idx] + curr_token[idx + 1]
                        curr_token = curr_token[:idx] + [new_token] + curr_token[idx + 2 :]
                        merged_this_round = True
                        # 结束并重新从头开始扫描 merges
                        break
                # 如果未能找到匹配的 merge 规则，则说明这个 token 已经合并完成了
                if not merged_this_round:
                    break
            for token in curr_token:
                if token in reverse_vocab_dict:
                    ids.append(reverse_vocab_dict[token])
                else:
                    # 词表中没有找到对应的 token
                    # 理论上这不应该发生，只要 vocab 和 merges 文件是一致的。
                    # 但是稳妥起见设置一个 fallback 规则
                    for b in token:
                        single = bytes([b])
                        if single in reverse_vocab_dict:
                            ids.append(reverse_vocab_dict[single])
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
