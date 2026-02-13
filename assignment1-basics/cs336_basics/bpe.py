import os
from multiprocessing import Pool
from typing import BinaryIO

import regex as re
from tqdm import tqdm, trange

# 编译正则表达式以提高效率
# GPT-2 预分词正则表达式
PAT: re.Pattern[str] = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    This function is copied from pretokenization_example.py and the file is deleted now.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    根据 special tokens 分割文本，并返回纯文本 chunk 列表。
    这些 chunk 可以安全地进行后续的预分词。
    """
    if not special_tokens:
        return [text]

    # 1. 转义所有特殊标记
    # re.escape() 会处理特殊字符，例如 <|endoftext|> 会变成 <\|endoftext\|>
    escaped_tokens = [re.escape(token) for token in special_tokens]

    # 2. 构建正则表达式模式：使用 '|' 连接所有转义后的标记
    # 例如：<\|endoftext\|>|\[CLS\]|\[SEP\]
    delimiter_pattern = "|".join(escaped_tokens)

    # 3. 使用 re.split 分割文本
    # re.split 会移除匹配到的分隔符，并返回分隔符之间的内容
    raw_chunks = re.split(delimiter_pattern, text)

    # 4. 过滤掉可能出现的空字符串（例如，如果分隔符在开头、结尾或连续出现）
    # 示例: "TOKEN1<SP_TOKEN>TOKEN2" -> ['TOKEN1', 'TOKEN2']
    # 示例: "<SP_TOKEN>TOKEN1<SP_TOKEN>TOKEN2" -> ['', 'TOKEN1', 'TOKEN2'] -> ['TOKEN1', 'TOKEN2']
    # 示例: "TOKEN1<SP_TOKEN>" -> ['TOKEN1', ''] -> ['TOKEN1']
    text_chunks = [chunk for chunk in raw_chunks if chunk]

    return text_chunks


def pre_tokenize_for_chunk(args: tuple[str | os.PathLike, list[str], int, int]) -> dict[tuple[bytes, ...], int]:
    pre_token_counts: dict[tuple[bytes, ...], int] = {}

    input_path, special_tokens, start, end = args
    with open(input_path, mode="rb") as f:
        f.seek(start)
        # 截取当前线程的 chunk
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 根据 special tokens 进行分割，确保 special tokens 被移除且这些 token 两侧的文本不会被粘连
        splitted_chunks = split_special_tokens(chunk, special_tokens)
        # 针对分割出的每个 chunk 进行预分词
        for splitted_chunk in splitted_chunks:
            for match in PAT.finditer(splitted_chunk):
                pre_token = match.group(0)
                byte_tokens = tuple(bytes([b]) for b in pre_token.encode("utf-8"))
                pre_token_counts[byte_tokens] = pre_token_counts.get(byte_tokens, 0) + 1
    return pre_token_counts


def pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    pre_token_counts: dict[tuple[bytes, ...], int] = {}

    # 一次性读取文件并确定边界
    with open(input_path, mode="rb") as f:
        # 根据CPU核心数确定分块数量，但设置上限避免过多小任务的开销
        cpu_count = os.cpu_count()
        if not cpu_count:
            raise ValueError("indeterminable CPU count")
        num_chunks = min(cpu_count * 2, 16)
        boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))  # noqa: UP012

        args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args.append((input_path, special_tokens, start, end))

    with Pool(cpu_count) as pool:
        for chunk_result in tqdm(
            pool.imap_unordered(pre_tokenize_for_chunk, args), total=len(args), desc="pre-tokenize chunks", unit="chunk"
        ):
            for token, count in chunk_result.items():
                pre_token_counts[token] = pre_token_counts.get(token, 0) + count

    return pre_token_counts


def get_pair_counts(words: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    统计所有相邻 token 对的频率
    """
    pair_counter: dict[tuple[bytes, bytes], int] = {}
    for word, frequency in words.items():
        word_len = len(word)
        # 遍历每个词中相邻的 token 对
        for i in range(word_len - 1):
            pair = (word[i], word[i + 1])
            # 累加该对的出现次数
            pair_counter[pair] = pair_counter.get(pair, 0) + frequency  # {(b'h', b'e'): 5, (b'l', b'l'): 10, ...}
    return pair_counter


def merge_pair(
    words: dict[tuple[bytes, ...], int], target_pair: tuple[bytes, bytes], pair_counts: dict[tuple[bytes, bytes], int]
) -> dict[tuple[bytes, ...], int]:
    new_words: dict[tuple[bytes, ...], int] = {}
    new_token = target_pair[0] + target_pair[1]

    for word, freq in words.items():
        if len(word) < 2:
            new_words[word] = freq
            continue

        # Check if the target_pair is in the word
        if target_pair not in zip(word, word[1:]):
            new_words[word] = freq
            continue

        # 1. Deduct old pair counts
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                pair_counts.pop(pair, None)

        # 2. Form new word
        new_word_list = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == target_pair:
                new_word_list.append(new_token)
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1
        new_word = tuple(new_word_list)
        new_words[new_word] = new_words.get(new_word, 0) + freq

        # 3. Add new pair counts
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq

    return new_words


def get_best_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """
    获取最高频的 token pair
    - 如果最高频的只有一个，则返回
    - 如果有多个，则按照字典顺序排序
    """
    return max(pair_counts.keys(), key=lambda p: (pair_counts[p], p[0], p[1]))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 生成初始包含 256 个字节字符
    vocab: list[bytes] = [bytes([i]) for i in range(0, 256)]
    # 添加 special_tokens
    vocab += [special_token.encode("utf-8") for special_token in special_tokens]

    # 预分词
    words = pre_tokenize(input_path, special_tokens)

    merges: list[tuple[bytes, bytes]] = []
    # 获取每个词中相邻的 token 对出现的次数
    # 仅在初始化时全量统计一次
    pair_counts = get_pair_counts(words)

    initial_vocab_len = len(vocab)
    # 计划新增 token 数
    planned_new = max(0, vocab_size - initial_vocab_len)

    # 使用 trange 显示合并进度；如果中途 pair_counts 为空会提前中断
    for _ in trange(planned_new, desc="BPE merges", unit="merge"):
        # 空 dict 将被视为 False，这表示语料库中所有的词都无法再分割出 token 了
        if not pair_counts:
            break

        # 获取需要合并的下一个 pair
        best_pair = get_best_pair(pair_counts)

        # 合并产生新 token
        new_token = best_pair[0] + best_pair[1]

        # 将新 token 加入词表和集合
        vocab.append(new_token)

        # 将本次的 pair 加入合并记录
        merges.append(best_pair)

        words = merge_pair(words, best_pair, pair_counts)

    print(f"Final vocab size: {len(vocab)}")
    return {i: v for i, v in enumerate(vocab)}, merges
