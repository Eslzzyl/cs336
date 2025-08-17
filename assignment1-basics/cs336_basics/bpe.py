import regex as re
import os
from collections import Counter
from typing import BinaryIO
from multiprocessing import Pool

# 编译正则表达式以提高效率
# GPT-2 预分词正则表达式
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
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


def split_special_tokens(text: str, special_tokens: list[str]):
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


def pre_tokenize_for_chunk(args: tuple[str | os.PathLike, list[str], int, int]) -> Counter:
    pre_token_counts = Counter()

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
                pre_token_counts[byte_tokens] += 1
    return pre_token_counts


def pre_tokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> Counter:
    pre_token_counts = Counter()

    # 一次性读取文件并确定边界
    with open(input_path, mode="rb") as f:
        # 根据CPU核心数确定分块数量，但设置上限避免过多小任务的开销
        num_chunks = min(os.cpu_count() * 2, 16)
        boundaries = find_chunk_boundaries(f, num_chunks, "<|endoftext|>".encode("utf-8"))  # noqa: UP012

        args = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args.append((input_path, special_tokens, start, end))

    with Pool(os.cpu_count()) as pool:
        for chunk_result in pool.imap_unordered(pre_tokenize_for_chunk, args):
            pre_token_counts.update(chunk_result)

    return pre_token_counts


def get_pair_counts(words: Counter) -> Counter:
    pair_counter = Counter()
    for word, frequency in words.items():
        word_len = len(word)
        # 遍历每个词中相邻的 token 对
        for i in range(word_len - 1):
            pair = (word[i], word[i + 1])
            # 累加该对的出现次数
            pair_counter[pair] += frequency
    return pair_counter


def merge_pair(words: Counter, pair_to_merge: tuple[bytes, bytes]) -> Counter:
    new_words = Counter()
    for word_tuple, frequency in words.items():
        # 将 tuple 转换为列表以便修改
        current_word_list = list(word_tuple)
        # 构建新的 word 序列
        merged_word_list = []
        i = 0
        while i < len(current_word_list):
            if (
                i + 1 < len(current_word_list)
                and current_word_list[i] == pair_to_merge[0]
                and current_word_list[i + 1] == pair_to_merge[1]
            ):
                new_token = pair_to_merge[0] + pair_to_merge[1]
                merged_word_list.append(new_token)
                i += 2
            else:
                merged_word_list.append(current_word_list[i])
                i += 1

        new_words[tuple(merged_word_list)] += frequency
    return new_words


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

    # 使用集合记录已有词汇，避免重复检查
    vocab_set = set(vocab)
    merges: list[tuple[bytes, bytes]] = []

    # 定期打印进度
    progress_interval = max(1, (vocab_size - len(vocab)) // 20)

    while len(vocab) < vocab_size:
        # 获取每个词中相邻的 token 对出现的次数
        pair_counts = get_pair_counts(words)

        # 空 Counter 将被视为 False，这表示语料库中所有的词都无法再分割出 token 了
        if not pair_counts:
            break

        best_pair = pair_counts.most_common(1)[0][0]
        # 如果这个 pair 已经在词表中，跳过
        if best_pair in vocab_set:
            continue

        # 合并产生新 token
        new_token = best_pair[0] + best_pair[1]

        # 将新 token 加入词表和集合
        vocab.append(new_token)
        vocab_set.add(new_token)

        # 将本次的 pair 加入合并记录
        merges.append(best_pair)

        words = merge_pair(words, best_pair)

        if len(vocab) % progress_interval == 0:
            print(f"Vocab size: {len(vocab)}/{vocab_size}, Merging: {best_pair}")

    print(f"Final vocab size: {len(vocab)}")
    return {i: v for i, v in enumerate(vocab)}, merges
