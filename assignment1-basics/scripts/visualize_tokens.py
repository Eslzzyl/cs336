"""
终端分词可视化工具
使用ANSI颜色码在终端中以彩色色块显示分词结果
"""

from os import path

from cs336_basics.tokenizer import Tokenizer

# 默认路径配置
DATA_ROOT = "./data"
TOKENIZER_ROOT = "tokenizer"
VOCAB_PATH = path.join(DATA_ROOT, TOKENIZER_ROOT, "vocab.json")
MERGES_PATH = path.join(DATA_ROOT, TOKENIZER_ROOT, "merges.txt")


def get_color_for_token(token_id: int) -> tuple[int, int]:
    """
    根据token ID生成ANSI 256色模式的背景色和前景色
    返回: (背景色代码, 前景色代码)

    使用柔和的颜色方案，确保可读性
    """
    # 使用256色模式中的柔和颜色范围
    # 颜色块范围 16-231: 6x6x6 色立方
    # 为了柔和效果，选择中间亮度的颜色 (值范围 1-4)

    # 使用token_id生成伪随机但一致的颜色
    # 跳过太暗(0)和太亮(5)的颜色值
    color_palette = []
    for r in [1, 2, 3, 4]:
        for g in [1, 2, 3, 4]:
            for b in [1, 2, 3, 4]:
                # 16 + 36*r + 6*g + b
                color_palette.append(16 + 36 * r + 6 * g + b)

    # 根据token_id选择颜色
    bg_color = color_palette[token_id % len(color_palette)]

    # 前景色：使用黑色或白色，根据背景亮度决定
    # 简单策略：使用亮白色 (231) 或深灰色 (238)
    fg_color = 232 if sum(divmod(bg_color - 16, 36)) % 2 == 0 else 255

    return bg_color, fg_color


def escape_special_chars(text: str) -> str:
    """
    将特殊字符转换为可读的转义表示
    """
    result = []
    for char in text:
        if char == "\n":
            result.append("↵")
        elif char == "\t":
            result.append("→")
        elif char == " ":
            result.append("·")
        elif char == "\r":
            result.append("⏎")
        elif ord(char) < 32 or ord(char) == 127:
            # 其他控制字符
            result.append(f"^{chr(ord(char) + 64)}")
        else:
            result.append(char)
    return "".join(result)


def colorize_token(token_bytes: bytes, token_id: int) -> str:
    """
    为token添加ANSI颜色
    """
    # 将bytes解码为字符串
    try:
        token_text = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # 如果解码失败，显示十六进制
        token_text = token_bytes.hex()

    # 转义特殊字符
    token_text = escape_special_chars(token_text)

    # 获取颜色
    bg_color, fg_color = get_color_for_token(token_id)

    # ANSI颜色码：
    # \033[38;5;{fg}m - 前景色
    # \033[48;5;{bg}m - 背景色
    # \033[0m - 重置
    return f"\033[38;5;{fg_color}m\033[48;5;{bg_color}m{token_text}\033[0m"


def visualize_tokens(tokenizer: Tokenizer, text: str):
    """
    可视化分词结果
    """
    if not text.strip():
        print("输入为空，请重新输入。")
        return

    # 分词
    token_ids = tokenizer.encode(text)

    # 显示原始文本
    print(f"\n\033[1m原文:\033[0m {repr(text)}")

    # 显示彩色token
    print("\033[1m分词:\033[0m ", end="")
    for token_id in token_ids:
        token_bytes = tokenizer.vocab[token_id]
        colored_token = colorize_token(token_bytes, token_id)
        print(colored_token, end="")
    print()  # 换行

    # 显示统计信息
    print(f"\033[90m共 {len(token_ids)} 个tokens\033[0m\n")


def main():
    """
    主函数：加载tokenizer并运行交互式循环
    """
    # 检查文件是否存在
    if not path.exists(VOCAB_PATH):
        print(f"错误: 找不到vocab文件: {VOCAB_PATH}")
        print("请先运行 train_bpe.py 训练tokenizer")
        return

    if not path.exists(MERGES_PATH):
        print(f"错误: 找不到merges文件: {MERGES_PATH}")
        print("请先运行 train_bpe.py 训练tokenizer")
        return

    # 加载tokenizer
    print("正在加载tokenizer...")
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens)
    print(f"Tokenizer加载成功! 词汇表大小: {len(tokenizer.vocab)}")
    print("输入文本进行分词，输入 'quit' 或 'exit' 退出\n")

    # 交互循环
    while True:
        try:
            text = input("\033[36m输入> \033[0m")

            # 检查退出命令
            if text.lower() in ["quit", "exit", "q"]:
                print("再见！")
                break

            # 可视化分词
            visualize_tokens(tokenizer, text)

        except KeyboardInterrupt:
            print("\n\n已中断。再见！")
            break
        except EOFError:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\033[31m错误: {e}\033[0m")


if __name__ == "__main__":
    main()
