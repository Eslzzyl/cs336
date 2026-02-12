import torch
from einx import rearrange

from cs336_basics.model import TransformerLM


def softmax_with_temperature(x: torch.Tensor, dim: int, temperature: float = 0.0):
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_val
    exp_x = torch.exp(x_stable / temperature)
    exp_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / exp_sum


def decode(
    model: TransformerLM,
    prompt: torch.Tensor,
    endoftext_token: int,
    max_tokens: int,
    temperature: float = 0.0,
    top_k: int = 20,
    top_p: float = 0.6,
) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature should be >= 0")
    if top_k < 1:
        raise ValueError("top_k should be >= 1")
    if top_p <= 0:
        raise ValueError("top_p should be > 0")

    generated_tokens_count = 0
    prompt = rearrange("... -> 1 ...")  # 增加一个 batch 维度兼容 forward
    while generated_tokens_count < max_tokens:
        logits = model(prompt)[0][-1]  # 取第一个 batch，然后取序列的最后一个位置，即最新的 token
        # 带温度采样的 softmax
        # 如果温度为 0，则直接选择 logits 中最大的值
        if temperature == 0.0:
            choiced_token = torch.argmax(logits)
        else:
            max_val = torch.max(logits).values
            logits_stable = logits - max_val
            exp_logits = torch.exp(logits_stable / temperature)
            exp_sum = torch.sum(exp_logits)
            softmax = exp_logits / exp_sum
            # 按概率降序排序
            softmax_sorted = torch.sort(softmax, descending=True, stable=True)
            # 执行 top_k
            softmax_sorted = softmax_sorted[:top_k]
            # 执行 top_p
            index = len(softmax_sorted) - 1
            while softmax_sorted[index] < top_p:
                index -= 1
            softmax_sorted = softmax_sorted[:index]
            # 从剩下的序列中随机采样
            choiced_token = torch.multinomial(softmax_sorted, num_samples=1)
