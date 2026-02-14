import torch
from einx import rearrange

from cs336_basics.model import TransformerLM


def _softmax_with_temperature(x: torch.Tensor, dim: int, temperature: float = 0.0):
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_val
    exp_x = torch.exp(x_stable / temperature)
    exp_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / exp_sum


def decode(
    model: TransformerLM,
    prompt: torch.Tensor,
    endoftext_token: int,
    max_tokens: int = 32,
    temperature: float = 0.7,
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
    prompt = rearrange("... -> 1 ...", prompt)  # 增加一个 batch 维度兼容 forward

    while generated_tokens_count < max_tokens:
        logits = model(prompt)[0][-1]  # 取第一个 batch，然后取序列的最后一个位置，即最新的 token

        # 带温度采样的 softmax
        # 如果温度为 0，则直接选择 logits 中最大的值
        if temperature == 0.0:
            choiced_token = torch.argmax(logits)
        else:
            softmax = _softmax_with_temperature(logits, dim=-1, temperature=temperature)
            # 按概率降序排序
            sorted_probs, sorted_indices = torch.sort(softmax, descending=True)

            # 执行 top_k
            top_k_probs = sorted_probs[:top_k]
            top_k_indices = sorted_indices[:top_k]

            # 执行 top_p
            cumsum_probs = torch.cumsum(top_k_probs, dim=-1)
            # 找到累积概率首次超过 top_p 的位置
            cutoff_mask = cumsum_probs > top_p
            if cutoff_mask.any():
                cutoff_index = cutoff_mask.nonzero()[0].item() + 1  # 包含刚好超过的那一项
            else:
                cutoff_index = len(top_k_probs)

            top_p_probs = top_k_probs[:cutoff_index]
            top_p_indices = top_k_indices[:cutoff_index]

            # 重新归一化
            top_p_probs = top_p_probs / top_p_probs.sum()

            # 从剩下的序列中随机采样
            sampled_index = torch.multinomial(top_p_probs, num_samples=1)
            choiced_token = top_p_indices[sampled_index]

        # 将生成的 token 添加到 prompt 中
        choiced_token = rearrange("vocab -> 1 vocab", choiced_token)  # 增加维度以匹配 batch 和 seq 维度
        prompt = torch.cat([prompt, choiced_token], dim=-1)

        # 检查是否遇到了 endoftext_token
        if choiced_token.item() == endoftext_token:
            break

        generated_tokens_count += 1

    # 返回生成的序列（去掉 batch 维度）
    return prompt.squeeze(0)
