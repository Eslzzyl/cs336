from typing import Iterable
import torch


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6
):
    # 过滤掉没有梯度的参数
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # 计算所有梯度的总L2范数
    # 为了避免创建巨大的拼接张量，我们可以计算每个梯度范数的平方和，然后开方
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2
    )
    # 根据总范数进行缩放
    if total_norm > max_l2_norm:
        # 计算缩放因子
        clip_coef = max_l2_norm / (total_norm + eps)

        # 用缩放因子乘以每个梯度
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)  # 使用 in-place 乘法提高效率
