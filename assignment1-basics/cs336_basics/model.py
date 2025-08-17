import torch
import torch.nn as nn
import einx
import math


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        sigma = math.sqrt(2 / (in_features + out_features))
        w = torch.normal(mean=0, std=sigma, size=(out_features, in_features), device=device, dtype=dtype)
        w = torch.nn.init.trunc_normal_(w, a=-3 * sigma, b=3 * sigma)
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("d_out d_in, ... d_in -> ... d_out", self.weight, x)


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., d_model
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        e = torch.normal(mean=0, std=1, size=(num_embeddings, embedding_dim), device=device, dtype=dtype)
        e = torch.nn.init.trunc_normal_(e, a=-3, b=3)
        self.weight = nn.Parameter(e)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Args:
            d_model (int): Hidden dimension of the model
            eps (float = 1e-5) Epsilon value for numerical stability
            device (torch.device | None = None) Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        self.epsilon = eps
        g = torch.ones(size=(d_model,), device=device, dtype=dtype)
        self.weight = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # upcast input to torch.float32 to prevent overflow when square the input
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # RMSNorm implementation
        square_sum = einx.reduce("... [d_model]", x * x, op=torch.sum)
        rms = torch.sqrt(1 / x.shape[-1] * square_sum + self.epsilon)
        rms = einx.rearrange("... -> ... 1", rms)
        result = x / rms * self.weight

        # Return the result in the original dtype
        return result.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    """
    Position-wise feed forward network with SwiGLU activation
    """

    def _get_d_ff(self, d_model: int) -> int:
        # d_ff 应当大致等于 f_model 的 8/3。因此，我们首先将 d_model 乘以 8/3，然后将其四舍五入到最近的 64 的整数倍，从而充分利用硬件。
        return int(round(d_model * (8 / 3) / 64) * 64)

    def __init__(
        self, d_model: int, d_ff: int = None, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Args:
            d_model (int): Hidden dimension of the model
            device (torch.device | None = None) Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = self._get_d_ff(d_model) if d_ff is None else d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 此处不再使用 einx 了，因为需要访问 Linear 模块内部的权重，这样不太好。
        w1x = self.w1(x)
        w1x_silu = silu(w1x)
        w3x = self.w3(x)
        return self.w2(w1x_silu * w3x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Args:
            theta (float): theta value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None = None): Device to store the buffer on
        """
        super().__init__()
        # RoPE 本身不包含任何可学习的参数，__init__ 主要用来计算所有可能的位置 i 和所有可能的维度对 k 的 cos 和 sin 值，并存储起来，从而减少重复计算。
        self.theta = theta
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k))  # shape: (d_k // 2,)
        position_indices = torch.arange(max_seq_len, dtype=torch.float)  # shape: (max_seq_len,)
        self.position_frequencies = einx.dot(
            "inv_freq_len, max_seq_len -> max_seq_len inv_freq_len", inv_freq, position_indices
        )
        cos_cached = torch.cos(self.position_frequencies)
        sin_cached = torch.sin(self.position_frequencies)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (..., seq_len, d_k)
            token_positions (..., seq_len)
        """
        current_cos = self.cos_cached[token_positions]  # shape: (..., seq_len, d_k // 2)
        current_sin = self.sin_cached[token_positions]  # 同上

        x_even = x[..., ::2]  # shape: (..., seq_len, d_k // 2)
        x_odd = x[..., 1::2]  # 同上

        rotated_x_even = x_even * current_cos - x_odd * current_sin
        rotated_x_odd = x_even * current_sin + x_odd * current_cos

        return torch.stack([rotated_x_even, rotated_x_odd], dim=-1).reshape(x.shape)


def softmax(x: torch.Tensor, dim: int):
    """
    对输入张量 x 的第 i 个维度进行 softmax，通过减去目标维度中最大的值来增加数值稳定性。
    Args:

    """
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_val
    exp_x = torch.exp(x_stable)
    exp_sum = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / exp_sum


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
    """
        Args:
            q (batch_size, ..., seq_len, d_k)
            k (batch_size, ..., seq_len, d_k)
            v (batch_size, ..., seq_len, d_v)
            mask (seq_len, seq_len)
        Returns a tensor with shape (batch_size, ..., d_v)
        The attention probabilities of positions with a mask value of True should collectively sum
    to 1, and the attention probabilities of positions with a mask value of False should be zero.
    """
    qk = einx.dot("... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k", q, k)
    d_k = q.shape[-1]
    pre_softmax = qk / math.sqrt(d_k)
    if mask is not None:
        pre_softmax.masked_fill_(mask=~mask, value=-torch.inf)
    post_softmax = softmax(pre_softmax, dim=-1)
    return einx.dot("... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v", post_softmax, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_max_seq_len: int = None,
        rope_theta: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model (int) Dimensionality of the Transformer block inputs.
            num_heads (int) Number of heads to use in multi-head self-attention.
            rope_max_seq_len (int = None) max_seq_len for RoPE. If None, will not use RoPE
            rope_theta (int = None) theta for RoPE. If None, will not use RoPE
            device (torch.device | None = None) Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # 使用整数除法来确保整除
        d_k = d_model // num_heads

        self.num_heads = num_heads
        self.d_model = d_model

        # 线性投影层
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 最终的输出层
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if rope_max_seq_len is not None or rope_theta is not None:
            assert rope_max_seq_len is not None and rope_theta is not None, (
                "rope_max_seq_len and rope_theta must be not None to enable RoPE"
            )
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta, d_k=d_k, max_seq_len=rope_max_seq_len, device=device
            )
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        """
        Args:
            x should be (batch_size, seq_len, d_model) in shape
            token_positions should be (..., seq_len) in shape
        """
        _, seq_len, _ = x.shape

        q_proj = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k_proj = self.k_proj(x)  # 同上
        v_proj = self.v_proj(x)  # 同上

        q = einx.rearrange(
            "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim",
            q_proj,
            num_heads=self.num_heads,
        )
        k = einx.rearrange(
            "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim",
            k_proj,
            num_heads=self.num_heads,
        )
        v = einx.rearrange(
            "batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim",
            v_proj,
            num_heads=self.num_heads,
        )

        # apply rope
        if self.rope:
            assert token_positions is not None, "token_positions is necessary when RoPE is enabled"
            q = self.rope(q, token_positions)  # (batch_size, num_heads, seq_len, head_dim)
            k = self.rope(k, token_positions)

        # scaled_dot_product_attention 对 mask 的要求：True 表示允许注意（不填充），False 表示需要掩盖（填充 -inf）。
        # 对因果掩码而言，这意味着下三角（包括对角线）应该是 True，上三角应该是 False。
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool).tril(diagonal=0)
        attn_output_per_head = scaled_dot_product_attention(
            q, k, v, mask=causal_mask
        )  # (batch_size, num_heads, seq_len, head_dim)

        attn_output_combined = einx.rearrange(
            "batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)",
            attn_output_per_head,
            num_heads=self.num_heads,
        )

        return self.output_proj(attn_output_combined)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_max_seq_len: int,
        rope_theta: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model (int) Dimensionality of the Transformer block inputs
            num_heads (int) Number of heads to use in multi-head self-attention
            d_ff (int) Dimensionality of the position-wise feed-forward inner layer
            rope_max_seq_len (int) max_seq_len for RoPE
            rope_theta (int) theta for RoPE
            device (torch.device | None = None) Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_max_seq_len=rope_max_seq_len,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
        # layer norm for sub-layer 1
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # layer norm for sub-layer 2
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        # 构造 token_positions：[0, 1, 2, ..., seq_len-1]
        _, seq_len, _ = x.shape  # 输入形状是 (batch_size, seq_len, d_model)
        token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)

        # 执行包含两个子层的前向传播过程
        sub_layer_1 = x + self.attn(self.ln1(x), token_positions)
        sub_layer_2 = sub_layer_1 + self.ffn(self.ln2(sub_layer_1))
        return sub_layer_2


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        rope_theta: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            d_model (int) Dimensionality of the Transformer block inputs
            num_heads (int) Number of heads to use in multi-head self-attention
            d_ff (int) Dimensionality of the position-wise feed-forward inner layer
            vocab_size (int) The size of the vocabulary
            context_length (int) The maximum context length
            num_layers (int) The number of Transformer blocks to use
            rope_theta (int) theta for RoPE
            device (torch.device | None = None) Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        """
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        # layer norm final
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        # output embedding (a linear layer)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        token_embeddings = self.token_embeddings(x)  # (batch_size, seq_len, d_model)

        hidden_states = token_embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)  # (batch_size, seq_len, d_model)

        post_norm = self.ln_final(hidden_states)  # (batch_size, seq_len, d_model)
        logits = self.lm_head(post_norm)  # (batch_size, seq_len, vocab_size)
        return logits
