import dataclasses
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

from .tokenizer import FishTokenizer
from .utilities import find_multiple

import loralib as lora

from flash_attn import flash_attn_func

@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0

@dataclass
class BaseModelArgs:
    model_type: str = "base"

    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = False

    # Codebook configs
    codebook_size: int = 160
    num_codebooks: int = 4

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

    # Initialize the model
    initializer_range: float = 0.02

    # Dummy vars
    is_reward_model: bool = False
    scale_codebook_embeddings: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_head

    @staticmethod
    def from_pretrained(path: str):
        path = Path(path)

        if path.is_dir():
            path = path / "config.json"

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return DualARModelArgs(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, ensure_ascii=False)


@dataclass
class DualARModelArgs(BaseModelArgs):
    model_type: str = "dual_ar"
    n_fast_layer: int = 4
    fast_dim: int | None = None
    fast_n_head: int | None = None
    fast_n_local_heads: int | None = None
    fast_head_dim: int | None = None
    fast_intermediate_size: int | None = None
    fast_attention_qkv_bias: bool | None = None
    fast_attention_qk_norm: bool | None = None
    fast_attention_o_bias: bool | None = None

    def __post_init__(self):
        super().__post_init__()

        self.fast_dim = self.fast_dim or self.dim
        self.fast_n_head = self.fast_n_head or self.n_head
        self.fast_n_local_heads = self.fast_n_local_heads or self.n_local_heads
        self.fast_head_dim = self.fast_head_dim or self.head_dim
        self.fast_intermediate_size = (
            self.fast_intermediate_size or self.intermediate_size
        )
        self.fast_attention_qkv_bias = (
            self.fast_attention_qkv_bias
            if self.fast_attention_qkv_bias is not None
            else self.attention_qkv_bias
        )
        self.fast_attention_qk_norm = (
            self.fast_attention_qk_norm
            if self.fast_attention_qk_norm is not None
            else self.attention_qk_norm
        )
        self.fast_attention_o_bias = (
            self.fast_attention_o_bias
            if self.fast_attention_o_bias is not None
            else self.attention_o_bias
        )


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        self.cached_len = 0
        cache_shape = (max_batch_size, max_seq_len, n_heads, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, k_val, v_val):
        seq_len = k_val.shape[1]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, self.cached_len: self.cached_len + seq_len] = k_val
        v_out[:, self.cached_len: self.cached_len + seq_len] = v_val
        self.cached_len += seq_len

        # return k_out, v_out
        return k_out[:, :self.cached_len, :, :], v_out[:, :self.cached_len, :, :]
    
    def clear(self):
        self.cached_len = 0

@dataclass
class TransformerForwardResult:
    token_logits: Tensor
    acoustic_token_logits: Tensor


class DualARTransformer(nn.Module):
    def __init__(self, config: DualARModelArgs, tokenizer: FishTokenizer) -> None:
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.is_apply_lora = False
        # For kv cache
        self.max_batch_size = -1
        self.max_seq_len = -1

        '''Slow transformer Configure'''
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks,
            config.dim,
        )
        self.layers = nn.ModuleList(
            TransformerBlock(config, use_sdpa=True) for _ in range(config.n_layer)
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if self.config.tie_word_embeddings is False:
            self.output = nn.Linear(
                config.dim,
                config.vocab_size,
                bias=False,
            )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_seq_len,
                config.head_dim,
                config.rope_base,
            ),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )


        '''Fast transformer Configure'''
        # Project to fast dim if needed
        if config.fast_dim is not None and config.fast_dim != config.dim:
            self.fast_project_in = nn.Linear(config.dim, config.fast_dim)
        else:
            self.fast_project_in = nn.Identity()

        # Fast transformer
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)

        # The equivalent bs is so large that sdpa doesn't work
        override_config = dataclasses.replace(
            config,
            dim=config.fast_dim,
            n_head=config.fast_n_head,
            n_local_heads=config.fast_n_local_heads,
            head_dim=config.fast_head_dim,
            intermediate_size=config.fast_intermediate_size,
            attention_qkv_bias=config.fast_attention_qkv_bias,
            attention_qk_norm=config.fast_attention_qk_norm,
            attention_o_bias=config.fast_attention_o_bias,
        )

        self.fast_layers = nn.ModuleList(
            TransformerBlock(override_config, use_sdpa=False)
            for _ in range(config.n_fast_layer)
        )
        self.fast_norm = RMSNorm(config.fast_dim, eps=config.norm_eps)
        self.fast_output = nn.Linear(
            config.fast_dim,
            config.codebook_size,
            bias=False,
        )

        self.register_buffer(
            "fast_freqs_cis",
            precompute_freqs_cis(
                config.num_codebooks,
                config.fast_head_dim,
                config.rope_base,
            ),
            persistent=False,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def setup_caches(
        self, max_batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.bfloat16
    ):
        max_seq_len = find_multiple(max_seq_len, 8)
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_len,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype=dtype,
            )

        # Fast transformer
        # The max seq len here is the number of codebooks
        for b in self.fast_layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                self.config.num_codebooks,
                self.config.fast_n_local_heads,
                self.config.fast_head_dim,
                dtype=dtype,
            )

    def embed(
        self, 
        # [b, t]
        tokens: Tensor,
        # [t', 10]
        vq: Optional[Tensor] = None,
        # [b, t]
        vq_mask_tokens: Optional[Tensor] = None,
    ) -> Tensor:
        # [b=1, T, dim]
        inp = self.embeddings(tokens)
        if vq is None:
            return inp

        '''rvq 反量化操作'''
        embeds = []
        for i in range(self.config.num_codebooks):
            # [T', dim]
            emb = self.codebook_embeddings(         
                vq[:, i] + i * self.config.codebook_size
            )
            embeds.append(emb)
        # [T', dim]
        vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)   # codebook变量相加

        
        '''合并 token embedding 和 codebook embedding'''
        _meger = inp[vq_mask_tokens] + vq_embeds_sum
        # 对codebook变量进行缩放
        if self.config.scale_codebook_embeddings:
            _meger = _meger / math.sqrt(self.config.num_codebooks + 1)
        inp[vq_mask_tokens] = _meger

        return inp

    def forward_slow(
        self,
        # [b, t]
        tokens: Tensor,
        # [t', 10]
        vq: Optional[Tensor] = None,
        # [b, t]
        vq_mask_tokens: Optional[Tensor] = None
    ):
        seq_len = tokens.size(1)

        x = self.embed(
            tokens=tokens,
            vq_mask_tokens=vq_mask_tokens,
            vq=vq
        )

        freqs_cis = self.freqs_cis[:seq_len]

        for layer in self.layers:
            x = layer(x, freqs_cis)

        # We got slow_out here
        slow_out = self.norm(x)

        if self.config.tie_word_embeddings:
            token_logits = F.linear(slow_out, self.embeddings.weight)
        else:
            token_logits = self.output(slow_out)

        return (token_logits, x)

    def forward_fast(
        self,
        # [b, t, dim]
        x: Tensor,
        # [t', 10]
        vq: Optional[Tensor] = None,
        # [b, t]
        vq_mask_tokens: Optional[Tensor] = None):

        batch_size = x.size(0)

        fast_seq_len = self.config.num_codebooks
        # rope 位置编码的 sin cos
        fast_freqs_cis = self.fast_freqs_cis[:fast_seq_len]

        # 获取 vq 位置的输出
        # [t', dim]
        _temp = torch.zeros(size=(batch_size, 1), dtype=torch.bool, device=vq_mask_tokens.device.type)
        _mask = torch.cat([vq_mask_tokens[:, 1:], _temp], dim=1)
        x = x[_mask]

        # [t', 9]
        acoustic_tokens = vq[..., :-1]
        # 将训练数据的 codebooks 转为 embeddings
        # [t', 9, dim]
        acoustic_tokens_embeddings = self.fast_embeddings(acoustic_tokens)

        # 拼接
        # [t', 10, dim]
        x = torch.cat([x[:, None], acoustic_tokens_embeddings], dim=1)

        # 通过 decoder 层
        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis)

        # 线性映射
        fast_out = self.fast_norm(x)
        # [t', 10, cb_size]
        acoustic_token_logits = self.fast_output(fast_out)

        assert acoustic_token_logits.shape[1] == self.config.num_codebooks

        return acoustic_token_logits

    def forward(
        self,
        # [b, t]
        tokens: Tensor,
        # [t', 10]
        vq: Optional[Tensor] = None,
        # [b, t]
        vq_mask_tokens: Optional[Tensor] = None
    ) -> TransformerForwardResult:
        token_logits, hidden_states = self.forward_slow(tokens, vq, vq_mask_tokens)
        acoustic_token_logits = self.forward_fast(hidden_states, vq, vq_mask_tokens)

        return TransformerForwardResult(
            token_logits=token_logits,
            acoustic_token_logits=acoustic_token_logits,
        )


    def forward_generate_slow(
        self,
        # [b=1, T]
        tokens: Tensor,
        # [b=1, T]
        vq_mask_tokens: Tensor,
        # [T', 10]
        vq: Optional[Tensor],
        # [T]
        input_pos: Optional[Tensor]
    ):
        
        x = self.embed(
            tokens=tokens,
            vq_mask_tokens=vq_mask_tokens,
            vq=vq
        )

        # 获取 Rope freqs_cis 矩阵
        freqs_cis = self.freqs_cis[input_pos]

        '''Decoder * n'''
        for layer in self.layers:
            x = layer(x, freqs_cis)

        # 取最后一个输出
        # [b=1, t=1, dim]
        x = x[:, -1:]

        # We got slow_out here
        slow_out = self.norm(x)
        token_logits = self.output(slow_out)

        # token_logits: [b=1, t=1, vec_size]
        # x: [b=1, t=1, dim]
        return token_logits, x

    def forward_generate_fast(      # fast transformer
        self,
        # [b=1, t=1, dim]
        x: Tensor,
        # [1] 
        input_pos: Optional[Tensor]
    ) -> Tensor:
        # rope sin cos 矩阵
        fast_freqs_cis = self.fast_freqs_cis[input_pos]

        for layer in self.fast_layers:
            x = layer(x, fast_freqs_cis)

        # unflatten the batch and num_codebooks
        fast_out = self.fast_norm(x)  # only take the last token
        codebook_logits = self.fast_output(fast_out)

        # [b=1, t=1, cb_size]
        return codebook_logits

    def setup_lora(self, lora_config: LoraConfig):
        self.is_apply_lora = True

        # Replace the embedding layer with a LoRA layer
        self.embeddings = lora.Embedding(
            num_embeddings=self.embeddings.num_embeddings,
            embedding_dim=self.embeddings.embedding_dim,
            padding_idx=self.embeddings.padding_idx,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
        )
        self.codebook_embeddings = lora.Embedding(
            num_embeddings=self.codebook_embeddings.num_embeddings,
            embedding_dim=self.codebook_embeddings.embedding_dim,
            padding_idx=self.codebook_embeddings.padding_idx,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
        )
        self.fast_embeddings = lora.Embedding(
            num_embeddings=self.fast_embeddings.num_embeddings,
            embedding_dim=self.fast_embeddings.embedding_dim,
            padding_idx=self.fast_embeddings.padding_idx,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
        )

        # Replace output layer with a LoRA layer
        linears = [(self, "output")]

        # Slow
        for layer in self.layers:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            linears.extend(
                [
                    (layer.feed_forward, "w1"),
                    (layer.feed_forward, "w2"),
                    (layer.feed_forward, "w3"),
                ]
            )
        
        # Fast
        linears.append((self, "fast_output"))
        for layer in self.fast_layers:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            linears.extend(
                [
                    (layer.feed_forward, "w1"),
                    (layer.feed_forward, "w2"),
                    (layer.feed_forward, "w3"),
                ]
            )

        for module, layer in linears:
            updated_linear = lora.Linear(
                in_features=getattr(module, layer).in_features,
                out_features=getattr(module, layer).out_features,
                bias=getattr(module, layer).bias,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
            )
            setattr(module, layer, updated_linear)

        # Mark only the LoRA layers as trainable
        lora.mark_only_lora_as_trainable(self, bias="none")

    def save_pretrained(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(path / "config.json")
        torch.save(self.state_dict(), path / "model.pth")
        self.tokenizer.save_pretrained(path)

        if self.is_apply_lora:
            torch.save(lora.lora_state_dict(self), path / "lora.pth")

    @staticmethod
    def from_pretrained(
        path: str,
        lora_config: LoraConfig | None = None,
    ) -> "BaseTransformer":
        config = BaseModelArgs.from_pretrained(str(path))
            
        tokenizer = FishTokenizer.from_pretrained(path)

        model = DualARTransformer(config, tokenizer=tokenizer)

        if lora_config is not None:
            model.setup_lora(lora_config)
        
        model_path = Path(path) / "model.pth"
        model.load_state_dict(torch.load(model_path), strict=False)

        lora_path = Path(path) / "lora.pth"
        if lora_path.exists():
            model.load_state_dict(torch.load(lora_path), strict=False)
        
        return model


class TransformerBlock(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True) -> None:
        super().__init__()
        self.attention = Attention(config, use_sdpa=use_sdpa)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(
        self, x: Tensor, freqs_cis: Tensor
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: BaseModelArgs, use_sdpa: bool = True):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(
            config.dim, total_head_dim, bias=config.attention_qkv_bias
        )
        self.wo = nn.Linear(
            config.n_head * config.head_dim, config.dim, bias=config.attention_o_bias
        )
        self.kv_cache: Optional[KVCache] = None

        if config.attention_qk_norm:
            self.q_norm = nn.RMSNorm(config.head_dim, config.norm_eps)
            self.k_norm = nn.RMSNorm(config.head_dim, config.norm_eps)

        self.dropout = config.dropout
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.use_sdpa = use_sdpa
        self.attention_qk_norm = config.attention_qk_norm
        self.config = config

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.attention_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # att_mask 是必须的
        causal = True
        if self.kv_cache is not None:
            # 如果使用缓存，则只有再第一次时才应用 att_mask
            causal = self.kv_cache.cached_len == 0
            k, v = self.kv_cache.update(k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=2)

        y = flash_attn_func(q, k, v, causal=causal)

        y = y.view(bsz, seqlen, q_size)

        return self.wo(y)


class FeedForward(nn.Module):
    def __init__(self, config: BaseModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    """
    Precomputes frequency tensors for complex exponentials (cis)

    Args:
        seq_len: Length of the sequence for which positional embeddings are needed.
        n_elem: Number of elements in the frequency tensor.
        base: Base value for the frequency scaling (default: 10000).

    Returns:
        A tensor containing the precomputed frequencies in real and imaginary parts (bfloat16).
    """
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
