import functools
import os
from typing import Optional
import torch
import torchaudio
from tqdm import tqdm

from openaudio.dac import DAC

from .dual_ar import DualARTransformer, LoraConfig
from .tokenizer import IM_END_TOKEN, FishTokenizer, TextAndVQ
from .utilities import sample


def decode_one_token_ar(
    model: DualARTransformer,
    # [b=1, t]
    tokens,
    # [b=1, t]
    vq_mask_tokens,
    # [T', 10]
    vq,
    input_pos_slow: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: Optional[torch.Tensor] = None,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 执行 slow transformer
    # logits: [b=1, t=1, vec_size]
    # hidden_states: [b=1, t=1, dim]
    logits, hidden_states = model.forward_generate_slow(
        tokens,
        vq_mask_tokens,
        vq,
        input_pos_slow
    )

    # 对 logits 进行采样得到 codebook1 的 token
    # [b=1, t=1]
    pre_token = sample(     # 根据 temperature 进行采样
            logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens if previous_tokens is not None else None
            ),
        )[0]
    # list [b=1, 1]
    pre_codebooks = []
    # [b=1, 1]
    acoustic_token = pre_token - model.tokenizer.semantic_begin_id
    acoustic_token[acoustic_token < 0] = 0
    pre_codebooks.append(acoustic_token)

    # 清理 fast transformer 的 kv 缓存
    for layer in model.fast_layers:
        layer.attention.kv_cache.clear()

    # 指示 fast token 位置
    input_pos_fast = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos_fast)


    # 获得 codebook 变量
    # [b=1, 1, dim]
    hidden_states = model.fast_embeddings(acoustic_token)
    # 执行循环 1->9 循环，获得 cb2->10 token
    for codebook_idx in range(1, model.config.num_codebooks):
        # 指示 fast token 位置 
        input_pos_fast = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        # 通过前一个 codebook embedding 预测下一个 codebook token
        # [b, 1, cb_size]
        logits = model.forward_generate_fast(hidden_states, input_pos_fast)

        # 对除 cb1 外的token，只取 1024 范围内
        short_logits = logits[:, :, :1024]

        # 采样，得出 acoustic token
        # [b, 1]
        acoustic_token = sample(
            short_logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )[0]
        pre_codebooks.append(acoustic_token)

        # 获取 codebook embedding
        hidden_states = model.fast_embeddings(acoustic_token)

    # [b=1, 10]
    pre_codebooks = torch.cat(pre_codebooks, dim=1)
    # pre_token: [b=1, 1]
    # pre_codebooks: [b=1, 10]
    return pre_token, pre_codebooks

def encode(model: DAC, audio, sr):
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device.type

    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)

    audio = torchaudio.functional.resample(audio, sr, model.sample_rate)
    audios = audio[None].to(device, dtype=dtype)

    # VQ Encoder
    audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
    indices, indices_lens = model.encode(audios, audio_lengths)
    if indices.ndim == 3:
        indices = indices[0]

    return indices

def generate_acoustic_token(model, text: str, prompt_text = None, prompt_accoustic_token = None):
    device = next(model.parameters()).device.type
    temperature = 0.2
    top_p = 0.5
    repetition_penalty = 1.2

    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=torch.bfloat16,
        )

    # 编码
    tokenizer: FishTokenizer = model.tokenizer
    vqs = list[TextAndVQ]()
    if prompt_text is not None:
        vqs.append(TextAndVQ(speaker=0, text=prompt_text, vq=prompt_accoustic_token))
    vqs.append(TextAndVQ(speaker=0, text=text, vq=None))
    _result = tokenizer.encodebylist(vqs)

    # tokens 融合了 提示语音文本、提示语音、要转录的文本
    # [T]
    tokens = _result.tokens.to(device=device)
    # [T]
    vq_mask_tokens = _result.vq_mask_tokens.to(device=device)
    # [T', 10]
    vq = _result.vq.T.to(device=device) if _result.vq is not None else None
    # token 长度
    tokens_length = tokens.size(0)
    # tokens 位置
    input_pos = torch.arange(0, tokens_length, device=device)
    del _result

    # 增加一个维度
    # [b=1, T]
    tokens = tokens[None]
    # [b=1, T]
    vq_mask_tokens = vq_mask_tokens[None]
    # 采样参数
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    # 重复惩罚
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )
    # 当前可生成的最大 token 数量
    max_new_tokens = model.config.max_seq_len - tokens_length
    
    # 先生成一个 token
    # pre_token: [b=1, t=1]
    # pre_acoustic_token: [b=1, 10]
    pre_token, pre_acoustic_token = decode_one_token_ar(   # 生成一个 token
        model=model,
        tokens=tokens,
        vq_mask_tokens=vq_mask_tokens,
        vq=vq,
        input_pos_slow=input_pos,
        temperature=temperature,
        top_p=top_p
    )
    # [b=1, t]
    pre_token_list = pre_token.clone()
    # [t, 10]
    pre_codebook_token_list = pre_acoustic_token.clone()
    input_pos = torch.tensor([tokens_length], device=device, dtype=torch.long)
    vq_mask_tokens = torch.tensor([True], device=device, dtype=torch.bool)
    
    # 生成后续 token
    for i in tqdm(range(max_new_tokens)):
        # 用于判断是否存在重复的窗口大小
        # 存在重复时会应用重复惩罚
        win_size = 16
        if i < win_size:
            window = pre_token_list[:, :win_size]
        else:
            window = pre_token_list[:, i - win_size : i]
        
        # pre_token: [b=1, t=1]
        # pre_codebook_token: [b=1, 10]
        pre_token, pre_acoustic_token = decode_one_token_ar(
                model=model,
                tokens=pre_token,
                vq_mask_tokens=vq_mask_tokens,
                vq=pre_acoustic_token,
                input_pos_slow=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
        
        pre_token_list = torch.cat([pre_token_list, pre_token], dim=1)
        pre_codebook_token_list = torch.cat([pre_codebook_token_list, pre_acoustic_token], dim=0)
        if pre_token[0, 0] == model.tokenizer.im_end_id:       # 停止token
            break
        
        input_pos += 1
    
    # [10, t]
    return pre_codebook_token_list.T
    
def decode(model: DAC, indices):
    device = next(model.parameters()).device.type
    indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)

    fake_audios = model.decode(indices)

    return fake_audios, model.sample_rate

