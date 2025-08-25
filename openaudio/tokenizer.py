import base64
from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
import torch
import numpy as np

import tiktoken

logger = logging.getLogger(__name__)

# This is a modified version of the default pattern from GPT-4o, that better handles punctuations.
FISH_TIKTOKEN_PATTERN = "|".join(
    [
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
        r"\p{P}",
        r"[^\r\n\p{L}\p{N}]?\p{L}+",
        r"\p{N}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(\?!\S)",
        r"\s+",
    ]
)
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
PAD_TOKEN = "<|pad|>"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
PHONEME_START_TOKEN = "<|phoneme_start|>"
PHONEME_END_TOKEN = "<|phoneme_end|>"
TOOL_CALL_START_TOKEN = "<|tool_call_start|>"
TOOL_CALL_END_TOKEN = "<|tool_call_end|>"

MODALITY_TEXT_TOKEN = "<|text|>"
MODALITY_VOICE_TOKEN = "<|voice|>"
MODALITY_INTERLEAVE_TOKEN = "<|interleave|>"
AUDIO_START_TOKEN = "<|audio_start|>"
AUDIO_END_TOKEN = "<|audio_end|>"
AUDIO_EMBED_TOKEN = "<|audio|>"

@dataclass(kw_only=True)
class TextAndVQ:
    speaker: int
    text: str
    # shape(10, T)
    vq: np.ndarray | None

@dataclass(kw_only=True)
class EncodedMessage:
    tokens: torch.Tensor
    vq_mask_tokens: torch.Tensor
    vq: torch.Tensor | None

class FishTokenizer:
    def __init__(
        self, token2ids: dict, special_tokens_to_ids: dict, semantic_begin_id: int, semantic_end_id: int
    ) -> None:
        self.special_tokens_to_ids = special_tokens_to_ids

        # Acoustic Token 的起始索引
        self.semantic_begin_id = semantic_begin_id
        # Acoustic Token 的终止索引
        self.semantic_end_id = semantic_end_id

        self.tkt_model = tiktoken.core.Encoding(
            name='fish',
            pat_str=FISH_TIKTOKEN_PATTERN,
            mergeable_ranks=token2ids,
            special_tokens=special_tokens_to_ids,
        )

        self.im_end_id = self.tkt_model.encode(IM_END_TOKEN, allowed_special='all', disallowed_special=set())[0]
    
    def encode(self, s: str):
        return self.tkt_model.encode(s, allowed_special='all', disallowed_special=set())
    
    def encodebylist(self, vqs: list[TextAndVQ], isinference = True):
        _token = self.encode(MODALITY_INTERLEAVE_TOKEN)

        # 所有的 token
        all_tokens = [] + _token
        # 用于指示 vq 在 all token 的位置的 mask
        vq_masks_all_tokens = [False] * len(_token)
        # vq 数据
        all_vq_parts = []

        len_vqs = len(vqs)
        for i in range(len_vqs):
            text_vq = vqs[i]
            assert text_vq.speaker is not None
            assert text_vq.text is not None

            _token = self.encode(f"<|speaker:{text_vq.speaker}|>")
            all_tokens += _token
            vq_masks_all_tokens += [False] * len(_token)


            _token = self.encode(text_vq.text)
            all_tokens += _token
            vq_masks_all_tokens += [False] * len(_token)


            if text_vq.vq is not None:
                _token = (text_vq.vq[0] + self.semantic_begin_id).tolist()
                all_tokens += _token
                vq_masks_all_tokens += [True] * len(_token)
                all_vq_parts.append(text_vq.vq)
            
            # 除了 i 是最后一个且 isinference == True 不加 end 标签
            # 其余情况都要加 end 标签
            if not (i == (len_vqs - 1) and isinference == True):
                _token = self.encode(IM_END_TOKEN)
                all_tokens += _token
                vq_masks_all_tokens += [False] * len(_token)


        # [T]
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        # [T]
        tokens_vq_masks = torch.tensor(vq_masks_all_tokens, dtype=torch.bool)
        # [10, T']
        vq = torch.tensor(np.concatenate(all_vq_parts, axis=1), dtype=torch.long) if len(all_vq_parts) > 0 else None

        return EncodedMessage(
            # [T]
            tokens=tokens,
            # [T]
            vq_mask_tokens=tokens_vq_masks,
            # [10, T]
            vq=vq,
        )

    def decode(self, tokens: list[int]) -> str:
        return self.tkt_model.decode(tokens)

    def save_pretrained(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "tokenizer.tiktoken", "w") as f:
            for token, rank in self.tkt_model._mergeable_ranks.items():
                a = base64.b64encode(token).decode()
                if a == "":
                    a = "="
                f.write(f"{a} {rank}\n")

        with open(path / "special_tokens.json", "w") as f:
            json.dump(
                self.special_tokens_to_ids,
                f,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def from_pretrained(path: str):
        special_tokens_path = Path(path) / "special_tokens.json"
        tiktoken_bpe_file = Path(path) / "tokenizer.tiktoken"

        assert special_tokens_path.exists()
        assert tiktoken_bpe_file.exists()

        with open(special_tokens_path) as f:
            special_tokens_to_ids = json.load(f)

        with open(tiktoken_bpe_file) as f:
            token2ids = {}
            for line in f:
                if not line:
                    continue
                token, rank = line.split()
                if token == "=":
                    continue
                token2ids[base64.b64decode(token)] = int(rank)

        return FishTokenizer(
            token2ids, special_tokens_to_ids, semantic_begin_id=151658, semantic_end_id=155753
        )
