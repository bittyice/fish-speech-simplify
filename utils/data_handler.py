from dataclasses import dataclass
from html import unescape
import re
from typing import Callable, TypeVar, Generic
import unicodedata
import librosa
import numpy as np
import torch

@dataclass
class AudioData:
    audio: np.ndarray
    sr: int


T = TypeVar('T')
class Middleware(Generic[T]):
    def __call__(self, data: T, context) -> T:
        raise NotImplementedError()


class Handler(Generic[T]):
    def __init__(self):
        self.mids: list[Middleware[T]] = []

    def add_middleware(self, mid: Middleware[T] | Callable[[T, dict], T]):
        self.mids.append(mid)
    
    def handle(self, data: T, context) -> T:
        for mid in self.mids:
            data = mid(data, context)
            if data is None:
                break
        return data


class ComminTextMiddleware(Middleware[str]):
    def __call__(self, text: str, context: dict):
        """清洗单条中文文本，用于 LLM 训练预处理"""
        if not text or not isinstance(text, str):
            return None

        # 1. Unicode 标准化（解决全角半角、兼容形等问题）
        text = unicodedata.normalize("NFKC", text)

        # 2. 去除 HTML 标签
        text = re.sub(r"<[^>]+>", "", text)

        # 3. 去除 Markdown 语法
        text = re.sub(r"(`{1,3}.*?`{1,3})", "", text)  # 行内/多行代码块
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # 图片链接
        text = re.sub(r"\[[^\]]*\]\([^)]+\)", "", text)  # 普通链接
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # 标题

        # 4. 去除控制字符（包括零宽字符、不可见字符等）
        text = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]", "", text)
        text = re.sub(r"[\x00-\x1F\x7F]", "", text)

        # 5. 替换异常空白字符为普通空格
        text = re.sub(r"\s+", " ", text)

        # 6. 中文标点标准化（可选）
        # 例如将英文标点换成中文标点，保证训练时统一
        text = text.replace(",", "，").replace("?", "？").replace("!", "！").replace(";", "；").replace(":", "：")

        # 7. 删除连续重复的标点（避免“！！！！”这种）
        text = re.sub(r"([，。！？\-—_])\1+", r"\1", text)

        # 8. 删除过短或无意义内容（例如单个标点、单个字）
        if len(text.strip()) < 2:
            return None

        return text.strip()


class CommonAudioMiddleware(Middleware[AudioData]):
    def __call__(self, data: AudioData, context):
        audio = data.audio
        sr = data.sr

        # 音量归一化
        audio = audio / np.max(np.abs(audio))
        # 静音裁剪
        audio, _ = librosa.effects.trim(audio, top_db=20)

        return AudioData(
            audio=audio,
            sr=sr
        )


class TextScoreMiddleware(Middleware[str]):
    '''
    该中间件对文本进行评分，然后将评分结果写到 context['score'] 中.
    model: Transformer 文本自回归模型.
        示例: model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
    tokenizer: Transformer 文本自回归 tokenizer.
        示例: tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
    '''
    def __init__(self, model: torch.nn.Module, tokenizer, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_key = 'score'

    def __call__(self, text: str, context):
        input = self.tokenizer([text], return_tensors="pt")
        labels = input['input_ids']
        labels = torch.cat([labels[:, 1:], torch.tensor([[-100]], dtype=torch.long)], dim=-1)
        input['labels'] = labels
        input = { key: val.to(device=self.device) for key, val in input.items() }
        
        with torch.no_grad():
            output = self.model(**input)

        context[self.context_key] = output['loss'].item()

        return text

