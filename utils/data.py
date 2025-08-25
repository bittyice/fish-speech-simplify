from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class Content:
    type: Literal['text', 'audio', 'other']
    text: Optional[str] = None
    audio: Optional[str] = None
    other: any = None

@dataclass
class Message:
    role: Literal['user', 'assistant']
    content: Content

@dataclass
class DialogData:
    messages: list[Message] = field(default_factory=list)
    context: dict = field(default_factory=dict)
