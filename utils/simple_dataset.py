from typing import Callable
import torch
from typing import TypeVar

T = TypeVar('T')

class SimpleTextDataset(torch.utils.data.Dataset):
    '''
    数据集
    '''
    def __init__(self, datas: list[T], tokenizer: Callable[[T], dict[str, list]]):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index: int):
        data = self.datas[index]
        return self.tokenizer(data)


class SimpleTextDataCollator:
    '''
    批量采集类
    给类用于 Dataloader，对 SimpleTextDataset 获取的批量数据按照给的的 padding_val 进行填充
    '''
    def __init__(self, padding_val = 0):
        super().__init__()
        self.padding_val = padding_val

    def __call__(self, features: list[dict[str, list]]):
        keys = features[0].keys()
        result: dict[str, list[torch.Tensor]] = {key: [] for key in keys}

        for ft in features:
           for key, val in ft.items():
               result[key].append(torch.tensor(val))

        for key, val in result.items():
            result[key] = torch.nn.utils.rnn.pad_sequence(result[key], batch_first=True, padding_value=self.padding_val)
        
        return result
