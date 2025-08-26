
from dataclasses import dataclass
import json
import random
import numpy as np
import torch
import torchaudio
from openaudio.dac import DAC, DACConfig
from openaudio.inference import encode
from openaudio.dual_ar import DualARTransformer, LoraConfig
from openaudio.tokenizer import IM_END_TOKEN, FishTokenizer, TextAndVQ
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

acoustic_token_file = 'dataset/openaudio_ft.jsonl'

# name2speakerid = {"派蒙": 1, "温迪": 2, "纳西妲": 3, "八重神子": 4, "阿贝多": 5, "枫原万叶": 6}
name2speakerid = {"枫原万叶": 0}

@dataclass
class Data: 
    text: str
    acoustic_token: list[list]
    npcName: str
    score: float

class IceSummaryWriter:
    def __init__(self, log_dir='logs', includes = None):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.includes = includes
    
    def log_update_rate(self, model: torch.nn.Module, global_step: int, lr: float, num_fraction=23):
        weight_info = {}
        for (name, param) in model.named_parameters():
            if self.includes != None and name not in self.includes:
                continue
            
            param_abs = torch.abs(param)
            param_grad_abs = torch.abs(param.grad)

            epsilong = torch.pow(2, torch.log2(param_abs) - num_fraction) / 2
            update_sum = (torch.abs(param_grad_abs) * lr > epsilong).sum().item() / param.numel()

            weight_info[name] = update_sum
            
        self.writer.add_scalars(main_tag='update_rate', tag_scalar_dict=weight_info, global_step=global_step)
    
    def log_grad_mean(self, model: torch.nn.Module, global_step: int):
        weight_info = {}
        for (name, param) in model.named_parameters():
            if self.includes != None and name not in self.includes:
                continue
            
            param_grad_abs = torch.abs(param.grad)
            weight_info[name] = param_grad_abs.mean().item()
            
        self.writer.add_scalars(main_tag='grad_mean', tag_scalar_dict=weight_info, global_step=global_step)
    
    def log_loss(self, loss: float, global_step: int):
        self.writer.add_scalar(tag='loss', scalar_value=loss, global_step=global_step)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datas: list[Data], tokenizer: FishTokenizer):
        super().__init__()

        self.datas = datas
        self.len = len(datas)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        text = self.datas[index].text
        acoustic_token = np.array(self.datas[index].acoustic_token)

        vqs = list[TextAndVQ]()
        
        speakerid = name2speakerid[self.datas[index].npcName]
        vqs.append(TextAndVQ(speaker=speakerid, text=text, vq=acoustic_token))
        _result = self.tokenizer.encodebylist(vqs, isinference=False)

        # tokens: [T]
        # vq_mask_tokens: [T]
        # vq: [10, T]
        return _result.tokens, _result.vq_mask_tokens, _result.vq

def collate_fn(batch):
    # 分离输入序列和其他目标
    tokens, vq_mask_tokens, vqs = zip(*batch)
    tokens = list(tokens)
    vq_mask_tokens = list(vq_mask_tokens)
    vqs = list(vqs)
    
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    vq_mask_tokens = pad_sequence(vq_mask_tokens, batch_first=True, padding_value=False)
    vqs = torch.cat(vqs, dim=1)
    
    # tokens: [B, T]
    # vq_mask_tokens: [B, T]
    # vq: [10, T]
    return tokens, vq_mask_tokens, vqs

def train():
    device = 'cuda'
    path = 'checkpoints/openaudio-s1-mini'
    lr = 3e-4
    total_epoch = 2
    batch_size = 2

    armodel: DualARTransformer = DualARTransformer.from_pretrained(path, LoraConfig(r=128, lora_alpha=16))
    # armodel: DualARTransformer = DualARTransformer.from_pretrained(path)
    armodel = armodel.to(device=device, dtype=torch.bfloat16)
    tokenizer = FishTokenizer.from_pretrained(path)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimer = torch.optim.AdamW(armodel.parameters(), lr=lr)

    datas: list[Data] = []
    with open(acoustic_token_file, mode='r', encoding='utf8') as f:
        for line in f:
            _obj = json.loads(line)
            _data = Data(text=_obj['text'], acoustic_token=_obj['acoustic_token'], npcName=_obj['npcName'], score=None)
            if _data.npcName not in name2speakerid:
                continue
            if _data.text is None or _data.text.isspace():
                continue
            # if _data.score > 5.5:
            #     continue
            datas.append(_data)
    
    dataset = MyDataset(datas, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    num_step_preepoch = len(dataloader)
    num_step_total = num_step_preepoch * total_epoch
    num_warmup_steps = int(num_step_total / 50)

    # 学习率优化
    scheduler = get_cosine_schedule_with_warmup(
        optimer, num_warmup_steps=num_warmup_steps, num_training_steps=num_step_total
    )

    for epoch in range(total_epoch):
        for curstep, (tokens, vq_mask_tokens, vq) in enumerate(dataloader):
            # [B, T]
            tokens = tokens.to(device=device)
            # [B, T]
            vq_mask_tokens = vq_mask_tokens.to(device=device)
            # [T1'+T2'+..., 10]
            vq = vq.transpose(0, 1).to(device=device)

            result = armodel.forward(
                tokens=tokens,
                vq=vq,
                vq_mask_tokens=vq_mask_tokens
            )

            # [b, t, voc_size]
            token_logits: torch.Tensor = result.token_logits
            # [t', 10, cb_size]
            acoustic_token_logits: torch.Tensor = result.acoustic_token_logits

            batch_size = token_logits.size(0)
            cb_size = acoustic_token_logits.size(2)

            '''slow token损失值'''
            # 获取 vq 位置对应的输出
            _mask_end = tokens == tokenizer.im_end_id
            real_mask = vq_mask_tokens + _mask_end

            _temp = torch.zeros(size=(batch_size, 1), dtype=torch.bool, device=vq_mask_tokens.device.type)
            _mask = torch.cat([real_mask[:, 1:], _temp], dim=1)

            token_logits_predict = token_logits[_mask]
            token_real = tokens[real_mask]
            token_loss = loss_fn(token_logits_predict, token_real)
            
            
            '''acoustic token损失值'''
            acoustic_token_logits_predict = acoustic_token_logits[:, 1:, :].reshape(-1, cb_size)
            acoustic_token_real = vq[:, 1:].reshape(-1)

            acoustic_token_loss = loss_fn(acoustic_token_logits_predict, acoustic_token_real)
            

            '''梯度下降'''
            vqloss_num = token_real.shape[0]
            cbloss_num = acoustic_token_real.shape[0]
            total = vqloss_num + cbloss_num
            loss: torch.Tensor = token_loss * (vqloss_num / total) + acoustic_token_loss * (cbloss_num / total)

            optimer.zero_grad()
            loss.backward()
            optimer.step()
            scheduler.step()

            '''信息输出'''
            if (curstep + 1) % 10 == 0:
                print(f'当前 epoch {epoch}, 当前迭代 {curstep + 1}, 当前损失值 {loss.item()}')
                # global_steps = epoch * num_step_preepoch + curstep
                # writer.log_grad_mean(armodel, global_steps)
                # writer.log_loss(loss.item(), global_steps)
                # 清理内存
                del result.token_logits
                del result.acoustic_token_logits
                torch.cuda.empty_cache()
            
            if (curstep + 1) % 3000 == 0:
                armodel.save_pretrained(f"checkpoints/openaudio-s1-mini-{epoch}")
        
        armodel.save_pretrained(f"checkpoints/openaudio-s1-mini-{epoch}")

def print_model_layer():
    path = 'checkpoints/openaudio-s1-mini'

    armodel: DualARTransformer = DualARTransformer.from_pretrained(path)

    names = [name for name, _ in armodel.named_parameters()]
    print(json.dumps(names))
    

train()
# print_model_layer()