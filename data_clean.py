import json
import os
from utils.data_handler import Handler, ComminTextMiddleware, CommonAudioMiddleware, AudioData, TextScoreMiddleware
import numpy as np
import torch
import torchaudio
from openaudio.dac import DAC, DACConfig
from openaudio.inference import encode, decode
from datasets import load_dataset
import soundfile as sf

acoustic_token_file = 'dataset/openaudio_ft.jsonl'

# 提前将语音进行编码，方便后续进行训练
def encode_wav():
    text_handler = Handler[str]()
    text_handler.add_middleware(ComminTextMiddleware())

    audio_handler = Handler[AudioData]()
    audio_handler.add_middleware(CommonAudioMiddleware())

    device = 'cuda'
    dac = DAC(DACConfig.create())
    dac.load_state_dict(torch.load("checkpoints/openaudio-s1-mini/codec.pth"), strict=False, assign=True)
    dac = dac.to(device=device, dtype=torch.bfloat16)
    dac.eval()

    dataset = load_dataset(
        "hanamizuki-ai/genshin-voice-v3.3-mandarin", 
        split="train",
        cache_dir="./dataset")

    '''
    [{
        "audio": {
                "path": "vo_XMAQ103_3_candace_01.wav", 
                "array": [4.88281250e-04, ...], 
                "sampling_rate": 48000
        }, 
        "language": "CHS", 
        "npcName": "坎蒂丝", 
        "text": "喂——那边的人，这里这里！快来这边躲躲！", 
        "type": "Dialog"
    }...]
    '''

    with torch.no_grad():
        with open(acoustic_token_file, mode='w', encoding='utf8') as f:
            for i, item in enumerate(dataset, start=1):
                if i % 1000 == 0:
                    print(f'已处理 {i} 条数据')

                if item['npcName'] != '枫原万叶':
                    continue

                context = {}

                text = text_handler.handle(item['text'], context)
                if text is None:
                    continue

                audio = item['audio']['array']
                sr = item['audio']['sampling_rate']
                
                audio_data = audio_handler.handle(AudioData(audio=audio, sr=sr), context)

                audio = audio_data.audio
                sr = audio_data.sr

                audio = torch.tensor(audio, dtype=torch.float32)
                audio = audio[None, :]
                audio = audio.to(device=device)

                indices: torch.Tensor = encode(dac, audio, sr)
                indices = indices.cpu().tolist()
                line_obj = {
                    'text': text,
                    'acoustic_token': indices,
                    'npcName': item['npcName']
                }
                f.write(json.dumps(line_obj, ensure_ascii=False) + '\r')

star_rail_path = 'dataset/StarRail/三月七'
def encode_wav_star_rail():
    text_handler = Handler[str]()
    text_handler.add_middleware(ComminTextMiddleware())

    audio_handler = Handler[AudioData]()
    audio_handler.add_middleware(CommonAudioMiddleware())

    device = 'cuda'
    dac = DAC(DACConfig.create())
    dac.load_state_dict(torch.load("checkpoints/openaudio-s1-mini/codec.pth"), strict=False, assign=True)
    dac = dac.to(device=device, dtype=torch.bfloat16)
    dac.eval()

    with torch.no_grad():
        with open(acoustic_token_file, mode='w', encoding='utf8') as f:
            for file_name in os.listdir(star_rail_path):
                if not file_name.endswith('.lab'):
                    continue

                file_name = file_name.replace('.lab', '')
                with open(f'{star_rail_path}/{file_name}.lab', mode='r', encoding='utf8') as _tempf:
                    text = _tempf.read()

                context = {}
                text = text_handler.handle(text, context)
                if text is None:
                    continue
                    
                audio, sr = torchaudio.load(f'{star_rail_path}/{file_name}.wav')
                audio = torchaudio.functional.resample(audio, sr, 16_000)
                sr = 16_000
                audio = audio[0].numpy()
                audio_data = audio_handler.handle(AudioData(audio=audio, sr=sr), context)

                audio = audio_data.audio
                sr = audio_data.sr

                audio = torch.tensor(audio, dtype=torch.float32)
                audio = audio[None, :]
                audio = audio.to(device=device)

                indices: torch.Tensor = encode(dac, audio, sr)
                indices = indices.cpu().tolist()
                line_obj = {
                    'text': text,
                    'acoustic_token': indices,
                    'npcName': '三月七'
                }
                f.write(json.dumps(line_obj, ensure_ascii=False) + '\r')

def test():
    text = '你好，我是小小，我来自中国，以后请多多关照哈！'
    audio_path = 'temp.wav'

    # text = '我整理完今天的照片就休息啦，你也别熬夜打游戏哦！'
    # audio_path = 'dataset/StarRail/三月七/archive_mar7th_3.wav'

    text_handler = Handler[str]()
    text_handler.add_middleware(ComminTextMiddleware())

    audio_handler = Handler[AudioData]()
    audio_handler.add_middleware(CommonAudioMiddleware())

    device = 'cuda'
    dac = DAC(DACConfig.create())
    dac.load_state_dict(torch.load("checkpoints/openaudio-s1-mini/codec.pth"), strict=False, assign=True)
    dac = dac.to(device=device, dtype=torch.bfloat16)
    dac.eval()

    with torch.no_grad():
        audio, sr = torchaudio.load(audio_path)
        audio = audio[0].numpy()

        context = {}
        text = text_handler.handle(text, context)
        print(text)

        
        # audio_data = audio_handler.handle(AudioData(audio=audio, sr=sr), context)
        # audio = audio_data.audio
        # sr = audio_data.sr

        audio = torch.tensor(audio, dtype=torch.float32)
        audio = audio[None, :]
        indices: torch.Tensor = encode(dac, audio, sr)

        fake_audios, _ = decode(dac, indices)
        fake_audios = fake_audios[0, 0].float().cpu().detach().numpy()
        output_path = 'fake.wav'
        sf.write(output_path, fake_audios, dac.sample_rate)


if __name__ == '__main__':
    # encode_wav()
    encode_wav_star_rail()
    # test()
