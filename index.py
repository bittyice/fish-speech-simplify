import librosa
import numpy as np
import torchaudio
import torch
import soundfile as sf

from openaudio.dac import DAC, DACConfig
from openaudio.dual_ar import DualARTransformer, LoraConfig
from openaudio.inference import generate_acoustic_token, decode, encode


def test_openaudio():
    device = 'cuda'

    dac = DAC(DACConfig.create())
    dac.load_state_dict(torch.load("checkpoints/openaudio-s1-mini/codec.pth"), strict=False, assign=True)
    dac = dac.to(device=device, dtype=torch.bfloat16)
    dac.eval()

    # armodel = DualARTransformer.from_pretrained('checkpoints/openaudio-s1-mini-1', LoraConfig(r=128, lora_alpha=16))
    armodel = DualARTransformer.from_pretrained('checkpoints/openaudio-s1-mini')
    armodel = armodel.to(device=device, dtype=torch.bfloat16)
    armodel.eval()

    with torch.no_grad():
        text = "他之前是这么说的，但不代表他之后也都这么想。"
        
        # audio, sr = torchaudio.load(str("./Temp.wav"))
        # prompt_text = '你好，我是小小，我来自中国，以后请多多关照哈！'
        # prompt_acoustic_tokens: torch.Tensor = encode(dac, audio, sr)
        # prompt_acoustic_tokens = prompt_acoustic_tokens.cpu().numpy()

        # prompt_text = '大家跟我来吧。'
        # prompt_acoustic_tokens = np.array([[1791, 2897, 3392, 3177, 1571, 684, 3355, 1866, 1185, 1834, 2508, 1992, 259, 3692, 2617], [872, 264, 45, 663, 408, 94, 99, 393, 615, 294, 406, 268, 1019, 172, 529], [91, 453, 741, 133, 692, 680, 184, 65, 602, 151, 449, 836, 617, 812, 88], [747, 285, 339, 887, 95, 440, 894, 604, 680, 50, 708, 429, 515, 719, 770], [396, 338, 838, 919, 447, 914, 136, 609, 679, 272, 737, 226, 501, 367, 582], [93, 734, 125, 734, 974, 1014, 266, 685, 680, 624, 436, 320, 734, 482, 9], [466, 992, 393, 794, 147, 493, 593, 87, 760, 695, 126, 314, 745, 689, 608], [1008, 927, 476, 172, 56, 142, 355, 867, 336, 205, 613, 480, 858, 751, 914], [862, 181, 614, 55, 713, 294, 451, 54, 998, 847, 39, 296, 456, 41, 885], [249, 799, 307, 310, 884, 724, 820, 24, 10, 850, 81, 516, 587, 351, 1]])

        prompt_text = None
        prompt_acoustic_tokens = None

        acoustic_tokens = generate_acoustic_token(armodel, text, prompt_text, prompt_acoustic_tokens)
        fake_audios, sample_rate = decode(dac, acoustic_tokens)
        fake_audios = fake_audios[0, 0].float().cpu().numpy()
        
        output_path = 'fake.wav'
        sf.write(output_path, fake_audios, sample_rate)
        print(f"Saved audio to {output_path}")


test_openaudio()
