import os
import torch

from models import (SynthesizerTrn)
from scipy.io.wavfile import write

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ͡"
_letters_rime_phonemes = ",i#DmT*XSClMpjYAt()Wb1N-w.ezZ&y!af\"xI@2`hJsrOE?0gRdU'Gvuok:n…^"


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa) + list(_letters_rime_phonemes)

filter_length = 1024
hop_length = 256
segment_size = 8192

hps_model = {"inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        "upsample_rates": [8,8,2,2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16,16,4,4],
        "n_layers_q": 3,
        "use_spectral_norm": False
    }

def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    state_dict = model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    return model, optimizer, learning_rate, iteration

net_g = SynthesizerTrn(
    len(symbols),
    filter_length // 2 + 1,
    segment_size // hop_length,
    **hps_model).cuda()

_ = net_g.eval()

_ = load_checkpoint(os.path.join('checkpoints', "net_g.pt"), net_g)

_symbol_to_id = {s: i for i, s in enumerate(symbols)}

def text_to_sequence(text):
    sequence = []

    # clean_text = _clean_text(text, cleaner_names)
    clean_text = text
    
    # convert cleaned text to sequence like [1, 3, 5]
    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id] 
    return sequence

stn_tst = torch.LongTensor(text_to_sequence("ð ə ɹ ˈoʊ m ə n l ˈɛ t ɚ w ə z j ˈu s t s ˈaɪ d b ˈaɪ s ˈaɪ d w ˈɪ θ ð ə ɡ ˈɑ θ ɪ k ."))
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    # the .infer() is showed in the picture.
    audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    write(data=audio, rate=22_050, filename="infer.wav")
