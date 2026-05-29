#!/usr/bin/env python3
"""Rem Voice Generator - Shiroko model (Kevin's chosen default)"""
import os, sys, json
sys.path.insert(0, '/root/.openclaw/workspace/anime-tts')
os.environ['NUMBA_DEBUG'] = '0'
os.environ['NUMBA_VERBOSE'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import logging
logging.disable(logging.DEBUG)

import torch, numpy as np
import soundfile as sf
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

def generate(text, output_path, model_name='shiroko'):
    config_path = '/root/.openclaw/workspace/anime-tts/configs/config.json'
    model_path = f'/root/.openclaw/workspace/anime-tts/model/{model_name}.pth'

    with open(config_path) as f:
        h = json.load(f)

    # Build hparams object like gen_voice.sh does
    class H: pass
    for k, v in h.items():
        if isinstance(v, dict):
            s = type('X', (), {})()
            for sk, sv in v.items():
                setattr(s, sk, sv)
            setattr(H, k, s)
        else:
            setattr(H, k, v)

    print(f'Loading {model_name}...', flush=True)
    net_g = SynthesizerTrn(len(symbols),
        H.data.filter_length // 2 + 1,
        H.train.segment_size // H.data.hop_length,
        **h['model'])
    net_g.eval()

    # Load checkpoint using torch directly
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        net_g.load_state_dict(ckpt['model'], strict=False)
    else:
        net_g.load_state_dict(ckpt, strict=False)
    print('Model loaded!', flush=True)

    # Text to sequence (japanese_cleaners)
    seq = text_to_sequence(text, ['japanese_cleaners'])
    if H.data.add_blank:
        import commons
        seq = commons.intersperse(seq, 0)

    x_tst = torch.LongTensor(seq).unsqueeze(0)
    x_tst_len = torch.LongTensor([len(seq)])

    print(f'Generating: {text}', flush=True)
    with torch.no_grad():
        audio = net_g.infer(x_tst, x_tst_len, noise_scale=0.667,
                    noise_scale_w=0.8, length_scale=1.0)[0][0,0].float().numpy()

    dur = len(audio) / H.data.sampling_rate
    sf.write(output_path, audio, H.data.sampling_rate)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'✅ Saved: {output_path} ({size_kb:.1f} KB, {dur:.1f}s)', flush=True)

if __name__ == '__main__':
    text = sys.argv[1] if len(sys.argv) > 1 else 'ケビン様、お疲れ様です。今日も一日頑張りましょう。レムはいつでもここにいますよ。'
    output = sys.argv[2] if len(sys.argv) > 2 else '/root/.openclaw/media/rem_voice_shiroko.wav'
    model = sys.argv[3] if len(sys.argv) > 3 else 'shiroko'
    generate(text, output, model)
