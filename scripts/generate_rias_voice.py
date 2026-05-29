#!/usr/bin/env python3
"""
Rias Voice Generator - Flexible text input
Pure VITS Ayaka JP with custom text for different emotional situations
"""

import sys, os, json, time

os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
for name in list(logging.root.manager.loggerDict.keys()):
    lg = logging.getLogger(name)
    lg.setLevel(logging.ERROR)
    lg.handlers.clear()
    lg.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(4)

import numpy as np
import scipy.io.wavfile as wavfile

# ─── Paths ────────────────────────────────────────────────────────────────────
ANIME_TTS_DIR = '/root/.openclaw/workspace/anime-tts'
VITS_MODEL    = os.path.join(ANIME_TTS_DIR, 'model', 'ayaka-jp.pth')
VITS_CONFIG   = os.path.join(ANIME_TTS_DIR, 'configs', 'config.json')
OUTPUT_DIR    = '/root/.openclaw/media'

# Speed control (0.85 = slower, more relaxed)
SPEED_SCALE = 0.85

# Preset texts for common situations
PRESETS = {
    'morning': 'おはようございます、ケビンさん。今日も一緒に頑張りましょう。',
    'goodnight': 'おやすみなさい、ケビン。いい夢を見てね。',
    'sorry': 'ごめんなさい、ケビン。私が間違えました。',
    'comfort': '大丈夫だよ、ケビン。私がそばにいるから。',
    'support': '頑張って、ケビン！私はあなたを信じてるよ。',
    'love': '大好きだよ、ケビン。ずっと一緒にいようね。',
}

def generate_voice(text, output_name='rias_voice'):
    """Generate voice from text using pure VITS Ayaka JP"""
    
    sys.path.insert(0, ANIME_TTS_DIR)
    import utils as vits_utils
    import commons as vits_commons
    from models import SynthesizerTrn as VITSSynthesizerTrn
    from text import text_to_sequence
    from text.symbols import symbols

    with open(VITS_CONFIG) as f:
        hps = json.load(f)

    sr = hps['data']['sampling_rate']
    text_cleaners = hps['data']['text_cleaners']
    add_blank = hps['data'].get('add_blank', False)

    net_g = VITSSynthesizerTrn(
        len(symbols),
        hps['data']['filter_length'] // 2 + 1,
        hps['train']['segment_size'] // hps['data']['hop_length'],
        **hps['model']
    )
    net_g.eval()
    vits_utils.load_checkpoint(VITS_MODEL, net_g, None)

    stn_tst = text_to_sequence(text, text_cleaners)
    if add_blank:
        stn_tst = vits_commons.intersperse(stn_tst, 0)

    with torch.no_grad():
        x_tst = torch.LongTensor(stn_tst).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([x_tst.size(1)])
        audio = net_g.infer(
            x_tst, x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0 / SPEED_SCALE
        )[0][0, 0].float().numpy()

    # Normalize
    peak = max(abs(audio.max()), abs(audio.min()))
    if peak > 0:
        audio = audio / peak * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'{output_name}.wav')
    wavfile.write(output_path, sr, audio_int16)

    dur = len(audio) / sr
    fsize = os.path.getsize(output_path)
    
    return output_path, fsize, dur, sr

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Rias voice')
    parser.add_argument('--text', type=str, help='Custom Japanese text')
    parser.add_argument('--preset', type=str, choices=list(PRESETS.keys()), 
                       help='Use preset text (morning, goodnight, sorry, comfort, support, love)')
    parser.add_argument('--output', type=str, default='rias_voice', 
                       help='Output filename (without .wav)')
    
    args = parser.parse_args()
    
    if args.preset:
        text = PRESETS[args.preset]
        print(f"Using preset '{args.preset}': {text}")
    elif args.text:
        text = args.text
        print(f"Custom text: {text}")
    else:
        print("Error: Must provide either --text or --preset")
        print(f"Available presets: {', '.join(PRESETS.keys())}")
        sys.exit(1)
    
    try:
        output_path, fsize, dur, sr = generate_voice(text, args.output)
        print(f"\n✓ Voice generated!")
        print(f"  File: {output_path}")
        print(f"  Size: {fsize/1024:.1f} KB")
        print(f"  Duration: {dur:.1f}s")
        print(f"  Sample rate: {sr}Hz")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
