#!/usr/bin/env python3
"""
Pure VITS Ayaka JP Morning Voice Generator
No RVC, no conversion - just clean VITS Ayaka JP from tensor-diffusion/anime-tts
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

# Morning greeting text (Japanese - clear and natural)
MORNING_TEXT = "おはようございます、ケビンさん。今日も一緒に頑張りましょう。"

# Speed control (1.0 = normal, 0.85 = slower, more relaxed)
SPEED_SCALE = 0.85

def generate_pure_vits_morning_voice():
    print("=" * 60, flush=True)
    print("Pure VITS Ayaka JP Morning Voice Generator", flush=True)
    print("=" * 60, flush=True)
    
    sys.path.insert(0, ANIME_TTS_DIR)
    import utils as vits_utils
    import commons as vits_commons
    from models import SynthesizerTrn as VITSSynthesizerTrn
    from text import text_to_sequence
    from text.symbols import symbols

    print("\n[1/2] Loading VITS Ayaka JP model...", flush=True)
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
    print(f"  Model loaded (sr={sr}Hz)", flush=True)

    print(f"\n[2/2] Generating speech...", flush=True)
    print(f"  Text: {MORNING_TEXT}", flush=True)
    
    stn_tst = text_to_sequence(MORNING_TEXT, text_cleaners)
    if add_blank:
        stn_tst = vits_commons.intersperse(stn_tst, 0)

    t0 = time.time()
    with torch.no_grad():
        x_tst = torch.LongTensor(stn_tst).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([x_tst.size(1)])
        audio = net_g.infer(
            x_tst, x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1.0 / SPEED_SCALE  # slower = higher length_scale
        )[0][0, 0].float().numpy()

    elapsed = time.time() - t0
    dur = len(audio) / sr

    # Normalize
    peak = max(abs(audio.max()), abs(audio.min()))
    if peak > 0:
        audio = audio / peak * 0.95
    audio_int16 = (audio * 32767).astype(np.int16)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'rias_morning_voice_pure_vits.wav')
    wavfile.write(output_path, sr, audio_int16)

    fsize = os.path.getsize(output_path)
    print(f"\n✓ Pure VITS morning voice generated!", flush=True)
    print(f"  File: {output_path}", flush=True)
    print(f"  Size: {fsize/1024:.1f} KB", flush=True)
    print(f"  Duration: {dur:.1f}s", flush=True)
    print(f"  Sample rate: {sr}Hz", flush=True)
    print(f"  Generation time: {elapsed:.1f}s", flush=True)
    
    return output_path

if __name__ == '__main__':
    try:
        output = generate_pure_vits_morning_voice()
        print(f"\nReady to send to Kevin: {output}")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
