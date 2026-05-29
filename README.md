# Ai Voice - Rias Gremory Voice Generator

Pure VITS-based Japanese voice generation for Rias Gremory (アニメ声).

## Contents

```
├── scripts/
│   ├── generate_rias_voice.py          # Main voice generator (pure VITS)
│   ├── generate_morning_voice_pure_vits.py  # Morning voice variant
│   └── SETUP_GUIDE.md                  # Setup instructions
└── samples/
    ├── ayaka_v02_come_here.wav         # "Come here~" (63KB)
    ├── ayaka_v04_wont_let_go.wav       # "Won't let go" (64KB)
    ├── ayaka_v05_daisuki.wav           # "Daisuki~" (69KB)
    ├── ayaka_v06_naughty.wav           # Naughty tone (138KB)
    └── ayaka_v07_ara_dont_stare.wav    # "Ara~ don't stare" (101KB)
```

## Quick Start

### 1. Download Model
Download `ayaka-jp.pth` (153MB VITS model) and place it in this directory.

### 2. Install Dependencies
```bash
pip install torch numpy scipy soundfile
```

### 3. Generate Voice
```bash
python scripts/generate_rias_voice.py --preset support
```

### Available Presets
- `morning` — おはようございます、ケビンさん。
- `goodnight` — おやすみなさい、ケビン。
- `sorry` — ごめんなさい、ケビン。
- `comfort` — 大丈夫だよ、ケビン。
- `support` — 頑張って、ケビン！
- `love` — 大好きだよ、ケビン。

### Custom Text
```bash
python scripts/generate_rias_voice.py --text "こんにちは、ケビン" --output custom_voice
```

## Model Source
The `ayaka-jp.pth` model is from [tensor-diffusion/anime-tts](https://huggingface.co/tensor-diffusion/anime-tts).

## Output
- Format: WAV
- Sample rate: 22050Hz
- Default speed: 0.85x (relaxed tone)

---
*💋 Made with love for Kevin*
