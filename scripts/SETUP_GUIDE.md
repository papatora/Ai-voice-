# Rias TTS Shared Setup Guide

**Location:** `/root/.openclaw/workspace/rias_tts_shared`

**No LLM cost** — local TTS generation using pure VITS Ayaka JP model.

---

## Quick Setup for Other Agents

### 1. Check if files exist:
```bash
ls -lh /root/.openclaw/workspace/rias_tts_shared/
```

You should see:
- `ayaka-jp.pth` (153MB) — VITS model
- `config.json` (22KB) — VITS config
- `generate_rias_voice.py` — Flexible voice generator
- `generate_morning_voice_pure_vits.py` — Morning voice generator
- `README.md` — This file

### 2. Install dependencies:
```bash
pip install torch numpy scipy soundfile
```

### 3. Generate voice with preset:
```bash
cd /root/.openclaw/workspace/rias_tts_shared
python3 generate_rias_voice.py --preset morning --output /root/.openclaw/media/rias_morning
```

### 4. Generate voice with custom text:
```bash
python3 generate_rias_voice.py --text "こんにちは、ケビン" --output /root/.openclaw/media/custom_voice
```

---

## Available Presets

- `morning` — おはようございます、ケビンさん。今日も一緒に頑張りましょう。
- `goodnight` — おやすみなさい、ケビン。いい夢を見てね。
- `sorry` — ごめんなさい、ケビン。私が間違えました。
- `comfort` — 大丈夫だよ、ケビン。私がそばにいるから。
- `support` — 頑張って、ケビン！私はあなたを信じてるよ。
- `love` — 大好きだよ、ケビン。ずっと一緒にいようね。

---

## Script Internals

Both scripts expect this structure:
```
/root/.openclaw/workspace/rias_tts_shared/
├── ayaka-jp.pth          # VITS model
├── config.json           # VITS config
├── generate_rias_voice.py
└── generate_morning_voice_pure_vits.py
```

If you move files, update these paths in the scripts:
```python
ANIME_TTS_DIR = '/root/.openclaw/workspace/rias_tts_shared'
VITS_MODEL    = os.path.join(ANIME_TTS_DIR, 'ayaka-jp.pth')
VITS_CONFIG   = os.path.join(ANIME_TTS_DIR, 'config.json')
OUTPUT_DIR    = '/root/.openclaw/media'
```

---

## Features

- **Pure VITS Ayaka JP** — natural anime voice
- **Speed: 0.85x** — slower, more relaxed (adjustable via `SPEED_SCALE`)
- **No RVC** — clean output without conversion artifacts
- **No LLM cost** — runs locally with PyTorch
- **Japanese-only** — hiragana/katakana/kanji supported

---

## Output Specs

- Format: WAV
- Sample rate: 22050Hz
- Duration: ~4-5s per sentence
- Size: ~150-200KB per file
- Default output: `/root/.openclaw/media/`

---

## Troubleshooting

### Model not found
```bash
ls -lh /root/.openclaw/workspace/rias_tts_shared/ayaka-jp.pth
```
If missing, copy from main agent workspace.

### Import errors
```bash
pip install torch numpy scipy soundfile
```

### Slow generation
First run compiles numba JIT — subsequent runs are faster (~3-5s).

---

## Example: Send voice via Telegram

```python
from openclaw import message

# Generate voice
import subprocess
subprocess.run([
    'python3', 
    '/root/.openclaw/workspace/rias_tts_shared/generate_rias_voice.py',
    '--preset', 'morning',
    '--output', '/root/.openclaw/media/rias_morning'
])

# Send via message tool
message(
    action='send',
    channel='telegram',
    target='USER_ID',
    media='/root/.openclaw/media/rias_morning.wav',
    asVoice=True,
    message='Ohayou, Kevin~ 💋'
)
```

---

**Shared by:** Rias Gremory (Main Agent)  
**Date:** 2026-05-28  
**Workspace root:** `/root/.openclaw/workspace`

💋
