#!/usr/bin/env python
"""
Generate VITS training audio with custom punctuation pause durations.
Uses WrappedDP to force specific pause lengths while preserving
model-predicted durations for all other tokens.
"""

import torch
import json
import csv
import os
import numpy as np
from scipy.io.wavfile import write
from torch import LongTensor
from models import SynthesizerTrn
import utils

# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
MODEL_PATH = "/home/ash/dev/vits/models/G^latest.pth"
CONFIG_PATH = "/home/ash/dev/vits/models/config.json"
SYMBOL_TABLE_PATH = "/home/ash/dev/dhammatalks.org-suttas/tts.model/vits_2023-symbol-table.json" # Your provided JSON
INPUT_CSV = "/home/ash/dev/dhammatalks.org-suttas/tts.train/corpus/rtx3050/test.cleaned.csv"
OUTPUT_DIR = "/home/ash/dev/dhammatalks.org-suttas/tts.train/corpus/rtx3050/test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SID = 129

# PAUSE_MAP_MS = {
#     ",": 200, ";": 300, ":": 300, "—": 250, "–": 300, "-": 180,
#     ".": 500, "!": 500, "?": 550, "…": 850, "•": 500,
#     "(": 200, ")": 300, "[": 250, "]": 350, "{": 250, "}": 350
# }
PAUSE_MAP_MS = {
    ",": "_",           # string, not int
    ".": "__",          # string
    "?": "___",
    "!": "___",
    ";": "____",
    ":": "____",
    "…": "_____",
}   
MIN_FRAMES = 5  # Minimum duration for non-punctuation tokens

# ═══════════════════════════════════════════════════════════
# LOAD CONFIG
# ═══════════════════════════════════════════════════════════
with open(CONFIG_PATH, 'r') as f:
    hps = json.load(f)

sampling_rate = hps['data']['sampling_rate']
hop_length = hps['data']['hop_length']
FRAMES_PER_MS = (sampling_rate / 1000.0) / hop_length
ADD_BLANK = hps['data'].get('add_blank', False)
print(f"SR: {sampling_rate}, Hop: {hop_length}, Frames/ms: {FRAMES_PER_MS:.4f}, add_blank: {ADD_BLANK}")

# ═══════════════════════════════════════════════════════════
# LOAD SYMBOL TABLE
# ═══════════════════════════════════════════════════════════
with open(SYMBOL_TABLE_PATH, 'r') as f:
    symbol_data = json.load(f)

char_to_id = {}
for char, ids in symbol_data.items():
    if isinstance(ids, list) and len(ids) > 0:
        char_to_id[char] = ids[0]
    elif isinstance(ids, int):
        char_to_id[char] = ids

punct_token_ids = set()
for char, ms in PAUSE_MAP_MS.items():
    if char in char_to_id:
        punct_token_ids.add(char_to_id[char])

print(f"Symbols: {len(char_to_id)}, Punct IDs: {len(punct_token_ids)}")

# ═══════════════════════════════════════════════════════════
# LOAD MODEL (exact same as tts.py)
# ═══════════════════════════════════════════════════════════
net_g = SynthesizerTrn(
    len(char_to_id),
    hps['data']['filter_length'] // 2 + 1,
    hps['train']['segment_size'] // hps['data']['hop_length'],
    n_speakers=hps['data']['n_speakers'],
    **hps['model']
).to(DEVICE)

_ = net_g.eval()
_ = utils.load_checkpoint(MODEL_PATH, net_g, None)
print(f"Model loaded: {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════
# TEXT TO IDS (with add_blank interspersing)
# ═══════════════════════════════════════════════════════════
def text_to_ids(phoneme_text):
    ids = [char_to_id[c] for c in phoneme_text if c in char_to_id]
    if ADD_BLANK:
        ids = [x for pair in zip(ids, [0] * len(ids)) for x in pair]
    return ids

# ═══════════════════════════════════════════════════════════
# BUILD LOG-DURATIONS
# ═══════════════════════════════════════════════════════════
def build_log_durations(text_ids, x, x_mask, g, original_dp):
    logw = original_dp(x, x_mask, w=None, g=g, reverse=True, noise_scale=0.6)

    for i, token_id in enumerate(text_ids):
        if token_id in punct_token_ids:
            for char_key, ms_val in PAUSE_MAP_MS.items():
                if char_to_id.get(char_key) == token_id:
                    frames = max(2, int(ms_val * FRAMES_PER_MS))
                    logw[0, 0, i] = torch.log(torch.tensor(float(frames)))
                    
                    # CRITICAL: Zero out the blank token AFTER the punctuation
                    if i + 1 < len(text_ids) and text_ids[i + 1] == 0:
                        logw[0, 0, i + 1] = torch.log(torch.tensor(1.0))  # 1 frame (~12ms)
                    break
    return logw   

def inject_pause_symbols(phoneme_text, pause_map):
    """
    Replace punctuation with the model's learned pause symbols.
    
    pause_map: {
        ',': '_',          # 1 blank → ~200ms
        '.': '__',         # 2 blanks → ~500ms  
        '?': '___',        # 3 blanks → ~550ms
        '!': '___',        # 3 blanks → ~550ms
        ';': '____',       # 4 blanks → ~300ms
        ':': '____',       # 4 blanks → ~300ms
    }
    """
    result = []
    for char in phoneme_text:
        if char in pause_map:
            result.append(pause_map[char])
        else:
            result.append(char)
    return ''.join(result)

# Usage:
# Original: "ʌŋˈɡʌs, mʌɡʌðhʌns, kʌsɪs"
# Modified: "ʌŋˈɡʌs_ mʌɡʌðhʌns_ kʌsɪs"  (commas → single blank)   

def text_to_ids_with_pauses(phoneme_text, pause_map):
    """Convert phoneme text to IDs, injecting pause symbols."""
    # First, inject the pause symbols
    modified = inject_pause_symbols(phoneme_text, pause_map)
    
    # Then convert to IDs (with add_blank interspersing)
    ids = [char_to_id[c] for c in modified if c in char_to_id]
    if ADD_BLANK:
        ids = [x for pair in zip(ids, [0] * len(ids)) for x in pair]
    return ids   

# ═══════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

original_dp = net_g.dp
sid = torch.LongTensor([SID]).to(DEVICE)

print(f"\nProcessing: {INPUT_CSV}\n")

with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    for row_idx, row in enumerate(reader):
        if len(row) < 2:
            continue
        filename = row[0].strip()
        phoneme_text = row[1].strip()

        # text_ids = text_to_ids(phoneme_text)
        text_ids = text_to_ids_with_pauses(phoneme_text, PAUSE_MAP_MS)
        if not text_ids:
            continue

        stn_tst = torch.LongTensor(text_ids).unsqueeze(0).to(DEVICE)
        stn_tst_lengths = LongTensor([len(text_ids)]).to(DEVICE)

        try:
            with torch.no_grad():
                # Encode to get x for duration building
                x, m_p, logs_p, x_mask = net_g.enc_p(stn_tst, stn_tst_lengths)
                g = net_g.emb_g(sid).unsqueeze(-1) if hasattr(net_g, 'emb_g') else None
                if g is None and hasattr(net_g.enc_p, 'emb_g'):
                    g = net_g.enc_p.emb_g(sid).unsqueeze(-1)

                log_durations = build_log_durations(text_ids, x, x_mask, g, original_dp)

                # DEBUG: Show which tokens are punctuation
                for i, token_id in enumerate(text_ids[:20]):
                    char = "?"
                    for c, tid in char_to_id.items():
                        if tid == token_id:
                            char = c
                            break
                    is_punct = "PUNCT" if token_id in punct_token_ids else ""
                    print(f"  [{i}] id={token_id} char='{char}' frames={torch.exp(log_durations[0,0,i]):.1f} {is_punct}")           


                # Patch DP and call infer
                # net_g.dp = WrappedDP(original_dp, log_durations)
                result = net_g.infer(
                    stn_tst, stn_tst_lengths, sid=sid,
                    noise_scale=0.667,
                    length_scale=1.0,
                    noise_scale_w=0.6,
                )

            audio_np = result[0][0, 0].data.cpu().float().numpy()
            audio_int16 = (audio_np * 32768.0).astype(np.int16)
            write(os.path.join(OUTPUT_DIR, filename), sampling_rate, audio_int16)
            print(f"  [{row_idx}] OK {filename} ({len(audio_int16)/sampling_rate:.2f}s)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{row_idx}] ERROR {filename}: {e}")
        finally:
            net_g.dp = original_dp
        exit(0)

print(f"\nDone. Output: {OUTPUT_DIR}")   