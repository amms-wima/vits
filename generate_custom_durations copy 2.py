import torch
import json
import csv
import os
import librosa
import numpy as np
from scipy.io.wavfile import write
from models import SynthesizerTrn  # From original VITS repo
from text import text_to_sequence # From original VITS repo (or your custom text module)
# from monotonic_align import maximum_path, mask_from_lengths

# Remove the failing import
# from monotonic_align import maximum_path, mask_from_lengths 

# Use this instead:
from monotonic_align import maximum_path
import torch

# Define the missing helper function directly in your script
def sequence_mask(length, max_length=None):
    """Creates a binary mask from a length tensor."""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def mask_from_lengths(lengths, max_length=None):
    """Wrapper to match the expected function signature."""
    return sequence_mask(lengths, max_length).float()   

# --- CONFIGURATION ---
MODEL_PATH = "/home/ash/dev/vits/models/G^latest.pth"
CONFIG_PATH = "/home/ash/dev/vits/models/config.json"
SYMBOL_TABLE_PATH = "/home/ash/dev/dhammatalks.org-suttas/tts.model/vits_2023-symbol-table.json" # Your provided JSON
INPUT_CSV = "/home/ash/dev/dhammatalks.org-suttas/tts.train/corpus/rtx3050/test.cleaned.csv"
OUTPUT_DIR = "/home/ash/dev/dhammatalks.org-suttas/tts.train/corpus/rtx3050/test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Your Preferred Punctuation Table (milliseconds)
PAUSE_MAP_MS = {
    ",": 200, ";": 300, ":": 300, "—": 250, "–": 300, "-": 180,
    ".": 500, "!": 500, "?": 550, "…": 850, "•": 500,
    "(": 200, ")": 300, "[": 250, "]": 350, "{": 250, "}": 350
}

# Load Config
with open(CONFIG_PATH, 'r') as f:
    hps = json.load(f)

sampling_rate = hps['data']['sampling_rate']
hop_length = hps['data']['hop_length']
FRAMES_PER_MS = (sampling_rate / 1000) / hop_length

print(f"Sampling Rate: {sampling_rate}, Hop: {hop_length}, Frames/ms: {FRAMES_PER_MS:.4f}")

# Load Symbol Table to map Characters -> Token IDs
# Your JSON format: { "_": [0], " ": [16], ";": [1], ... }
with open(SYMBOL_TABLE_PATH, 'r') as f:
    symbol_data = json.load(f)

# Create a reverse map: Character -> ID (taking the first ID if list)
char_to_id = {}
for char, ids in symbol_data.items():
    if isinstance(ids, list) and len(ids) > 0:
        char_to_id[char] = ids[0]
    elif isinstance(ids, int):
        char_to_id[char] = ids

# Identify Punctuation Token IDs
punct_token_ids = set()
for char, ms in PAUSE_MAP_MS.items():
    if char in char_to_id:
        punct_token_ids.add(char_to_id[char])
    else:
        print(f"WARNING: Character '{char}' not found in symbol table!")

# Load Model
n_vocab = len(char_to_id) # Or extract from config if different
# Extract necessary params from config for SynthesizerTrn init
# Note: You might need to manually pass these if config structure differs slightly
model_params = hps.get('model', hps) # Handle nested or flat config

net_g = SynthesizerTrn(
    n_vocab=n_vocab,
    spec_channels=hps['data']['filter_length'] // 2 + 1,
    segment_size=hps['train'].get('segment_size', 0) // hps['data']['hop_length'],
    inter_channels=model_params.get('inter_channels', 192),
    hidden_channels=model_params.get('hidden_channels', 192),
    filter_channels=model_params.get('filter_channels', 768),
    n_heads=model_params.get('n_heads', 2),
    n_layers=model_params.get('n_layers', 6),
    kernel_size=model_params.get('kernel_size', 3),
    p_dropout=model_params.get('p_dropout', 0.1),
    resblock=model_params.get('resblock', "1"),
    resblock_kernel_sizes=model_params.get('resblock_kernel_sizes', [3, 7, 11]),
    resblock_dilation_sizes=model_params.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
    upsample_rates=model_params.get('upsample_rates', [8, 8, 2, 2]),
    upsample_initial_channel=model_params.get('upsample_initial_channel', 512),
    upsample_kernel_sizes=model_params.get('upsample_kernel_sizes', [16, 16, 4, 4]),
    n_speakers=model_params.get('n_speakers', 0),
    gin_channels=model_params.get('gin_channels', 0),
    use_sdp=model_params.get('use_sdp', True)
).to(DEVICE)

_ = net_g.eval()


# Load Checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if 'state_dict' in checkpoint:
    net_g.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    net_g.load_state_dict(checkpoint, strict=False)

print(f"Model loaded from {MODEL_PATH}")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def text_to_ids_custom(phoneme_text):
    """
    Converts phoneme string to ID list using your symbol table.
    Assumes the cleaned CSV contains characters present in your symbol table.
    """
    ids = []
    for char in phoneme_text:
        if char in char_to_id:
            ids.append(char_to_id[char])
        else:
            # Handle unknown chars (skip or map to unknown token if you have one)
            # For Pali/English, ensure your symbol table covers all IPA symbols used.
            pass 
    return ids

# Load the speaker embedding extracted from ONNX
speaker_emb_weight = torch.load("/home/ash/dev/vits/models/speaker_emb_130x256.pth", map_location=DEVICE)
print(f"Loaded speaker embeddings: shape {speaker_emb_weight.shape}")  # Should be [130, 256]   

sid = torch.LongTensor([129]).to(DEVICE)

# --- MAIN LOOP ---
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        if len(row) < 2:
            continue
        filename, phoneme_text = row[0].strip(), row[1].strip()
        
        # 1. Convert Text to IDs
        text_ids = text_to_ids_custom(phoneme_text)
        if not text_ids:
            print(f"Skipping {filename}: No valid tokens found.")
            continue
            
        stn_tst = torch.LongTensor(text_ids).unsqueeze(0).to(DEVICE)
        stn_tst_lengths = torch.LongTensor([len(text_ids)]).to(DEVICE)

        # 2. Construct Manual Duration Tensor
        # Shape: [Batch, Time] -> [1, len(text_ids)]
        durations = torch.zeros_like(stn_tst, dtype=torch.long)
        
        for i, token_id in enumerate(text_ids):
            # Check if this token ID corresponds to a punctuation mark
            # We need to know which character produced this ID. 
            # Since multiple chars could map to same ID (unlikely for punct), 
            # we reverse lookup. 
            # Better: Iterate the string and map directly.
            char = phoneme_text[i] if i < len(phoneme_text) else None
            
            # Robust check: Does the ID at this position match a known punct ID?
            if token_id in punct_token_ids:
                # Find which char corresponds to this ID to get the correct MS
                # (Simple 1-to-1 assumption based on your CSV structure)
                # We iterate PAUSE_MAP to find the matching ID
                found = False
                for char_key, ms_val in PAUSE_MAP_MS.items():
                    if char_to_id.get(char_key) == token_id:
                        frame_count = int(ms_val * FRAMES_PER_MS)
                        durations[0, i] = max(1, frame_count) # Ensure at least 1 frame
                        found = True
                        break
                if not found:
                    # Fallback: Assign a standard punctuation duration if ID matches but char lookup fails
                    durations[0, i] = int(300 * FRAMES_PER_MS) 
            else:
                # For non-punctuation, we CANNOT easily set a duration without the model.
                # STRATEGY: We will let the model predict durations for non-punct, 
                # but OVERRIDE the punctuation indices in the final duration vector.
                # However, the standard forward() call calculates ALL durations internally.
                # We must bypass the standard forward() to inject our hybrid durations.
                durations[0, i] = 0 # Placeholder, will be overwritten or handled below
                pass


        # 3. Inference with Duration Override
        with torch.no_grad():
            stn_tst = torch.LongTensor(text_ids).unsqueeze(0).to(DEVICE)
            stn_tst_lengths = torch.LongTensor([len(text_ids)]).to(DEVICE)
            
            # Build custom log-durations [1, 1, T_text]
            log_durations = torch.zeros(1, 1, len(text_ids)).to(DEVICE)
            for i, token_id in enumerate(text_ids):
                if token_id in punct_token_ids:
                    for char_key, ms_val in PAUSE_MAP_MS.items():
                        if char_to_id.get(char_key) == token_id:
                            frames = max(1, int(ms_val * FRAMES_PER_MS))
                            log_durations[0, 0, i] = torch.log(torch.tensor(float(frames)))
                            break
                else:
                    log_durations[0, 0, i] = torch.log(torch.tensor(5.0))  # Default ~58ms

            # Monkey-patch the duration predictor
            original_dp = net_g.dp
            class CustomDP(torch.nn.Module):
                def forward(self, x, x_mask, g=None, reverse=False, noise_scale=0.667):
                    return log_durations * x_mask
            net_g.dp = CustomDP()
            
            
            # Call infer
            audio = net_g.infer(
                stn_tst,
                stn_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1.0,
            )[0][0, 0].cpu().numpy()
            
            # Restore
            net_g.dp = original_dp

            # infer returns a 1D numpy array directly
            audio_np = audio

            # Save audio
            audio_int16 = (audio_np * 32768.0).astype(np.int16)
            write(os.path.join(OUTPUT_DIR, filename), sampling_rate, audio_int16)   
            print(f"Generated: {filename}")   

print("Dataset generation complete.")   