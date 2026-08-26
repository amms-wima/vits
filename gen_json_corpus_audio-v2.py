import os
import sys
import json
import argparse
import re
import torch
import numpy as np
import librosa
from scipy.io import wavfile

# Standard VITS imports from root repo directory
try:
    import utils
    import commons
    from models import SynthesizerTrn
    from text import text_to_sequence
except ImportError:
    print("Error: Run this script from the root of your VITS repository.")
    sys.exit(1)

PAUSE_DURATIONS_MS = {
    ",": 200,
    ";": 300,
    ":": 300,
    "—": 250,
    "–": 300,
    "-": 180,
    ".": 500,
    "!": 500,
    "?": 550,
    "…": 850,
    "•": 500,
}

def apply_edge_fades(audio_np: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    """Applies a smooth fade-in and fade-out to prevent boundary sibilance/clicks."""
    fade_samples = int((fade_ms / 1000.0) * sr)
    if len(audio_np) < fade_samples * 2:
        return audio_np

    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)

    audio_np[:fade_samples] *= fade_in
    audio_np[-fade_samples:] *= fade_out
    return audio_np


def synthesize_clean_chunk(text_chunk: str, vits_model, hparams, sid=None) -> np.ndarray:
    """
    Pads text, converts to sequence with intersperse blanks, runs VITS inference 
    with original scales, and trims edge silences.
    """
    # Safe padding string that won't break phonemizer
    padded_text = f"… {text_chunk} …"

    cleaner_names = getattr(hparams.data, "text_cleaners", [])
    
    # Get sequence IDs
    text_norm, _ = text_to_sequence(
        padded_text, 
        cleaner_names, 
        backend="espeak", 
        lang="en-us"
    )

    # CRITICAL: Reinsert blank tokens between phonemes for VITS models trained with add_blank
    if getattr(hparams.data, "add_blank", False):
        text_norm = commons.intersperse(text_norm, 0)

    x = torch.LongTensor(text_norm).unsqueeze(0)
    x_lengths = torch.LongTensor([len(text_norm)])

    device = next(vits_model.parameters()).device
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    
    if sid is not None:
        sid = torch.LongTensor([sid]).to(device)

    with torch.no_grad():
        # Matching your working tts.py parameters
        audio = vits_model.infer(
            x,
            x_lengths,
            sid=sid,
            noise_scale=0.667,
            length_scale=1.0,
            noise_scale_w=0.6,
        )[0][0, 0].data.cpu().numpy()

    # Trim padding silences cleanly
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
    return audio_trimmed


def build_stitched_utterance(chunks, vits_model, hparams, sid=None) -> np.ndarray:
    sr = hparams.data.sampling_rate
    audio_segments = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        audio_np = synthesize_clean_chunk(text, vits_model, hparams, sid=sid)
        audio_segments.append(audio_np)

        punct = chunk.get("punct", "")
        pause_ms = PAUSE_DURATIONS_MS.get(punct, 100)
        pause_samples = int((pause_ms / 1000.0) * sr)

        if pause_samples > 0:
            silence = np.zeros(pause_samples, dtype=np.float32)
            audio_segments.append(silence)

    if not audio_segments:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(audio_segments)


def read_json_corpus(json_corpus_file: str):
    with open(json_corpus_file, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_json_corpus_cli(args):
    output_dir = args.output_dir
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"Loading config: {args.config_path}")
    hps = utils.get_hparams_from_file(args.config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on: {device}")

    num_speakers = getattr(hps.data, "n_speakers", 0)
    net_g = SynthesizerTrn(
        len(hps.symbols) if hasattr(hps, "symbols") else hps.data.filter_length // 2 + 1,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=num_speakers,
        **hps.model,
    ).to(device)

    net_g.eval()
    print(f"Loading checkpoint: {args.model_path}")
    utils.load_checkpoint(args.model_path, net_g, None)

    json_corpus = read_json_corpus(args.input)
    metadata_lines = []
    sr = hps.data.sampling_rate
    sid = args.speaker_id if num_speakers > 1 else None

    print(f"Processing {len(json_corpus)} utterances...")

    for item in json_corpus:
        utt_id = item["id"]
        full_text = item["full_text"]
        chunks = item["chunks"]

        wav_filename = f"utt_{utt_id:05d}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        stitched_audio = build_stitched_utterance(chunks, net_g, hps, sid=sid)

        # Scale peak to 0.40 to match standard synthesis 0.wav amplitude
        max_val = np.max(np.abs(stitched_audio))
        if max_val > 0:
            stitched_audio = (stitched_audio / max_val) * 0.40

        audio_int16 = (stitched_audio * 32767).astype(np.int16)
        wavfile.write(wav_path, sr, audio_int16)

        metadata_lines.append(f"wavs/{wav_filename}|{full_text}")

        if utt_id % 50 == 0 or utt_id == len(json_corpus):
            print(f"Generated [{utt_id}/{len(json_corpus)}] -> {wav_filename}")

    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    print(f"\nSaved audio to: {wav_dir}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate json corpus audio")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to G^latest.pth")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config.json")
    parser.add_argument("-i", "--input", type=str, required=True, help="JSON corpus file")
    parser.add_argument("-o", "--output_dir", type=str, default="./dataset_output", help="Output directory")
    parser.add_argument("-sid", "--speaker_id", type=int, default=0, help="Speaker ID for multispeaker models")
    args = parser.parse_args()

    generate_json_corpus_cli(args)