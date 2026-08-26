# gen_json_corpus_audio.py
import os
import sys
import json
import argparse
import torch
import numpy as np
import librosa
from scipy.io import wavfile

import commons

# Standard VITS imports from root repo directory
try:
    import utils
    from models import SynthesizerTrn
    from text import text_to_sequence
except ImportError:
    print("Error: Run this script from the root of your VITS repository.")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

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


# def synthesize_clean_chunk(text_chunk: str, vits_model, hparams, sid) -> np.ndarray:
#     """
#     Pads short inputs with neutral padding symbols to prevent edge boundary
#     slurring, runs inference, and trims lead-in/lead-out silences via librosa.
#     """
#     padded_text = f"… {text_chunk} …"

#     cleaner_names = getattr(hparams.data, "text_cleaners", [])
#     stn_tst, ipa = text_to_sequence(padded_text, cleaner_names, backend = "espeak", lang="en-us")
#     stn_tst = commons.intersperse(stn_tst, 0)    
#     stn_tst = torch.LongTensor(stn_tst)

#     inferDevice = next(vits_model.parameters()).device
#     x = stn_tst.unsqueeze(0).to(inferDevice)
#     x_lengths = torch.LongTensor([stn_tst.size(0)]).to(inferDevice)

#     with torch.no_grad():
#         audio = (
#             vits_model.infer(
#                 x,
#                 x_lengths,
#                 sid=sid,
#                 noise_scale=0.667,
#                 length_scale=1.0,
#                 noise_scale_w=0.6,
#             )[0][0, 0]
#             .data.to(device)
#             .cpu()
#             .numpy()
#         )
#         # .data.cpu().numpy()

#     # Trim synthetic edge silences (top_db=30 isolates active speech cleanly)
#     audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
#     return audio_trimmed


def synthesize_clean_chunk(text_chunk: str, vits_model, hparams, sid) -> np.ndarray:
    """
    Pads short inputs with neutral padding symbols to prevent edge boundary
    slurring, runs inference, and trims lead-in/lead-out silences via librosa.
    """
    padded_text = f"… {text_chunk} …"

    cleaner_names = getattr(hparams.data, "text_cleaners", [])
    text_norm, ipa = text_to_sequence(padded_text, cleaner_names, "espeak", "en-us")

    if (hparams.data.add_blank):
        text_norm = commons.intersperse(text_norm, 0)

    x = torch.LongTensor(text_norm).unsqueeze(0)
    x_lengths = torch.LongTensor([len(text_norm)])

    device = next(vits_model.parameters()).device
    x = x.to(device)
    x_lengths = x_lengths.to(device)

    with torch.no_grad():
        audio = vits_model.infer(
            x,
            x_lengths,
            sid=sid,
            noise_scale=0.333,
            length_scale=1.0,
            noise_scale_w=0.5,
        )[0][0, 0].data.cpu().numpy()

    # Trim synthetic edge silences (top_db=30 isolates active speech cleanly)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
    return audio_trimmed

def build_stitched_utterance(chunks, vits_model, hparams, sid) -> np.ndarray:
    """
    Synthesizes each chunk independently and concatenates them with exact PCM
    zero-padded silence based on trailing punctuation.
    """
    sr = hparams.data.sampling_rate
    audio_segments = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        audio_np = synthesize_clean_chunk(text, vits_model, hparams, sid)
        audio_segments.append(audio_np)

        punct = chunk.get("punct", "")
        pause_ms = PAUSE_DURATIONS_MS.get(punct, 100)  # Default 100ms micro-pause
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
    sid = torch.LongTensor([args.sid]).to(device) if args.sid else None

    output_dir = args.output_dir
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"Loading config: {args.config_path}")
    hps = utils.get_hparams_from_file(args.config_path)

    print(f"Running inference on: {device}")

    num_speakers = getattr(hps.data, "n_speakers", 0)
    net_g = SynthesizerTrn(
        len(hps.symbols)
        if hasattr(hps, "symbols")
        else hps.data.filter_length // 2 + 1,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=num_speakers,
        **hps.model,
    ).to(device)

    net_g.eval()
    print(f"Loading model checkpoint: {args.model_path}")
    utils.load_checkpoint(args.model_path, net_g, None)

    json_corpus = read_json_corpus(args.input)
    metadata_lines = []
    sr = hps.data.sampling_rate

    print(f"Processing {len(json_corpus)} utterances into target audio dataset...")

    for item in json_corpus:
        utt_id = item["id"]
        full_text = item["full_text"]
        chunks = item["chunks"]

        wav_filename = f"utt_{utt_id:05d}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        stitched_audio = build_stitched_utterance(chunks, net_g, hps, sid)

        # Peak normalization & float32 to int16 PCM conversion
        max_val = np.max(np.abs(stitched_audio))
        if max_val > 0:
            stitched_audio = stitched_audio / max_val * 0.95
        audio_int16 = (stitched_audio * 32767).astype(np.int16)

        wavfile.write(wav_path, sr, audio_int16)

        # Format line for Piper dataset CSV format (relative_path|text)
        metadata_lines.append(f"wavs/{wav_filename}|{full_text}")

        if utt_id % 50 == 0 or utt_id == len(json_corpus):
            print(f"Generated [{utt_id}/{len(json_corpus)}] -> {wav_filename}")

    # Write Piper metadata.csv
    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    print(f"\nDone! Stitched WAVs saved to: {wav_dir}")
    print(f"Piper metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate json corpus audio")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        required=True,
        help="Path to G^latest.pth",
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to config.json",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="JSON formatted corpus with segmentation",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./dataset_output",
        help="Output dataset directory",
    )
    parser.add_argument('-s', '--sid', type=int)    
    args = parser.parse_args()

    generate_json_corpus_cli(args)