import os
import sys
import json
import argparse
import torch
import numpy as np
import librosa
import scipy.signal
from scipy.io import wavfile

# Standard VITS imports from root repository
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
    ")": 300,
    "]": 350,
    "}": 350,
}


def apply_edge_fades(audio_np: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    """Applies a smooth cosine fade-in and fade-out to prevent spectral pops."""
    fade_samples = int((fade_ms / 1000.0) * sr)
    if len(audio_np) < fade_samples * 2:
        return audio_np

    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)

    audio_np[:fade_samples] *= fade_in
    audio_np[-fade_samples:] *= fade_out
    return audio_np


def slice_carrier_core(y_full: np.ndarray, y_temp: np.ndarray, hop_length: int = 512) -> np.ndarray:
    """Performs RMS cross-correlation to slice target speech between carrier tokens."""
    env_full = librosa.feature.rms(y=y_full, hop_length=hop_length)[0]
    env_temp = librosa.feature.rms(y=y_temp, hop_length=hop_length)[0]

    corr = scipy.signal.correlate(env_full, env_temp, mode="valid")
    min_gap_frames = len(env_temp)

    peaks, _ = scipy.signal.find_peaks(
        corr,
        height=np.max(corr) * 0.35,
        distance=min_gap_frames,
    )

    if len(peaks) < 2:
        # Fallback to standard energy trim if envelope matching fails
        audio_trimmed, _ = librosa.effects.trim(y_full, top_db=30)
        return audio_trimmed

    first_token_frame = peaks[0]
    second_token_frame = peaks[-1]
    template_samples = len(y_temp)

    core_start = (first_token_frame * hop_length) + template_samples
    core_end = second_token_frame * hop_length

    if core_start >= core_end or core_start >= len(y_full):
        audio_trimmed, _ = librosa.effects.trim(y_full, top_db=30)
        return audio_trimmed

    return y_full[core_start:core_end]


def synthesize_chunk(text: str, vits_model, hparams, sid=None) -> np.ndarray:
    """Synthesizes text using VITS with intersperse blanks and target parameters."""
    cleaner_names = getattr(hparams.data, "text_cleaners", [])
    text_norm, _ = text_to_sequence(text, cleaner_names, backend="espeak", lang="en-us")

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
        audio = vits_model.infer(
            x,
            x_lengths,
            sid=sid,
            noise_scale=0.667,
            length_scale=1.0,
            noise_scale_w=0.6,
        )[0][0, 0].data.cpu().numpy()

    return audio


def build_stitched_utterance(chunks, vits_model, hparams, template_audio, sid=None) -> np.ndarray:
    sr = hparams.data.sampling_rate
    audio_segments = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        raw_audio = synthesize_chunk(text, vits_model, hparams, sid=sid)

        # Signal extraction: Envelope cross-correlation vs Standard Energy Trim
        if chunk.get("carrierIncluded", False) and template_audio is not None:
            core_audio = slice_carrier_core(raw_audio, template_audio)
        else:
            core_audio, _ = librosa.effects.trim(raw_audio, top_db=30)

        # Smooth boundaries and append
        core_audio = apply_edge_fades(core_audio, sr, fade_ms=10)
        audio_segments.append(core_audio)

        # Add exact target silence duration
        punct = chunk.get("punct", "")
        pause_ms = PAUSE_DURATIONS_MS.get(punct, 100)
        pause_samples = int((pause_ms / 1000.0) * sr)

        if pause_samples > 0:
            silence = np.zeros(pause_samples, dtype=np.float32)
            audio_segments.append(silence)

    if not audio_segments:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(audio_segments)


def generate_json_corpus_cli(args):
    output_dir = args.output_dir
    wav_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    print(f"Loading config: {args.config_path}")
    hps = utils.get_hparams_from_file(args.config_path)
    sr = hps.data.sampling_rate

    # Pre-load carrier template WAV
    template_audio = None
    if args.template_wav and os.path.exists(args.template_wav):
        print(f"Loading carrier template: {args.template_wav}")
        template_audio, _ = librosa.load(args.template_wav, sr=sr)
    else:
        print("Warning: Template WAV not specified or not found. Falling back to librosa.trim.")

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

    with open(args.input, "r", encoding="utf-8") as f:
        json_corpus = json.load(f)

    metadata_lines = []
    sid = args.speaker_id if num_speakers > 1 else None

    print(f"Processing {len(json_corpus)} utterances into training dataset...")

    for item in json_corpus:
        utt_id = item["id"]
        full_text = item["full_text"]
        chunks = item["chunks"]

        wav_filename = f"utt_{utt_id:05d}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        stitched_audio = build_stitched_utterance(chunks, net_g, hps, template_audio, sid=sid)

        # Normalize peak amplitude to 0.40 (~ -8 dBFS)
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

    print(f"\nSynthetic dataset generated successfully in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate json corpus audio")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to G^latest.pth")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to config.json")
    parser.add_argument("-i", "--input", type=str, required=True, help="Segmented JSON corpus file")
    parser.add_argument("-t", "--template_wav", type=str, default="/tmp/carrier_token2.wav", help="Path to carrier template WAV")
    parser.add_argument("-o", "--output_dir", type=str, default="./dataset_output", help="Output directory")
    parser.add_argument("-sid", "--speaker_id", type=int, default=0, help="Speaker ID")
    args = parser.parse_args()

    generate_json_corpus_cli(args)