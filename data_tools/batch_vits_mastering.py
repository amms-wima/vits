# https://gemini.google.com/app/9a6f419c67b93d2a & https://gemini.google.com/app/0568aef821114554
# refer Solution 2: Fix the Harsh "Essing" in Piper / ONNX
"""
pip install numpy soundfile pyloudnorm scipy
"""

import csv
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, sosfilt

# --- Target Silence Durations (Seconds) ---
PAUSE_DURATIONS = {
    ",": 0.20,
    ";": 0.30,
    ":": 0.30,
    "—": 0.25,
    "-": 0.18,
    ".": 0.50,
    "!": 0.50,
    "?": 0.55,
    "...": 0.85,
    "…": 0.85,
}


@dataclass
class AudioProcessingConfig:
    """Master configuration with optimal default parameters for neural TTS post-processing."""

    # Model & Execution
    use_cuda: bool = True
    noise_scale: float = (
        0.667  # Lower values reduce metallic vocoder artifacts
    )
    noise_w_scale: float = 0.6  # Controls phoneme duration variance
    sample_rate: int = 22050

    # Boundary & Click Prevention
    enable_boundary_fades: bool = True
    fade_duration_ms: float = (
        5.0  # Smooth 5ms fade-in/out on each clause chunk
    )

    # Silence Trimming
    enable_silence_trim: bool = True
    silence_threshold_db: float = -45.0
    trim_padding_ms: float = 25.0  # Retains natural consonant release tail

    # Spectral Cleaning (Filters)
    enable_highpass: bool = True
    highpass_cutoff_hz: float = 60.0  # Cuts sub-bass rumble and DC offset

    enable_lowpass: bool = True
    lowpass_cutoff_hz: float = (
        9800.0  # Eliminates Nyquist aliasing & high hiss (for 22.05kHz)
    )

    # Dynamic Split-Band De-Esser (Sibilance Control)
    enable_deesser: bool = True
    deesser_low_hz: float = 6500.0  # Sibilance target band start
    deesser_high_hz: float = 9500.0  # Sibilance target band end
    deesser_threshold_db: float = -20.0  # Activation threshold for sibilants
    deesser_ratio: float = 4.0  # Dynamic compression ratio for 's' / 'z' band
    deesser_attack_ms: float = 2.0  # Fast attack for quick transient spikes
    deesser_release_ms: float = 40.0  # Fast release to prevent dulling voice

    # Dynamic Control & Loudness
    enable_compression: bool = True
    comp_threshold_db: float = -16.0  # Soft-knee compression to glue clauses
    comp_ratio: float = (
        1.8  # Gentle, transparent ratio (prevents pump/breathing)
    )
    comp_attack_ms: float = 15.0
    comp_release_ms: float = 120.0

    enable_lufs_norm: bool = True
    target_lufs: float = -16.0  # Standard podcast/audiobook loudness target
    max_peak_dbfs: float = -1.0  # True-peak limit to prevent clipping post-DAC


# --- DSP Helper Functions ---


def apply_micro_fades(
    audio: np.ndarray, sample_rate: int, fade_ms: float = 5.0
) -> np.ndarray:
    """Applies cosine fade-in and fade-out to prevent zero-crossing clicks at clause boundaries."""
    fade_samples = int((fade_ms / 1000.0) * sample_rate)
    if len(audio) <= fade_samples * 2:
        return audio

    faded_audio = audio.copy()
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_samples)))

    faded_audio[:fade_samples] *= fade_in
    faded_audio[-fade_samples:] *= fade_out
    return faded_audio


def trim_silence_with_padding(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -45.0,
    padding_ms: float = 25.0,
) -> np.ndarray:
    """Trims trailing silence while preserving a short padding tail for natural consonant decay."""
    threshold = 10 ** (threshold_db / 20.0)
    non_silent_indices = np.where(np.abs(audio) > threshold)[0]

    if len(non_silent_indices) == 0:
        return audio

    last_active = non_silent_indices[-1]
    padding_samples = int((padding_ms / 1000.0) * sample_rate)
    end_index = min(len(audio), last_active + padding_samples)

    return audio[:end_index]


def apply_butterworth_filters(
    audio: np.ndarray,
    sample_rate: int,
    enable_hp: bool = True,
    hp_cutoff: float = 60.0,
    enable_lp: bool = True,
    lp_cutoff: float = 9800.0,
) -> np.ndarray:
    """Applies high-pass and low-pass Butterworth filters to clean spectral extremes."""
    processed = audio.copy()
    nyquist = sample_rate / 2.0

    if enable_hp and hp_cutoff < nyquist:
        sos_hp = butter(
            2, hp_cutoff, btype="highpass", fs=sample_rate, output="sos"
        )
        processed = sosfilt(sos_hp, processed)

    if enable_lp and lp_cutoff < nyquist:
        sos_lp = butter(
            2, lp_cutoff, btype="lowpass", fs=sample_rate, output="sos"
        )
        processed = sosfilt(sos_lp, processed)

    return processed.astype(np.float32)


def apply_split_band_deesser(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float = 6500.0,
    high_hz: float = 9500.0,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 2.0,
    release_ms: float = 40.0,
) -> np.ndarray:
    """Isolates high-frequency sibilance (6.5kHz - 9.5kHz) and dynamically compresses sibilant spikes."""
    nyquist = sample_rate / 2.0
    safe_high_hz = min(high_hz, nyquist - 100.0)

    if low_hz >= safe_high_hz or len(audio) == 0:
        return audio

    # 1. Extract sibilance band via 2nd-order Butterworth bandpass filter
    sos_bp = butter(
        2, [low_hz, safe_high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    sibilant_band = sosfilt(sos_bp, audio)

    # Base band contains all low/mid frequencies and ultra-high air above 9.5kHz
    base_band = audio - sibilant_band

    # 2. Envelope Detection & Dynamic Gain Reduction on Sibilance Band
    threshold = 10 ** (threshold_db / 20.0)
    attack_coeff = np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
    release_coeff = np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))

    envelope = 0.0
    compressed_sibilance = np.zeros_like(sibilant_band)

    for i in range(len(sibilant_band)):
        abs_val = abs(sibilant_band[i])
        if abs_val > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * abs_val
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * abs_val

        if envelope > threshold and envelope > 0:
            gain_db = (threshold_db - 20 * np.log10(envelope)) * (
                1.0 - 1.0 / ratio
            )
            gain = 10 ** (gain_db / 20.0)
        else:
            gain = 1.0

        compressed_sibilance[i] = sibilant_band[i] * gain

    # 3. Re-combine non-sibilant audio with compressed sibilance
    deessed_audio = base_band + compressed_sibilance
    return deessed_audio.astype(np.float32)


def apply_soft_compressor(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -16.0,
    ratio: float = 1.8,
    attack_ms: float = 15.0,
    release_ms: float = 120.0,
) -> np.ndarray:
    """Transparent single-band compressor to smooth loudness variance across synthesized chunks."""
    threshold = 10 ** (threshold_db / 20.0)
    attack_coeff = np.exp(-1.0 / (sample_rate * (attack_ms / 1000.0)))
    release_coeff = np.exp(-1.0 / (sample_rate * (release_ms / 1000.0)))

    envelope = 0.0
    compressed_audio = np.zeros_like(audio)

    for i in range(len(audio)):
        abs_val = abs(audio[i])
        if abs_val > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * abs_val
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * abs_val

        if envelope > threshold and envelope > 0:
            gain_db = (threshold_db - 20 * np.log10(envelope)) * (
                1.0 - 1.0 / ratio
            )
            gain = 10 ** (gain_db / 20.0)
        else:
            gain = 1.0

        compressed_audio[i] = audio[i] * gain

    return compressed_audio.astype(np.float32)


def normalize_lufs(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -16.0,
    max_peak_dbfs: float = -1.0,
) -> np.ndarray:
    """ITU-R BS.1770-4 loudness normalization with peak ceiling limiting."""
    meter = pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio)
    except ValueError:
        return audio

    if np.isinf(loudness) or np.isnan(loudness):
        return audio

    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)

    # True Peak Limiter Ceiling
    max_allowed_peak = 10 ** (max_peak_dbfs / 20.0)
    current_peak = np.max(np.abs(normalized))

    if current_peak > max_allowed_peak:
        normalized = (normalized / current_peak) * max_allowed_peak

    return normalized.astype(np.float32)


def generate_silence(duration_sec: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)


# --- Core Synthesis Pipeline ---


def synthesize_clause(
    ipa_text: str,
    cli_cfg_path: str,
    output_wav: str,
    speaker_id: int,
    cfg: AudioProcessingConfig,
) -> bool:
    """Executes VITS CLI synthesis."""
    cmd = [
        "python",
        "tts.py",
        "--cli_config",
        str(cli_cfg_path),
        "--output_file",
        str(output_wav),
        "--stdin",
        "-s",
        str(speaker_id),
        "--noise_scale",
        str(cfg.noise_scale),
        "--noise_scale_w",
        str(cfg.noise_w_scale),
    ]
    if cfg.use_cuda:
        cmd.append("--cuda")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        process.communicate(input=ipa_text)
        return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0
    except Exception as e:
        print(f"Subprocess error: {e}")
        return False


def process_single_ipa(
    ipa_input: str,
    cli_cfg_path: str,
    output_path: str,
    speaker_id: int,
    cfg: AudioProcessingConfig,
) -> bool:
    """Main processing loop per audio file."""
    raw_text = ipa_input.strip()
    if raw_text.startswith("[[") and raw_text.endswith("]]"):
        raw_text = raw_text[2:-2].strip()

    pattern = r"(\.\.\.|[\,\;\:\.\!\?\—\-\…])"
    tokens = re.split(pattern, raw_text)
    audio_segments = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        chunk_idx = 0
        current_text = ""

        for token in tokens:
            if not token:
                continue

            current_text += token

            if token in PAUSE_DURATIONS:
                clause_ipa = current_text.strip() # f"[[ {current_text.strip()} ]]"
                chunk_wav = os.path.join(tmp_dir, f"chunk_{chunk_idx}.wav")

                if synthesize_clause(
                    clause_ipa, cli_cfg_path, chunk_wav, speaker_id, cfg
                ):
                    data, sr = sf.read(chunk_wav, dtype="float32")

                    if cfg.enable_silence_trim:
                        data = trim_silence_with_padding(
                            data,
                            sr,
                            cfg.silence_threshold_db,
                            cfg.trim_padding_ms,
                        )

                    if cfg.enable_boundary_fades:
                        data = apply_micro_fades(data, sr, cfg.fade_duration_ms)

                    audio_segments.append(data)

                    silence_len = PAUSE_DURATIONS[token]
                    audio_segments.append(generate_silence(silence_len, sr))

                current_text = ""
                chunk_idx += 1

        if current_text.strip():
            clause_ipa = f"[[ {current_text.strip()} ]]"
            chunk_wav = os.path.join(tmp_dir, f"chunk_{chunk_idx}.wav")
            if synthesize_clause(
                clause_ipa, cli_cfg_path, chunk_wav, speaker_id, cfg
            ):
                data, sr = sf.read(chunk_wav, dtype="float32")
                if cfg.enable_silence_trim:
                    data = trim_silence_with_padding(
                        data, sr, cfg.silence_threshold_db, cfg.trim_padding_ms
                    )
                if cfg.enable_boundary_fades:
                    data = apply_micro_fades(data, sr, cfg.fade_duration_ms)
                audio_segments.append(data)

    if audio_segments:
        final_audio = np.concatenate(audio_segments)

        # 1. High-Pass & Low-Pass Filtering
        if cfg.enable_highpass or cfg.enable_lowpass:
            final_audio = apply_butterworth_filters(
                final_audio,
                cfg.sample_rate,
                cfg.enable_highpass,
                cfg.highpass_cutoff_hz,
                cfg.enable_lowpass,
                cfg.lowpass_cutoff_hz,
            )

        # 2. Dynamic Split-Band De-Essing (Targeted Sibilance Control)
        if cfg.enable_deesser:
            final_audio = apply_split_band_deesser(
                final_audio,
                cfg.sample_rate,
                cfg.deesser_low_hz,
                cfg.deesser_high_hz,
                cfg.deesser_threshold_db,
                cfg.deesser_ratio,
                cfg.deesser_attack_ms,
                cfg.deesser_release_ms,
            )

        # 3. Transparent Broadband Dynamic Compression
        if cfg.enable_compression:
            final_audio = apply_soft_compressor(
                final_audio,
                cfg.sample_rate,
                cfg.comp_threshold_db,
                cfg.comp_ratio,
                cfg.comp_attack_ms,
                cfg.comp_release_ms,
            )

        # 4. Integrated LUFS Normalization with Peak Limiting
        if cfg.enable_lufs_norm:
            final_audio = normalize_lufs(
                final_audio,
                cfg.sample_rate,
                cfg.target_lufs,
                cfg.max_peak_dbfs,
            )

        sf.write(output_path, final_audio, cfg.sample_rate)
        return True

    return False


def process_ljspeech_csv(
    csv_path: str,
    cli_cfg_path: str,
    output_dir: str,
    cfg: AudioProcessingConfig,
    id_col: int = 0,
    text_col: int = 1,
    speaker_col: int | None = None,
    default_speaker_id: int = 0,
    delimiter: str = "|",
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(cli_cfg_path):
        raise FileNotFoundError(f"VITS CLI CFG file not found: {cli_cfg_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        total_rows = 0
        success_count = 0

        for row_idx, row in enumerate(reader, 1):
            req = max(
                id_col, text_col, speaker_col if speaker_col is not None else 0
            )
            if not row or len(row) <= req:
                continue

            file_id = row[id_col].strip()
            ipa_text = row[text_col].strip()

            speaker_id = default_speaker_id
            if speaker_col is not None and len(row) > speaker_col:
                try:
                    speaker_id = int(row[speaker_col].strip())
                except ValueError:
                    pass

            if not ipa_text:
                continue

            output_file = os.path.join(output_dir, f"{file_id}")
            total_rows += 1

            print(
                f"[{total_rows}] Synthesizing {file_id} (Speaker: {speaker_id})..."
            )
            if process_single_ipa(
                ipa_text, cli_cfg_path, output_file, speaker_id, cfg
            ):
                success_count += 1

    print(
        f"\nProcessing Complete. Generated {success_count}/{total_rows} audio files."
    )


if __name__ == "__main__":
    # --- Initialize Default Configuration ---
    config = AudioProcessingConfig(
        use_cuda=False,
        noise_scale=0.667,      # 0.333
        noise_w_scale=0.6,      # 0.500
        sample_rate=22050,
        # Micro Fades & Trimming
        enable_boundary_fades=True,
        fade_duration_ms=5.0,
        enable_silence_trim=True,
        trim_padding_ms=25.0,
        # Filtering (High-pass @ 60Hz, Low-pass @ 9.8kHz)
        enable_highpass=True,
        highpass_cutoff_hz=60.0,
        enable_lowpass=True,
        lowpass_cutoff_hz=9800.0,
        # Dynamic Split-Band De-Esser (6.5kHz - 9.5kHz)
        enable_deesser=True,
        deesser_low_hz=6500.0,
        deesser_high_hz=9500.0,
        deesser_threshold_db=-20.0,
        deesser_ratio=4.0,
        deesser_attack_ms=2.0,
        deesser_release_ms=40.0,
        # Dynamic Processing & Mastering
        enable_compression=True,
        comp_threshold_db=-16.0,
        comp_ratio=1.8,
        enable_lufs_norm=True,
        target_lufs=-16.0,
        max_peak_dbfs=-1.0,
    )

    # --- Paths ---
    CSV_FILE = "/home/ash/prj/di/dhammatalks.org-suttas/word-lists/tts-finetune/colab/metadata-colab-enh-vits-no-sid.csv.cleaned.csv"
    CLI_CFG_PATH = "/home/ash/prj/tts-eval/vits/tmp/tts-config-corpus_pauses.json"
    OUTPUT_DIRECTORY = "/home/ash/prj/tts-eval/vits/tmp/pauses"

    process_ljspeech_csv(
        csv_path=CSV_FILE,
        cli_cfg_path=CLI_CFG_PATH,
        output_dir=OUTPUT_DIRECTORY,
        cfg=config,
        id_col=0,
        text_col=2,
        speaker_col=None,
        default_speaker_id=129,
        delimiter="|",
    )