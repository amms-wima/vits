#!/usr/bin/env python3
import argparse
import re
import unicodedata
from pathlib import Path
from pydub import AudioSegment
import torch
import torchaudio
from tqdm import tqdm

# Preferred Punctuation Table (Millisecond Pause Durations)
PUNCT_TABLE = {
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
    "(": 200,
    "[": 250,
    "{": 250,
    ")": 300,
    "]": 350,
    "}": 350,
}


def strip_pali_diacritics(text: str) -> str:
    """Decomposes Unicode characters and strips non-spacing combining marks."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def parse_punctuated_words(text: str):
    """Splits text into tokens and maps words to clean ASCII for CTC alignment."""
    tokens = text.split()
    parsed = []

    for idx, token in enumerate(tokens):
        lead_match = re.match(r"^([(\[{]+)", token)
        leading_p = lead_match.group(1) if lead_match else ""

        trail_match = re.search(r"([,;:—–\.\!\?…•\)\]\}]+)$", token)
        trailing_p = trail_match.group(1) if trail_match else ""

        clean_word = re.sub(r"^[(\[{]+|[,\;:—–\.\!\?…•\)\]\}]+$", "", token)
        ascii_word = strip_pali_diacritics(clean_word)
        align_word = re.sub(r"[^a-zA-Z]", "", ascii_word).upper()

        parsed.append(
            {
                "index": idx,
                "raw": token,
                "word": clean_word,
                "ascii_word": ascii_word,
                "align_word": align_word,
                "leading_punct": leading_p,
                "trailing_punct": trailing_p,
            }
        )

    return parsed


class TorchaudioAligner:
    """Native CTC forced aligner using torchaudio WAV2VEC2_ASR_BASE_960H."""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().to(self.device)
        self.model.eval()

        self.labels = self.bundle.get_labels()
        self.char_to_id = {c: i for i, c in enumerate(self.labels)}
        self.blank_id = self.char_to_id.get("-", 0)

    def align(self, wav_path: Path, parsed_tokens):
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.bundle.sample_rate)
            waveform = resampler(waveform)

        valid_tokens = [t for t in parsed_tokens if t["align_word"]]
        if not valid_tokens:
            return None

        target_words = [t["align_word"] for t in valid_tokens]
        full_target_str = "|".join(target_words)

        word_char_ranges = []
        curr_idx = 0
        for w in target_words:
            w_len = len(w)
            word_char_ranges.append((curr_idx, curr_idx + w_len - 1))
            curr_idx += w_len + 1

        target_ids = [self.char_to_id[c] for c in full_target_str if c in self.char_to_id]
        if not target_ids:
            return None

        targets = torch.tensor(target_ids, dtype=torch.int64, device=self.device)

        with torch.inference_mode():
            emissions, _ = self.model(waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)[0]

        num_frames = emissions.size(0)
        num_targets = targets.size(0)

        if num_frames < num_targets:
            return None

        # Build Trellis Alignment Matrix
        trellis = torch.full((num_frames + 1, num_targets + 1), -float("inf"), device=self.device)
        trellis[0, 0] = 0.0

        for t in range(num_frames):
            trellis[t + 1, 1:] = torch.maximum(
                trellis[t, 1:] + emissions[t, self.blank_id],
                trellis[t, :-1] + emissions[t, targets],
            )
            trellis[t + 1, 0] = trellis[t, 0] + emissions[t, self.blank_id]

        # Backtrack Optimal Path
        t, j = num_frames, num_targets
        path = []
        while t > 0 and j >= 0:
            p_stay = emissions[t - 1, self.blank_id] if j >= 0 else -float("inf")
            p_change = emissions[t - 1, targets[j - 1]] if j > 0 else -float("inf")

            stay_score = trellis[t - 1, j] + p_stay
            change_score = trellis[t - 1, j - 1] + p_change if j > 0 else -float("inf")

            if change_score >= stay_score and j > 0:
                path.append((t - 1, j - 1))
                j -= 1
            else:
                path.append((t - 1, j))
            t -= 1

        path.reverse()

        audio_duration_sec = waveform.shape[1] / self.bundle.sample_rate
        sec_per_frame = audio_duration_sec / num_frames

        word_spans = []
        for w_start, w_end in word_char_ranges:
            matched_frames = [f for f, c in path if w_start <= c <= w_end]
            if matched_frames:
                start_sec = min(matched_frames) * sec_per_frame
                end_sec = (max(matched_frames) + 1) * sec_per_frame
                word_spans.append((start_sec, end_sec))
            else:
                word_spans.append((0.0, 0.0))

        return word_spans, valid_tokens


def group_into_clauses(valid_tokens):
    """Groups words into continuous phrase/clause units bounded by punctuation."""
    clauses = []
    current_clause = []

    for token in valid_tokens:
        if token["leading_punct"] and current_clause:
            clauses.append(current_clause)
            current_clause = []

        current_clause.append(token)

        if token["trailing_punct"]:
            clauses.append(current_clause)
            current_clause = []

    if current_clause:
        clauses.append(current_clause)

    return clauses


def process_utterance(
    wav_path: Path,
    target_text: str,
    output_wav_path: Path,
    aligner: TorchaudioAligner,
):
    audio = AudioSegment.from_wav(wav_path)
    audio_len_ms = len(audio)
    parsed_tokens = parse_punctuated_words(target_text)

    alignment_result = aligner.align(wav_path, parsed_tokens)
    if alignment_result is None:
        return False

    word_spans, valid_tokens = alignment_result
    if not valid_tokens:
        return False

    token_to_span = {
        token["index"]: span for token, span in zip(valid_tokens, word_spans)
    }

    clauses = group_into_clauses(valid_tokens)
    new_audio = AudioSegment.silent(duration=0)
    last_end_ms = 0

    for clause in clauses:
        first_token = clause[0]
        last_token = clause[-1]

        start_sec = token_to_span[first_token["index"]][0]
        end_sec = token_to_span[last_token["index"]][1]

        if end_sec <= start_sec:
            continue

        raw_start_ms = int(start_sec * 1000)
        raw_end_ms = int(end_sec * 1000)

        # Apply safety acoustic padding (+35ms head, +45ms tail) to prevent clipping consonants
        start_ms = max(last_end_ms, raw_start_ms - 35)
        end_ms = min(audio_len_ms, raw_end_ms + 45)

        if end_ms <= start_ms:
            continue

        # Extract continuous clause audio slice and apply 10ms smooth fade to prevent clicks
        clause_slice = audio[start_ms:end_ms].fade_in(10).fade_out(10)

        # 1. Insert silence for opening punctuation (e.g., '(' or '[')
        if first_token["leading_punct"]:
            lead_pau_len = max(
                [PUNCT_TABLE.get(c, 200) for c in first_token["leading_punct"]]
            )
            new_audio += AudioSegment.silent(duration=lead_pau_len)

        # 2. Append intact phrase speech segment
        new_audio += clause_slice
        last_end_ms = end_ms

        # 3. Insert silence for trailing punctuation (e.g., ',', '.', ':')
        if last_token["trailing_punct"]:
            trail_pau_len = max(
                [PUNCT_TABLE.get(c, 200) for c in last_token["trailing_punct"]]
            )
            new_audio += AudioSegment.silent(duration=trail_pau_len)

    # Add final trailing room silence buffer (150ms)
    new_audio += AudioSegment.silent(duration=150)

    # Standardize output: 22050 Hz 16-bit Mono PCM WAV
    new_audio = new_audio.set_frame_rate(22050).set_channels(1)
    new_audio.export(output_wav_path, format="wav")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Repunctuate synthetic WAV audio at clause boundaries using CTC Forced Alignment."
    )
    parser.add_argument(
        "--metadata", "-m", type=Path, required=True, help="Path to metadata.csv"
    )
    parser.add_argument(
        "--wav-dir", "-w", type=Path, required=True, help="Input WAV directory"
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, required=True, help="Output WAV directory"
    )
    parser.add_argument(
        "--delimiter", "-d", type=str, default="|", help="CSV delimiter (default: '|')"
    )
    parser.add_argument(
        "--text-col", type=int, default=1, help="0-indexed text column position"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Wav2Vec2 CTC aligner on device: {args.device}...")
    aligner = TorchaudioAligner(device=args.device)

    with open(args.metadata, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} utterances from {args.metadata.name}...")
    success_count = 0
    fail_count = 0

    for line in tqdm(lines):
        parts = line.strip().split(args.delimiter)
        if len(parts) <= args.text_col:
            continue

        file_id = parts[0].replace(".wav", "")
        target_text = parts[args.text_col]

        input_wav = args.wav_dir / f"{file_id}.wav"
        output_wav = args.output_dir / f"{file_id}.wav"

        if not input_wav.exists():
            fail_count += 1
            continue

        try:
            ok = process_utterance(
                wav_path=input_wav,
                target_text=target_text,
                output_wav_path=output_wav,
                aligner=aligner,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"\nError processing {file_id}: {e}")
            fail_count += 1

    print(
        f"\nProcessing complete! Generated {success_count} files. (Failed/Skipped: {fail_count})"
    )


if __name__ == "__main__":
    main()